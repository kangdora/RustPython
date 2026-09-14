use super::{ExecutingFrame, FrameResult};
use crate::{
    AsObject, PyObject, PyResult, VirtualMachine,
    builtins::PyBaseExceptionRef,
    bytecode::{self, Label, OpArg},
};
use core::ffi::c_void;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use rustpython_tier2::{Env, HelperId, HelperTable, JitContext, compile};
use std::sync::OnceLock;

pub(crate) static COMPILED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static COMPILE_FAILED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static ENTERED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static ERRORS: AtomicUsize = AtomicUsize::new(0);

struct Ctx {
    frame: *mut c_void,
    vm: *const VirtualMachine,
    error: Option<PyBaseExceptionRef>,
}

fn with(
    ctx: *mut JitContext,
    f: impl FnOnce(&mut ExecutingFrame<'_>, &VirtualMachine) -> PyResult<i64>,
) -> i64 {
    let ctx = unsafe { &mut *(*ctx).user.cast::<Ctx>() };
    let frame = unsafe { &mut *ctx.frame.cast::<ExecutingFrame<'_>>() };
    let vm = unsafe { &*ctx.vm };
    match f(frame, vm) {
        Ok(v) => v,
        Err(e) => {
            ctx.error = Some(e);
            -1
        }
    }
}

fn unbound(frame: &ExecutingFrame<'_>, idx: usize, vm: &VirtualMachine) -> PyBaseExceptionRef {
    vm.new_unbound_local_error(format!(
        "local variable '{}' referenced before assignment",
        frame.code.varnames[idx]
    ))
}

extern "C" fn h_load_fast(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let idx = arg as usize;
        let x = f.localsplus.fastlocals()[idx]
            .clone()
            .ok_or_else(|| unbound(f, idx, vm))?;
        f.push_value(x);
        Ok(0)
    })
}

extern "C" fn h_load_fast_load_fast(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let idx1 = (arg >> 4) as usize;
        let idx2 = (arg & 15) as usize;
        let locals = f.localsplus.fastlocals();
        let x1 = locals[idx1].clone().ok_or_else(|| unbound(f, idx1, vm))?;
        let x2 = locals[idx2].clone().ok_or_else(|| unbound(f, idx2, vm))?;
        f.push_value(x1);
        f.push_value(x2);
        Ok(0)
    })
}

extern "C" fn h_store_fast(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let value = f.pop_value_opt();
        f.localsplus.fastlocals_mut()[arg as usize] = value;
        Ok(0)
    })
}

extern "C" fn h_store_fast_load_fast(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let value = f.pop_value_opt();
        let store_idx = (arg >> 4) as usize;
        let load_idx = (arg & 15) as usize;
        let load_value = {
            let locals = f.localsplus.fastlocals_mut();
            locals[store_idx] = value;
            locals[load_idx].clone()
        };
        f.push_value_opt(load_value);
        Ok(0)
    })
}

extern "C" fn h_store_fast_store_fast(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let idx1 = (arg >> 4) as usize;
        let idx2 = (arg & 15) as usize;
        let value1 = f.pop_value_opt();
        let value2 = f.pop_value_opt();
        let locals = f.localsplus.fastlocals_mut();
        locals[idx1] = value1;
        locals[idx2] = value2;
        Ok(0)
    })
}

extern "C" fn h_load_const(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let value = f.code.constants[(arg as u32).into()].clone();
        f.push_value(value.into());
        Ok(0)
    })
}

extern "C" fn h_load_small_int(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        f.push_value(vm.ctx.cached_int(arg as i32).to_owned().into());
        Ok(0)
    })
}

extern "C" fn h_load_global(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let name = &f.code.names[(arg >> 1) as usize];
        let x = f.load_global_or_builtin(name, vm)?;
        f.push_value(x);
        if arg & 1 != 0 {
            f.push_value_opt(None);
        }
        Ok(0)
    })
}

extern "C" fn h_binary_op(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let op = bytecode::BinaryOperator::try_from(arg as u32)
            .map_err(|_| vm.new_system_error("bad binary op in tier2 code"))?;
        f.execute_bin_op(vm, op)?;
        Ok(0)
    })
}

extern "C" fn h_compare_op(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        f.execute_compare(vm, OpArg::new(arg as u32))?;
        Ok(0)
    })
}

extern "C" fn h_to_bool(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        let b = obj.try_to_bool(vm)?;
        f.push_value(vm.ctx.new_bool(b).into());
        Ok(0)
    })
}

extern "C" fn h_unary_not(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        let b = obj.try_to_bool(vm)?;
        f.push_value(vm.ctx.new_bool(!b).into());
        Ok(0)
    })
}

extern "C" fn h_unary_negative(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        let value = vm._neg(&obj)?;
        f.push_value(value);
        Ok(0)
    })
}

extern "C" fn h_unary_invert(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        let value = vm._invert(&obj)?;
        f.push_value(value);
        Ok(0)
    })
}

extern "C" fn h_pop_is_true(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        Ok(obj.try_to_bool(vm)? as i64)
    })
}

extern "C" fn h_pop_is_none(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let obj = f.pop_value();
        Ok(vm.is_none(&obj) as i64)
    })
}

extern "C" fn h_pop_top(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        f.pop_stackref_opt();
        Ok(0)
    })
}

extern "C" fn h_copy(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let len = f.localsplus.stack_len();
        let value = f.localsplus.stack_index(len - arg as usize).cloned();
        f.push_stackref_opt(value);
        Ok(0)
    })
}

extern "C" fn h_swap(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, _vm| {
        let len = f.localsplus.stack_len();
        f.localsplus.stack_swap(len - 1, len - arg as usize);
        Ok(0)
    })
}

extern "C" fn h_for_iter(ctx: *mut JitContext, arg: u64) -> i64 {
    with(ctx, |f, vm| {
        let continued = f.execute_for_iter(vm, Label::from_u32(arg as u32))?;
        Ok((!continued) as i64)
    })
}

extern "C" fn h_eval_breaker(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |_f, vm| {
        if vm.eval_breaker_tripped() {
            vm.check_signals()?;
            #[cfg(feature = "threading")]
            vm.run_scheduled_gc();
        }
        Ok(0)
    })
}

extern "C" fn h_drop(_ctx: *mut JitContext, arg: u64) -> i64 {
    let ptr = arg as *mut PyObject;
    unsafe { PyObject::drop_at_zero(NonNull::new_unchecked(ptr)) };
    0
}

extern "C" fn h_unreachable(ctx: *mut JitContext, _arg: u64) -> i64 {
    with(ctx, |_f, vm| {
        Err(vm.new_system_error("tier2 helper not wired"))
    })
}

fn helper_table() -> &'static HelperTable {
    static TABLE: OnceLock<HelperTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = HelperTable::new(h_unreachable);
        t.set(HelperId::LoadFast, h_load_fast);
        t.set(HelperId::LoadFastLoadFast, h_load_fast_load_fast);
        t.set(HelperId::StoreFast, h_store_fast);
        t.set(HelperId::StoreFastLoadFast, h_store_fast_load_fast);
        t.set(HelperId::StoreFastStoreFast, h_store_fast_store_fast);
        t.set(HelperId::LoadConst, h_load_const);
        t.set(HelperId::LoadSmallInt, h_load_small_int);
        t.set(HelperId::LoadGlobal, h_load_global);
        t.set(HelperId::BinaryOp, h_binary_op);
        t.set(HelperId::CompareOp, h_compare_op);
        t.set(HelperId::ToBool, h_to_bool);
        t.set(HelperId::UnaryNot, h_unary_not);
        t.set(HelperId::UnaryNegative, h_unary_negative);
        t.set(HelperId::UnaryInvert, h_unary_invert);
        t.set(HelperId::PopIsTrue, h_pop_is_true);
        t.set(HelperId::PopIsNone, h_pop_is_none);
        t.set(HelperId::PopTop, h_pop_top);
        t.set(HelperId::Copy, h_copy);
        t.set(HelperId::Swap, h_swap);
        t.set(HelperId::ForIter, h_for_iter);
        t.set(HelperId::EvalBreaker, h_eval_breaker);
        t.set(HelperId::Drop, h_drop);
        t
    })
}

fn object_addr(obj: &PyObject) -> u64 {
    obj as *const PyObject as u64
}

impl ExecutingFrame<'_> {
    pub(super) fn tier2_run(&mut self, vm: &VirtualMachine) -> FrameResult {
        let code = self.code;
        let compiled = code.tier2_code.get_or_init(|| {
            let small_int = |i: i32| Some(object_addr(vm.ctx.cached_int(i).as_object()));
            let env = Env {
                refcount_offset: PyObject::refcount_offset() as i32,
                true_ptr: object_addr(vm.ctx.true_value.as_object()),
                false_ptr: object_addr(vm.ctx.false_value.as_object()),
                none_ptr: object_addr(vm.ctx.none.as_object()),
                small_int: &small_int,
            };
            let compiled = compile(&code.code.instructions, helper_table(), &env).ok();
            if compiled.is_some() {
                COMPILED.fetch_add(1, Relaxed);
            } else {
                COMPILE_FAILED.fetch_add(1, Relaxed);
            }
            compiled
        });
        let Some(compiled) = compiled else {
            return Ok(None);
        };
        let entry = self.lasti() as usize;
        if entry >= compiled.len() {
            return Ok(None);
        }
        ENTERED.fetch_add(1, Relaxed);
        let mut ctx = Ctx {
            frame: (self as *mut Self).cast::<c_void>(),
            vm,
            error: None,
        };
        let nlocalsplus = self.localsplus.nlocalsplus as u64;
        let locals = self.localsplus.data_as_mut_slice().as_mut_ptr();
        let stack_top: *mut u32 = &mut self.localsplus.stack_top;
        let mut jit_ctx = JitContext::new(
            (&mut ctx as *mut Ctx).cast::<c_void>(),
            locals,
            stack_top,
            nlocalsplus,
        );
        let lasti = unsafe { compiled.run(&mut jit_ctx, entry) };
        self.update_lasti(|i| *i = lasti);
        match ctx.error.take() {
            Some(exception) => {
                ERRORS.fetch_add(1, Relaxed);
                Err(exception)
            }
            None => Ok(None),
        }
    }
}
