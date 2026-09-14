use crate::asm::{Assembler, Cond, Label, Reg};
use crate::exec::{ARG0, ARG1, ExecError, ExecMemory, SHADOW_SPACE};
use core::ffi::c_void;
use rustpython_compiler_core::bytecode::{CodeUnit, Instruction, OpArgState};

#[repr(C)]
pub struct JitContext {
    pub user: *mut c_void,
    pub locals: *mut usize,
    pub stack_top: *mut u32,
    pub nlocalsplus: u64,
    /// Back-edges left before the eval breaker is consulted again.
    pub breaker_countdown: u64,
}

const CTX_LOCALS: i32 = 8;
const CTX_STACK_TOP: i32 = 16;
const CTX_NLOCALSPLUS: i32 = 24;
const CTX_BREAKER: i32 = 32;

const _: () = assert!(core::mem::size_of::<JitContext>() == 40);

/// How many back-edges run between two eval-breaker checks.
pub const BREAKER_INTERVAL: u64 = 64;

impl JitContext {
    pub fn new(
        user: *mut c_void,
        locals: *mut usize,
        stack_top: *mut u32,
        nlocalsplus: u64,
    ) -> Self {
        Self {
            user,
            locals,
            stack_top,
            nlocalsplus,
            breaker_countdown: 1,
        }
    }
}

pub type HelperFn = extern "C" fn(*mut JitContext, u64) -> i64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum HelperId {
    LoadFast,
    LoadFastLoadFast,
    StoreFast,
    StoreFastLoadFast,
    StoreFastStoreFast,
    LoadConst,
    LoadSmallInt,
    LoadGlobal,
    BinaryOp,
    CompareOp,
    ToBool,
    UnaryNot,
    UnaryNegative,
    UnaryInvert,
    PopIsTrue,
    PopIsNone,
    PopTop,
    Copy,
    Swap,
    ForIter,
    EvalBreaker,
    Drop,
    /// Compare the two raw stack operands without popping; returns 0/1, or
    /// `FAST_PATH_MISS` when the operands are not both plain ints.
    CompareFast,
    /// Arithmetic on two int operands; on success the helper has already
    /// replaced them with the result. Returns `FAST_PATH_MISS` otherwise.
    BinaryOpIntFast,
}

impl HelperId {
    pub const COUNT: usize = HelperId::BinaryOpIntFast as usize + 1;
}

pub const FAST_PATH_MISS: i64 = 2;

#[derive(Clone)]
pub struct HelperTable {
    fns: [HelperFn; HelperId::COUNT],
}

impl HelperTable {
    pub fn new(default: HelperFn) -> Self {
        Self {
            fns: [default; HelperId::COUNT],
        }
    }

    pub fn set(&mut self, id: HelperId, f: HelperFn) {
        self.fns[id as usize] = f;
    }

    fn address(&self, id: HelperId) -> u64 {
        self.fns[id as usize] as *const () as u64
    }
}

/// Object-model facts the generated code bakes in.
pub struct Env<'a> {
    /// Byte offset of the strong-count word inside an object.
    pub refcount_offset: i32,
    pub true_ptr: u64,
    pub false_ptr: u64,
    pub none_ptr: u64,
    /// Address of the interned small int for `i`, if one exists.
    pub small_int: &'a dyn Fn(i32) -> Option<u64>,
}

pub const BORROW_TAG: u64 = 1;
const LEAKED_SHIFT: u8 = 61;
const FLAG_BITS: u8 = 3;

#[derive(Debug)]
pub enum CompileError {
    Exec(ExecError),
}

impl From<ExecError> for CompileError {
    fn from(e: ExecError) -> Self {
        Self::Exec(e)
    }
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Exec(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompileError {}

pub struct Compiled {
    mem: ExecMemory,
    offsets: Vec<u32>,
}

impl Compiled {
    /// # Safety
    /// `ctx` must describe a live frame for the whole run and every helper in
    /// the table used at compile time must accept it.
    pub unsafe fn run(&self, ctx: *mut JitContext, entry: usize) -> u32 {
        let target = unsafe { self.mem.as_ptr().add(self.offsets[entry] as usize) };
        let f: extern "C" fn(*mut JitContext, *const u8) -> u64 =
            unsafe { core::mem::transmute(self.mem.as_ptr()) };
        f(ctx, target) as u32
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

const CTX: Reg = Reg::Rbx;
const TOP: Reg = Reg::R12;
const LOCALS: Reg = Reg::R13;
const STACK: Reg = Reg::R14;
const SCRATCH: Reg = Reg::R15;
const FRAME_SPACE: i32 = SHADOW_SPACE;

struct Emitter<'a> {
    a: Assembler,
    helpers: &'a HelperTable,
    env: &'a Env<'a>,
    labels: Vec<Label>,
    epilogue: Label,
    len: usize,
}

impl Emitter<'_> {
    fn exit(&mut self, lasti: usize) {
        self.a.mov_ri(Reg::Rax, lasti as u64);
        self.a.jmp(self.epilogue);
    }

    fn sync_out(&mut self) {
        self.a.mov_rm(Reg::Rcx, CTX, CTX_STACK_TOP);
        self.a.mov_mr32(Reg::Rcx, 0, TOP);
    }

    fn sync_in(&mut self) {
        self.a.mov_rm(Reg::Rcx, CTX, CTX_STACK_TOP);
        self.a.mov_rm32(TOP, Reg::Rcx, 0);
    }

    fn call_raw(&mut self, id: HelperId) {
        self.a.mov_rr(ARG0, CTX);
        self.a.mov_ri(Reg::Rax, self.helpers.address(id));
        self.a.call_r(Reg::Rax);
    }

    fn call(&mut self, id: HelperId, arg: u64, idx: usize) {
        self.sync_out();
        self.a.mov_ri(ARG1, arg);
        self.call_raw(id);
        self.sync_in();
        self.a.test_rr(Reg::Rax, Reg::Rax);
        let ok = self.a.new_label();
        self.a.jcc(Cond::NS, ok);
        self.exit(idx + 1);
        self.a.bind(ok);
    }

    fn call_with_reg_arg(&mut self, id: HelperId, arg: Reg) {
        self.sync_out();
        self.a.mov_rr(ARG1, arg);
        self.call_raw(id);
        self.sync_in();
    }

    fn push(&mut self, r: Reg) {
        self.a.mov_mr_idx8(STACK, TOP, 0, r);
        self.a.add_ri(TOP, 1);
    }

    fn pop(&mut self, r: Reg) {
        self.a.sub_ri(TOP, 1);
        self.a.mov_rm_idx8(r, STACK, TOP, 0);
    }

    fn peek(&mut self, r: Reg) {
        self.a.mov_rm_idx8(r, STACK, TOP, -8);
    }

    fn incref(&mut self, obj: Reg) {
        self.a.lock_add_mi(obj, self.env.refcount_offset, 1);
    }

    /// Decrement the strong count of the object in `rax`; hands the object to
    /// the drop helper when the count reaches zero. Clobbers rcx and rdx.
    fn decref_rax(&mut self) {
        let skip = self.a.new_label();
        self.a.mov_ri(Reg::Rdx, u64::MAX);
        self.a
            .lock_xadd(Reg::Rax, self.env.refcount_offset, Reg::Rdx);
        self.a.mov_rr(Reg::Rcx, Reg::Rdx);
        self.a.shr_ri(Reg::Rcx, LEAKED_SHIFT);
        self.a.and_ri(Reg::Rcx, 1);
        self.a.jcc(Cond::NE, skip);
        self.a.shl_ri(Reg::Rdx, FLAG_BITS);
        self.a.shr_ri(Reg::Rdx, FLAG_BITS);
        self.a.cmp_ri(Reg::Rdx, 1);
        self.a.jcc(Cond::NE, skip);
        self.call_with_reg_arg(HelperId::Drop, Reg::Rax);
        self.a.bind(skip);
    }

    /// Release a stack value held in `rax`: borrowed and NULL values are
    /// dropped without touching any count.
    fn discard_rax(&mut self) {
        let done = self.a.new_label();
        self.a.mov_rr(Reg::Rcx, Reg::Rax);
        self.a.and_ri(Reg::Rcx, BORROW_TAG as i32);
        self.a.jcc(Cond::NE, done);
        self.a.test_rr(Reg::Rax, Reg::Rax);
        self.a.jcc(Cond::E, done);
        self.decref_rax();
        self.a.bind(done);
    }

    /// Turn the stack value in `rax` into an owned reference.
    fn promote_rax(&mut self) {
        let owned = self.a.new_label();
        self.a.mov_rr(Reg::Rcx, Reg::Rax);
        self.a.and_ri(Reg::Rcx, BORROW_TAG as i32);
        self.a.jcc(Cond::E, owned);
        self.a.and_ri(Reg::Rax, -2);
        self.incref(Reg::Rax);
        self.a.bind(owned);
    }

    fn load_fast(&mut self, var: u32, idx: usize) {
        let slow = self.a.new_label();
        let done = self.a.new_label();
        self.a.mov_rm(Reg::Rax, LOCALS, var as i32 * 8);
        self.a.test_rr(Reg::Rax, Reg::Rax);
        self.a.jcc(Cond::E, slow);
        self.incref(Reg::Rax);
        self.push(Reg::Rax);
        self.a.jmp(done);
        self.a.bind(slow);
        self.call(HelperId::LoadFast, var.into(), idx);
        self.a.bind(done);
    }

    fn store_fast(&mut self, var: u32) {
        let done = self.a.new_label();
        self.pop(Reg::Rax);
        self.promote_rax();
        self.a.mov_rm(Reg::Rcx, LOCALS, var as i32 * 8);
        self.a.mov_mr(LOCALS, var as i32 * 8, Reg::Rax);
        self.a.mov_rr(Reg::Rax, Reg::Rcx);
        self.a.test_rr(Reg::Rax, Reg::Rax);
        self.a.jcc(Cond::E, done);
        self.decref_rax();
        self.a.bind(done);
    }

    fn pop_top(&mut self) {
        self.pop(Reg::Rax);
        self.discard_rax();
    }

    fn push_bool_from_scratch(&mut self) {
        let is_true = self.a.new_label();
        let done = self.a.new_label();
        self.a.cmp_ri(SCRATCH, 1);
        self.a.jcc(Cond::E, is_true);
        self.a.mov_ri(Reg::Rax, self.env.false_ptr | BORROW_TAG);
        self.push(Reg::Rax);
        self.a.jmp(done);
        self.a.bind(is_true);
        self.a.mov_ri(Reg::Rax, self.env.true_ptr | BORROW_TAG);
        self.push(Reg::Rax);
        self.a.bind(done);
    }

    /// Compare through the fast helper; on a miss fall back to the generic
    /// helper. `fused` carries the conditional jump that consumes the result,
    /// as (jump_on, target, index after the jump).
    fn compare_op(&mut self, arg: u32, idx: usize, fused: Option<(bool, usize, usize)>) {
        let miss = self.a.new_label();
        let done = self.a.new_label();
        self.call(HelperId::CompareFast, arg.into(), idx);
        self.a.cmp_ri(Reg::Rax, FAST_PATH_MISS as i32);
        self.a.jcc(Cond::E, miss);
        self.a.mov_rr(SCRATCH, Reg::Rax);
        self.pop_top();
        self.pop_top();
        match fused {
            Some((jump_on, target, _)) => {
                self.a.cmp_ri(SCRATCH, 1);
                self.branch_to(if jump_on { Cond::E } else { Cond::NE }, target);
            }
            None => self.push_bool_from_scratch(),
        }
        self.a.jmp(done);
        self.a.bind(miss);
        self.call(HelperId::CompareOp, arg.into(), idx);
        if let Some((jump_on, target, _)) = fused {
            self.call(HelperId::PopIsTrue, 0, idx);
            self.branch_if(if jump_on { Cond::E } else { Cond::NE }, target);
        }
        self.a.bind(done);
        if let Some((_, _, after)) = fused {
            self.jump(after);
        }
    }

    fn eval_breaker(&mut self, idx: usize) {
        let skip = self.a.new_label();
        self.a.sub_mi(CTX, CTX_BREAKER, 1);
        self.a.jcc(Cond::NE, skip);
        self.call(HelperId::EvalBreaker, 0, idx);
        self.a.mov_mi32(CTX, CTX_BREAKER, BREAKER_INTERVAL as i32);
        self.a.bind(skip);
    }

    fn binary_op_int(&mut self, arg: u32, idx: usize) {
        let done = self.a.new_label();
        self.call(HelperId::BinaryOpIntFast, arg.into(), idx);
        self.a.cmp_ri(Reg::Rax, FAST_PATH_MISS as i32);
        self.a.jcc(Cond::NE, done);
        self.call(HelperId::BinaryOp, arg.into(), idx);
        self.a.bind(done);
    }

    fn branch_to(&mut self, cond: Cond, target: usize) {
        if target < self.len {
            self.a.jcc(cond, self.labels[target]);
        } else {
            let skip = self.a.new_label();
            let flipped = match cond {
                Cond::E => Cond::NE,
                _ => Cond::E,
            };
            self.a.jcc(flipped, skip);
            self.exit(target);
            self.a.bind(skip);
        }
    }

    fn load_small_int(&mut self, i: u32, idx: usize) {
        match (self.env.small_int)(i as i32) {
            Some(ptr) => {
                self.a.mov_ri(Reg::Rax, ptr | BORROW_TAG);
                self.push(Reg::Rax);
            }
            None => self.call(HelperId::LoadSmallInt, i.into(), idx),
        }
    }

    fn jump(&mut self, target: usize) {
        if target < self.len {
            self.a.jmp(self.labels[target]);
        } else {
            self.exit(target);
        }
    }

    fn branch_if(&mut self, cond: Cond, target: usize) {
        self.a.cmp_ri(Reg::Rax, 1);
        self.branch_to(cond, target);
    }

    /// Pop a value and branch to `target` when it is the bool singleton
    /// selected by `jump_on`; anything else goes through the helper.
    fn pop_jump_if_bool(&mut self, jump_on: bool, target: usize, idx: usize) {
        let is_true = self.a.new_label();
        let is_false = self.a.new_label();
        let slow = self.a.new_label();
        let done = self.a.new_label();
        self.peek(Reg::Rax);
        self.a.mov_rr(Reg::Rcx, Reg::Rax);
        self.a.and_ri(Reg::Rcx, -2);
        self.a.mov_ri(Reg::Rdx, self.env.true_ptr);
        self.a.cmp_rr(Reg::Rcx, Reg::Rdx);
        self.a.jcc(Cond::E, is_true);
        self.a.mov_ri(Reg::Rdx, self.env.false_ptr);
        self.a.cmp_rr(Reg::Rcx, Reg::Rdx);
        self.a.jcc(Cond::E, is_false);
        self.a.jmp(slow);

        let (taken, not_taken) = if jump_on {
            (is_true, is_false)
        } else {
            (is_false, is_true)
        };
        self.a.bind(taken);
        self.a.sub_ri(TOP, 1);
        self.discard_rax();
        self.jump(target);
        self.a.bind(not_taken);
        self.a.sub_ri(TOP, 1);
        self.discard_rax();
        self.a.jmp(done);

        self.a.bind(slow);
        self.call(HelperId::PopIsTrue, 0, idx);
        self.branch_if(if jump_on { Cond::E } else { Cond::NE }, target);
        self.a.bind(done);
    }

    fn pop_jump_if_none(&mut self, jump_on_none: bool, target: usize) {
        let matched = self.a.new_label();
        let not_matched = self.a.new_label();
        let done = self.a.new_label();
        self.pop(Reg::Rax);
        self.a.mov_rr(Reg::Rcx, Reg::Rax);
        self.a.and_ri(Reg::Rcx, -2);
        self.a.mov_ri(Reg::Rdx, self.env.none_ptr);
        self.a.cmp_rr(Reg::Rcx, Reg::Rdx);
        self.a.jcc(Cond::NE, not_matched);
        self.discard_rax();
        self.a.jmp(matched);
        self.a.bind(not_matched);
        self.discard_rax();

        let not_matched_done = self.a.new_label();
        self.a.jmp(not_matched_done);
        if jump_on_none {
            self.a.bind(matched);
            self.jump(target);
            self.a.bind(not_matched_done);
        } else {
            self.a.bind(matched);
            self.a.jmp(done);
            self.a.bind(not_matched_done);
            self.jump(target);
        }
        self.a.bind(done);
    }
}

pub fn compile(
    units: &[CodeUnit],
    helpers: &HelperTable,
    env: &Env<'_>,
) -> Result<Compiled, CompileError> {
    let mut a = Assembler::new();
    let labels: Vec<Label> = (0..units.len()).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();

    a.push(CTX);
    a.push(TOP);
    a.push(LOCALS);
    a.push(STACK);
    a.push(SCRATCH);
    if FRAME_SPACE > 0 {
        a.sub_ri(Reg::Rsp, FRAME_SPACE);
    }
    a.mov_rr(CTX, ARG0);
    a.mov_rm(LOCALS, CTX, CTX_LOCALS);
    a.mov_rm(Reg::Rax, CTX, CTX_STACK_TOP);
    a.mov_rm32(TOP, Reg::Rax, 0);
    a.mov_rm(STACK, CTX, CTX_NLOCALSPLUS);
    a.shl_ri(STACK, 3);
    a.add_rr(STACK, LOCALS);
    a.jmp_r(ARG1);

    let mut e = Emitter {
        a,
        helpers,
        env,
        labels,
        epilogue,
        len: units.len(),
    };

    let mut arg_state = OpArgState::default();
    let mut idx = 0;
    let mut fused_jump: Option<usize> = None;
    while idx < units.len() {
        e.a.bind(e.labels[idx]);
        if fused_jump == Some(idx) {
            fused_jump = None;
            let caches = units[idx].op.cache_entries();
            e.exit(idx);
            for skipped in idx + 1..(idx + 1 + caches).min(units.len()) {
                e.a.bind(e.labels[skipped]);
            }
            idx += 1 + caches;
            continue;
        }
        let unit = units[idx];
        let (raw_op, arg) = arg_state.get(unit);
        let op = raw_op.deoptimize();
        let caches = raw_op.cache_entries();
        let next = idx + 1 + caches;
        let argv = u32::from(arg);
        let forward = |delta: u32| next + delta as usize;
        let backward = |delta: u32| next - delta as usize;

        match op {
            Instruction::ExtendedArg => {
                idx += 1;
                continue;
            }
            Instruction::Cache | Instruction::Nop | Instruction::NotTaken => {}
            Instruction::LoadFast { .. }
            | Instruction::LoadFastBorrow { .. }
            | Instruction::LoadFastCheck { .. } => e.load_fast(argv, idx),
            Instruction::LoadFastLoadFast { .. }
            | Instruction::LoadFastBorrowLoadFastBorrow { .. } => {
                e.load_fast(argv >> 4, idx);
                e.load_fast(argv & 15, idx);
            }
            Instruction::StoreFast { .. } => e.store_fast(argv),
            Instruction::StoreFastLoadFast { .. } => {
                e.store_fast(argv >> 4);
                e.load_fast(argv & 15, idx);
            }
            Instruction::StoreFastStoreFast { .. } => {
                e.store_fast(argv >> 4);
                e.store_fast(argv & 15);
            }
            Instruction::LoadConst { .. } => e.call(HelperId::LoadConst, argv.into(), idx),
            Instruction::LoadSmallInt { .. } => e.load_small_int(argv, idx),
            Instruction::LoadGlobal { .. } => e.call(HelperId::LoadGlobal, argv.into(), idx),
            Instruction::BinaryOp { .. } => match raw_op {
                Instruction::BinaryOpAddInt
                | Instruction::BinaryOpSubtractInt
                | Instruction::BinaryOpMultiplyInt => e.binary_op_int(argv, idx),
                _ => e.call(HelperId::BinaryOp, argv.into(), idx),
            },
            Instruction::CompareOp { .. } => {
                let fused = units.get(next).and_then(|pj| {
                    let jump_on = match pj.op.deoptimize() {
                        Instruction::PopJumpIfFalse { .. } => false,
                        Instruction::PopJumpIfTrue { .. } => true,
                        _ => return None,
                    };
                    let after = next + 1 + pj.op.cache_entries();
                    Some((jump_on, after + pj.arg.as_u32() as usize, after))
                });
                if fused.is_some() {
                    fused_jump = Some(next);
                }
                e.compare_op(argv, idx, fused);
            }
            Instruction::ToBool => e.call(HelperId::ToBool, 0, idx),
            Instruction::UnaryNot => e.call(HelperId::UnaryNot, 0, idx),
            Instruction::UnaryNegative => e.call(HelperId::UnaryNegative, 0, idx),
            Instruction::UnaryInvert => e.call(HelperId::UnaryInvert, 0, idx),
            Instruction::PopTop | Instruction::EndFor | Instruction::PopIter => e.pop_top(),
            Instruction::Copy { .. } => e.call(HelperId::Copy, argv.into(), idx),
            Instruction::Swap { .. } => e.call(HelperId::Swap, argv.into(), idx),
            Instruction::PopJumpIfFalse { .. } => e.pop_jump_if_bool(false, forward(argv), idx),
            Instruction::PopJumpIfTrue { .. } => e.pop_jump_if_bool(true, forward(argv), idx),
            Instruction::PopJumpIfNone { .. } => e.pop_jump_if_none(true, forward(argv)),
            Instruction::PopJumpIfNotNone { .. } => e.pop_jump_if_none(false, forward(argv)),
            Instruction::ForIter { .. } => {
                let mut target = forward(argv);
                e.call(HelperId::ForIter, target as u64, idx);
                if matches!(
                    units.get(target).map(|u| u.op),
                    Some(Instruction::EndFor | Instruction::InstrumentedEndFor)
                ) {
                    target += 1;
                }
                e.branch_if(Cond::E, target);
            }
            Instruction::JumpForward { .. } => e.jump(forward(argv)),
            Instruction::JumpBackward { .. } => {
                e.eval_breaker(idx);
                e.jump(backward(argv));
            }
            Instruction::JumpBackwardNoInterrupt { .. } => e.jump(backward(argv)),
            _ => e.exit(idx),
        }
        for skipped in idx + 1..next.min(units.len()) {
            e.a.bind(e.labels[skipped]);
        }
        idx = next;
    }

    e.exit(units.len());
    e.a.bind(e.epilogue);
    e.a.mov_rm(Reg::Rcx, CTX, CTX_STACK_TOP);
    e.a.mov_mr32(Reg::Rcx, 0, TOP);
    if FRAME_SPACE > 0 {
        e.a.add_ri(Reg::Rsp, FRAME_SPACE);
    }
    e.a.pop(SCRATCH);
    e.a.pop(STACK);
    e.a.pop(LOCALS);
    e.a.pop(TOP);
    e.a.pop(CTX);
    e.a.ret();

    let offsets = e
        .labels
        .iter()
        .map(|l| e.a.label_offset(*l).expect("every unit is bound") as u32)
        .collect();
    let mem = ExecMemory::new(&e.a.finish())?;
    Ok(Compiled { mem, offsets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use rustpython_compiler_core::bytecode::{
        Arg, BinaryOperator, CodeUnit, ComparisonOperator, Instruction, OpArgByte,
    };

    const LEAKED: usize = 1 << 61;

    #[repr(C)]
    struct Obj {
        rc: AtomicUsize,
    }

    impl Obj {
        fn new(rc: usize) -> Box<Self> {
            Box::new(Self {
                rc: AtomicUsize::new(rc),
            })
        }

        fn ptr(&self) -> u64 {
            self as *const Self as u64
        }

        fn rc(&self) -> usize {
            self.rc.load(Relaxed)
        }
    }

    struct Frame {
        data: Vec<usize>,
        stack_top: u32,
        nlocalsplus: usize,
    }

    impl Frame {
        fn new(nlocalsplus: usize, stacksize: usize) -> Self {
            Self {
                data: vec![0; nlocalsplus + stacksize],
                stack_top: 0,
                nlocalsplus,
            }
        }

        fn push(&mut self, v: u64) {
            self.data[self.nlocalsplus + self.stack_top as usize] = v as usize;
            self.stack_top += 1;
        }

        fn stack(&self) -> &[usize] {
            &self.data[self.nlocalsplus..self.nlocalsplus + self.stack_top as usize]
        }
    }

    struct Recorder {
        calls: Vec<(HelperId, u64)>,
        script: Vec<i64>,
    }

    thread_local! {
        static REC: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    }

    fn record(id: HelperId, arg: u64) -> i64 {
        REC.with(|r| {
            let mut r = r.borrow_mut();
            let rec = r.as_mut().unwrap();
            rec.calls.push((id, arg));
            if rec.script.is_empty() {
                0
            } else {
                rec.script.remove(0)
            }
        })
    }

    macro_rules! fake {
        ($name:ident, $id:expr) => {
            extern "C" fn $name(_ctx: *mut JitContext, arg: u64) -> i64 {
                record($id, arg)
            }
        };
    }

    fake!(f_load_fast, HelperId::LoadFast);
    fake!(f_store_fast, HelperId::StoreFast);
    fake!(f_load_small_int, HelperId::LoadSmallInt);
    fake!(f_load_const, HelperId::LoadConst);
    fake!(f_compare_op, HelperId::CompareOp);
    /// Pops the tested value through the shared stack_top word, like the
    /// real helper does.
    extern "C" fn f_pop_is_true(ctx: *mut JitContext, arg: u64) -> i64 {
        unsafe {
            let top = &mut *(*ctx).stack_top;
            *top -= 1;
        }
        record(HelperId::PopIsTrue, arg)
    }
    fake!(f_pop_is_none, HelperId::PopIsNone);
    fake!(f_pop_top, HelperId::PopTop);
    fake!(f_for_iter, HelperId::ForIter);
    fake!(f_eval_breaker, HelperId::EvalBreaker);
    fake!(f_drop, HelperId::Drop);
    fake!(f_compare_fast, HelperId::CompareFast);
    fake!(f_to_bool, HelperId::ToBool);
    fake!(f_generic, HelperId::LoadGlobal);

    /// Pretends the int fast path succeeded: drops one operand slot.
    extern "C" fn f_binary_op_int_fast(ctx: *mut JitContext, arg: u64) -> i64 {
        let r = record(HelperId::BinaryOpIntFast, arg);
        if r == 0 {
            unsafe {
                let top = &mut *(*ctx).stack_top;
                *top -= 1;
            }
        }
        r
    }

    /// Pretends to pop two operands and push one result, through the shared
    /// stack_top word like a real helper would.
    extern "C" fn f_binary_op(ctx: *mut JitContext, arg: u64) -> i64 {
        unsafe {
            let top = &mut *(*ctx).stack_top;
            *top -= 1;
        }
        record(HelperId::BinaryOp, arg)
    }

    fn table() -> HelperTable {
        let mut t = HelperTable::new(f_generic);
        t.set(HelperId::LoadFast, f_load_fast);
        t.set(HelperId::StoreFast, f_store_fast);
        t.set(HelperId::LoadSmallInt, f_load_small_int);
        t.set(HelperId::LoadConst, f_load_const);
        t.set(HelperId::BinaryOp, f_binary_op);
        t.set(HelperId::CompareOp, f_compare_op);
        t.set(HelperId::PopIsTrue, f_pop_is_true);
        t.set(HelperId::PopIsNone, f_pop_is_none);
        t.set(HelperId::PopTop, f_pop_top);
        t.set(HelperId::ForIter, f_for_iter);
        t.set(HelperId::EvalBreaker, f_eval_breaker);
        t.set(HelperId::Drop, f_drop);
        t.set(HelperId::CompareFast, f_compare_fast);
        t.set(HelperId::ToBool, f_to_bool);
        t.set(HelperId::BinaryOpIntFast, f_binary_op_int_fast);
        t
    }

    struct Singletons {
        t: Box<Obj>,
        f: Box<Obj>,
        n: Box<Obj>,
        five: Box<Obj>,
    }

    fn singletons() -> Singletons {
        Singletons {
            t: Obj::new(LEAKED | 1),
            f: Obj::new(LEAKED | 1),
            n: Obj::new(LEAKED | 1),
            five: Obj::new(LEAKED | 1),
        }
    }

    fn run_with(
        units: &[CodeUnit],
        entry: usize,
        script: Vec<i64>,
        frame: &mut Frame,
        s: &Singletons,
    ) -> (u32, Vec<(HelperId, u64)>) {
        REC.with(|r| {
            *r.borrow_mut() = Some(Recorder {
                calls: vec![],
                script,
            })
        });
        let five = s.five.ptr();
        let small_int = move |i: i32| if i == 5 { Some(five) } else { None };
        let env = Env {
            refcount_offset: 0,
            true_ptr: s.t.ptr(),
            false_ptr: s.f.ptr(),
            none_ptr: s.n.ptr(),
            small_int: &small_int,
        };
        let compiled = compile(units, &table(), &env).expect("compile");
        let mut ctx = JitContext::new(
            core::ptr::null_mut(),
            frame.data.as_mut_ptr(),
            &mut frame.stack_top,
            frame.nlocalsplus as u64,
        );
        let lasti = unsafe { compiled.run(&mut ctx, entry) };
        let calls = REC.with(|r| r.borrow_mut().take().unwrap().calls);
        (lasti, calls)
    }

    fn run(units: &[CodeUnit], entry: usize, script: Vec<i64>) -> (u32, Vec<(HelperId, u64)>) {
        let mut frame = Frame::new(4, 8);
        let s = singletons();
        run_with(units, entry, script, &mut frame, &s)
    }

    fn u(op: Instruction, arg: u8) -> CodeUnit {
        CodeUnit::new(op, OpArgByte::new(arg))
    }

    fn caches(op: Instruction) -> u8 {
        op.cache_entries() as u8
    }

    fn with_caches(op: Instruction, arg: u8) -> Vec<CodeUnit> {
        let mut v = vec![u(op, arg)];
        for _ in 0..caches(op) {
            v.push(u(Instruction::Cache, 0));
        }
        v
    }

    const LOAD_FAST: Instruction = Instruction::LoadFast {
        var_num: Arg::marker(),
    };
    const STORE_FAST: Instruction = Instruction::StoreFast {
        var_num: Arg::marker(),
    };
    const LOAD_SMALL_INT: Instruction = Instruction::LoadSmallInt { i: Arg::marker() };
    const BINARY_OP: Instruction = Instruction::BinaryOp { op: Arg::marker() };
    const POP_JUMP_IF_FALSE: Instruction = Instruction::PopJumpIfFalse {
        delta: Arg::marker(),
    };
    const POP_JUMP_IF_TRUE: Instruction = Instruction::PopJumpIfTrue {
        delta: Arg::marker(),
    };
    const POP_JUMP_IF_NONE: Instruction = Instruction::PopJumpIfNone {
        delta: Arg::marker(),
    };
    const POP_JUMP_IF_NOT_NONE: Instruction = Instruction::PopJumpIfNotNone {
        delta: Arg::marker(),
    };
    const JUMP_BACKWARD: Instruction = Instruction::JumpBackward {
        delta: Arg::marker(),
    };
    const FOR_ITER: Instruction = Instruction::ForIter {
        delta: Arg::marker(),
    };
    const CALL: Instruction = Instruction::Call {
        argc: Arg::marker(),
    };
    const RETURN: Instruction = Instruction::ReturnValue;

    fn ids(calls: &[(HelperId, u64)]) -> Vec<HelperId> {
        calls.iter().map(|c| c.0).collect()
    }

    #[test]
    fn load_fast_pushes_owned_clone_without_helper() {
        let a = Obj::new(1);
        let mut frame = Frame::new(2, 4);
        frame.data[0] = a.ptr() as usize;
        let s = singletons();
        let units = [u(LOAD_FAST, 0), u(RETURN, 0)];
        let (lasti, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(lasti, 1);
        assert!(calls.is_empty());
        assert_eq!(frame.stack(), [a.ptr() as usize]);
        assert_eq!(a.rc(), 2);
    }

    #[test]
    fn load_fast_of_empty_slot_goes_to_helper() {
        let units = [u(LOAD_FAST, 1), u(RETURN, 0)];
        let (lasti, calls) = run(&units, 0, vec![-1]);
        assert_eq!(lasti, 1);
        assert_eq!(calls, [(HelperId::LoadFast, 1)]);
    }

    #[test]
    fn store_fast_replaces_slot_and_releases_old_value() {
        let a = Obj::new(1);
        let b = Obj::new(2);
        let mut frame = Frame::new(2, 4);
        frame.data[0] = b.ptr() as usize;
        frame.push(a.ptr());
        let s = singletons();
        let units = [u(STORE_FAST, 0), u(RETURN, 0)];
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert!(calls.is_empty());
        assert_eq!(frame.stack_top, 0);
        assert_eq!(frame.data[0], a.ptr() as usize);
        assert_eq!(a.rc(), 1);
        assert_eq!(b.rc(), 1);
    }

    #[test]
    fn store_fast_drops_old_value_when_its_count_hits_zero() {
        let a = Obj::new(1);
        let b = Obj::new(1);
        let mut frame = Frame::new(2, 4);
        frame.data[0] = b.ptr() as usize;
        frame.push(a.ptr());
        let s = singletons();
        let units = [u(STORE_FAST, 0), u(RETURN, 0)];
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(calls, [(HelperId::Drop, b.ptr())]);
        assert_eq!(b.rc(), 0);
    }

    #[test]
    fn store_fast_never_drops_leaked_objects() {
        let a = Obj::new(1);
        let b = Obj::new(LEAKED | 1);
        let mut frame = Frame::new(2, 4);
        frame.data[0] = b.ptr() as usize;
        frame.push(a.ptr());
        let s = singletons();
        let units = [u(STORE_FAST, 0), u(RETURN, 0)];
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert!(calls.is_empty());
        assert_eq!(b.rc(), LEAKED);
    }

    #[test]
    fn store_fast_promotes_borrowed_values() {
        let a = Obj::new(1);
        let mut frame = Frame::new(2, 4);
        frame.push(a.ptr() | BORROW_TAG);
        let s = singletons();
        let units = [u(STORE_FAST, 1), u(RETURN, 0)];
        run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(frame.data[1], a.ptr() as usize);
        assert_eq!(a.rc(), 2);
    }

    #[test]
    fn pop_top_releases_owned_but_not_borrowed_or_null() {
        let a = Obj::new(1);
        let b = Obj::new(1);
        let mut frame = Frame::new(1, 4);
        frame.push(a.ptr());
        frame.push(b.ptr() | BORROW_TAG);
        frame.push(0);
        let s = singletons();
        let units = [
            u(Instruction::PopTop, 0),
            u(Instruction::PopTop, 0),
            u(Instruction::PopTop, 0),
            u(RETURN, 0),
        ];
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(calls, [(HelperId::Drop, a.ptr())]);
        assert_eq!(frame.stack_top, 0);
        assert_eq!(b.rc(), 1);
    }

    #[test]
    fn load_small_int_pushes_borrowed_pointer_when_interned() {
        let mut frame = Frame::new(1, 4);
        let s = singletons();
        let units = [u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 7), u(RETURN, 0)];
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(calls, [(HelperId::LoadSmallInt, 7)]);
        assert_eq!(frame.stack(), [(s.five.ptr() | BORROW_TAG) as usize]);
    }

    fn branchy(op: Instruction) -> (Vec<CodeUnit>, u32) {
        let mut v = vec![];
        let pj = v.len();
        v.extend(with_caches(op, 0));
        v.push(u(LOAD_SMALL_INT, 5));
        v.push(u(Instruction::PopTop, 0));
        let ret = v.len();
        v.push(u(RETURN, 0));
        let after = pj + 1 + caches(op) as usize;
        v[pj].arg = OpArgByte::new((ret - after) as u8);
        (v, ret as u32)
    }

    #[test]
    fn pop_jump_if_false_on_bool_singletons_needs_no_helper() {
        let s = singletons();
        let (units, ret) = branchy(POP_JUMP_IF_FALSE);
        for (value, expect_body) in [
            (s.f.ptr() | BORROW_TAG, false),
            (s.t.ptr() | BORROW_TAG, true),
            (s.f.ptr(), false),
            (s.t.ptr(), true),
        ] {
            let mut frame = Frame::new(1, 4);
            frame.push(value);
            let (lasti, calls) = run_with(&units, 0, vec![], &mut frame, &s);
            assert_eq!(lasti, ret);
            assert!(calls.is_empty());
            assert_eq!(frame.stack_top, 0);
            let body_ran = frame.data[1] == (s.five.ptr() | BORROW_TAG) as usize;
            assert_eq!(body_ran, expect_body);
        }
        assert_eq!(s.t.rc(), LEAKED);
        assert_eq!(s.f.rc(), LEAKED);
    }

    #[test]
    fn pop_jump_if_true_on_bool_singletons_needs_no_helper() {
        let s = singletons();
        let (units, ret) = branchy(POP_JUMP_IF_TRUE);
        for (value, expect_body) in [(s.f.ptr() | BORROW_TAG, true), (s.t.ptr(), false)] {
            let mut frame = Frame::new(1, 4);
            frame.push(value);
            let (lasti, calls) = run_with(&units, 0, vec![], &mut frame, &s);
            assert_eq!(lasti, ret);
            assert!(calls.is_empty());
            let body_ran = frame.data[1] == (s.five.ptr() | BORROW_TAG) as usize;
            assert_eq!(body_ran, expect_body);
        }
    }

    #[test]
    fn pop_jump_if_false_on_other_objects_asks_helper() {
        let s = singletons();
        let other = Obj::new(1);
        let (units, ret) = branchy(POP_JUMP_IF_FALSE);
        let mut frame = Frame::new(1, 4);
        frame.push(other.ptr());
        let (lasti, calls) = run_with(&units, 0, vec![1], &mut frame, &s);
        assert_eq!(lasti, ret);
        assert_eq!(ids(&calls), [HelperId::PopIsTrue]);
    }

    #[test]
    fn pop_jump_if_none_compares_pointers_inline() {
        let s = singletons();
        let other = Obj::new(3);
        for (op, value, expect_body) in [
            (POP_JUMP_IF_NONE, s.n.ptr() | BORROW_TAG, false),
            (POP_JUMP_IF_NONE, other.ptr(), true),
            (POP_JUMP_IF_NOT_NONE, s.n.ptr(), true),
            (POP_JUMP_IF_NOT_NONE, other.ptr(), false),
        ] {
            let (units, ret) = branchy(op);
            let mut frame = Frame::new(1, 4);
            frame.push(value);
            let (lasti, calls) = run_with(&units, 0, vec![], &mut frame, &s);
            assert_eq!(lasti, ret);
            assert!(calls.is_empty());
            assert_eq!(frame.stack_top, 0);
            let body_ran = frame.data[1] == (s.five.ptr() | BORROW_TAG) as usize;
            assert_eq!(body_ran, expect_body);
        }
        assert_eq!(other.rc(), 1);
    }

    #[test]
    fn helper_calls_see_and_update_the_shared_stack_top() {
        let a = Obj::new(1);
        let b = Obj::new(1);
        let mut frame = Frame::new(2, 4);
        frame.data[0] = a.ptr() as usize;
        frame.data[1] = b.ptr() as usize;
        let s = singletons();
        let mut units = vec![u(LOAD_FAST, 0), u(LOAD_FAST, 1)];
        units.extend(with_caches(BINARY_OP, BinaryOperator::Add as u8));
        units.push(u(STORE_FAST, 0));
        units.push(u(RETURN, 0));
        let (_, calls) = run_with(&units, 0, vec![], &mut frame, &s);
        assert_eq!(ids(&calls), [HelperId::BinaryOp]);
        assert_eq!(frame.stack_top, 0);
        assert_eq!(frame.data[0], a.ptr() as usize);
        assert_eq!(a.rc(), 1);
        assert_eq!(b.rc(), 2);
    }

    #[test]
    fn helper_error_exits_just_after_faulting_instruction() {
        let mut units = vec![u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 5)];
        units.extend(with_caches(BINARY_OP, BinaryOperator::Add as u8));
        units.push(u(RETURN, 0));
        let (lasti, calls) = run(&units, 0, vec![-1]);
        assert_eq!(lasti, 3);
        assert_eq!(calls.len(), 1);
    }

    fn while_loop() -> (Vec<CodeUnit>, u32, usize) {
        let mut v = vec![u(LOAD_FAST, 0)];
        let pj = v.len();
        v.extend(with_caches(POP_JUMP_IF_FALSE, 0));
        let body = v.len();
        v.push(u(LOAD_SMALL_INT, 5));
        v.push(u(Instruction::PopTop, 0));
        let jb = v.len();
        let jb_after = jb + 1 + caches(JUMP_BACKWARD) as usize;
        v.extend(with_caches(JUMP_BACKWARD, jb_after as u8));
        let ret = v.len();
        v.push(u(RETURN, 0));
        let after = pj + 1 + caches(POP_JUMP_IF_FALSE) as usize;
        v[pj].arg = OpArgByte::new((ret - after) as u8);
        (v, ret as u32, body)
    }

    #[test]
    fn loop_runs_until_condition_false_and_checks_eval_breaker() {
        let (units, ret, _) = while_loop();
        let other = Obj::new(1);
        let mut frame = Frame::new(1, 4);
        frame.data[0] = other.ptr() as usize;
        let s = singletons();
        let (lasti, calls) = run_with(&units, 0, vec![1, 0, 1, 0], &mut frame, &s);
        assert_eq!(lasti, ret);
        use HelperId::*;
        assert_eq!(ids(&calls), [PopIsTrue, EvalBreaker, PopIsTrue, PopIsTrue]);
        assert_eq!(other.rc(), 4);
    }

    #[test]
    fn eval_breaker_is_consulted_on_the_first_back_edge_then_every_interval() {
        let (units, _, _) = while_loop();
        let other = Obj::new(1);
        let mut frame = Frame::new(1, 4);
        frame.data[0] = other.ptr() as usize;
        let s = singletons();
        let iterations = BREAKER_INTERVAL as usize + 2;
        let mut script = vec![];
        for _ in 0..iterations {
            script.push(1);
        }
        script.push(0);
        let (_, calls) = run_with(&units, 0, script, &mut frame, &s);
        let breaker_calls = calls
            .iter()
            .filter(|c| c.0 == HelperId::EvalBreaker)
            .count();
        assert_eq!(breaker_calls, 2);
        let positions: Vec<usize> = calls
            .iter()
            .enumerate()
            .filter(|(_, c)| c.0 == HelperId::EvalBreaker)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions[0], 1);
        assert_eq!(positions[1], 2 + BREAKER_INTERVAL as usize);
    }

    #[test]
    fn eval_breaker_error_exits_after_jump_instruction() {
        let (units, _, _) = while_loop();
        let other = Obj::new(1);
        let mut frame = Frame::new(1, 4);
        frame.data[0] = other.ptr() as usize;
        let s = singletons();
        let (lasti, calls) = run_with(&units, 0, vec![1, -1], &mut frame, &s);
        let jb = units
            .iter()
            .position(|c| matches!(c.op, Instruction::JumpBackward { .. }))
            .unwrap();
        assert_eq!(lasti, jb as u32 + 1);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn can_enter_in_the_middle_of_the_code() {
        let (units, ret, body) = while_loop();
        let s = singletons();
        let mut frame = Frame::new(1, 4);
        frame.data[0] = s.f.ptr() as usize;
        let (lasti, calls) = run_with(&units, body, vec![], &mut frame, &s);
        assert_eq!(lasti, ret);
        assert_eq!(ids(&calls), [HelperId::EvalBreaker]);
    }

    #[test]
    fn extended_arg_widens_the_operand() {
        let mut v = vec![u(Instruction::ExtendedArg, 1)];
        v.extend(with_caches(
            Instruction::LoadGlobal {
                namei: Arg::marker(),
            },
            4,
        ));
        v.push(u(RETURN, 0));
        let (lasti, calls) = run(&v, 0, vec![]);
        assert_eq!(lasti, v.len() as u32 - 1);
        assert_eq!(calls, [(HelperId::LoadGlobal, 0x104)]);
    }

    #[test]
    fn unsupported_instruction_exits_before_it_runs() {
        let mut v = vec![u(LOAD_SMALL_INT, 5)];
        v.extend(with_caches(CALL, 0));
        v.push(u(RETURN, 0));
        let mut frame = Frame::new(1, 4);
        let s = singletons();
        let (lasti, calls) = run_with(&v, 0, vec![], &mut frame, &s);
        assert_eq!(lasti, 1);
        assert!(calls.is_empty());
        assert_eq!(frame.stack_top, 1);
    }

    #[test]
    fn specialized_ops_are_compiled_as_their_base_op() {
        let mut v = vec![u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 5)];
        v.extend(with_caches(
            Instruction::BinaryOpSubscrListInt,
            BinaryOperator::Subscr as u8,
        ));
        v.extend(with_caches(Instruction::ToBoolInt, 0));
        v.push(u(RETURN, 0));
        let (_, calls) = run(&v, 0, vec![]);
        assert_eq!(
            calls[0],
            (HelperId::BinaryOp, BinaryOperator::Subscr as u64)
        );
        assert_eq!(calls[1], (HelperId::ToBool, 0));
    }

    #[test]
    fn for_iter_exhaustion_skips_end_for() {
        let mut v = vec![];
        let fi = v.len();
        v.extend(with_caches(FOR_ITER, 0));
        v.push(u(STORE_FAST, 0));
        let jb = v.len();
        v.extend(with_caches(
            JUMP_BACKWARD,
            (jb + 1 + caches(JUMP_BACKWARD) as usize) as u8,
        ));
        let end_for = v.len();
        v.push(u(Instruction::EndFor, 0));
        v.push(u(Instruction::PopIter, 0));
        v.push(u(RETURN, 0));
        let after = fi + 1 + caches(FOR_ITER) as usize;
        v[fi].arg = OpArgByte::new((end_for - after) as u8);
        let mut frame = Frame::new(1, 4);
        frame.push(0);
        frame.push(0);
        let s = singletons();
        let (lasti, calls) = run_with(&v, 0, vec![0, 0, 1], &mut frame, &s);
        assert_eq!(lasti, v.len() as u32 - 1);
        use HelperId::*;
        assert_eq!(ids(&calls), [ForIter, EvalBreaker, ForIter]);
        assert_eq!(calls[0].1, end_for as u64);
    }

    #[test]
    fn compare_fast_hit_pops_operands_and_pushes_bool_singleton() {
        let a = Obj::new(1);
        let b = Obj::new(1);
        let s = singletons();
        for (result, expect) in [(1i64, s.t.ptr()), (0, s.f.ptr())] {
            let mut frame = Frame::new(1, 4);
            frame.push(a.ptr());
            frame.push(b.ptr() | BORROW_TAG);
            a.rc.store(2, Relaxed);
            let mut units = vec![];
            units.extend(with_caches(
                Instruction::CompareOp {
                    opname: Arg::marker(),
                },
                ComparisonOperator::Less as u8,
            ));
            units.push(u(RETURN, 0));
            let (lasti, calls) = run_with(&units, 0, vec![result], &mut frame, &s);
            assert_eq!(lasti, units.len() as u32 - 1);
            assert_eq!(
                calls,
                [(HelperId::CompareFast, ComparisonOperator::Less as u64)]
            );
            assert_eq!(frame.stack(), [(expect | BORROW_TAG) as usize]);
            assert_eq!(a.rc(), 1);
            assert_eq!(b.rc(), 1);
        }
    }

    #[test]
    fn compare_fast_miss_falls_back_to_generic_helper() {
        let mut units = vec![u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 5)];
        units.extend(with_caches(
            Instruction::CompareOpInt,
            ComparisonOperator::Equal as u8,
        ));
        units.push(u(RETURN, 0));
        let (_, calls) = run(&units, 0, vec![FAST_PATH_MISS]);
        assert_eq!(ids(&calls), [HelperId::CompareFast, HelperId::CompareOp]);
    }

    fn compare_then_jump(jump_op: Instruction) -> (Vec<CodeUnit>, u32) {
        let mut v = vec![u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 5)];
        v.extend(with_caches(
            Instruction::CompareOp {
                opname: Arg::marker(),
            },
            ComparisonOperator::Less as u8,
        ));
        let pj = v.len();
        v.extend(with_caches(jump_op, 0));
        v.push(u(LOAD_SMALL_INT, 5));
        v.push(u(STORE_FAST, 0));
        let ret = v.len();
        v.push(u(RETURN, 0));
        let after = pj + 1 + caches(jump_op) as usize;
        v[pj].arg = OpArgByte::new((ret - after) as u8);
        (v, ret as u32)
    }

    #[test]
    fn compare_fused_with_pop_jump_branches_without_touching_the_stack() {
        let s = singletons();
        for (op, result, expect_body) in [
            (POP_JUMP_IF_FALSE, 0i64, false),
            (POP_JUMP_IF_FALSE, 1, true),
            (POP_JUMP_IF_TRUE, 1, false),
            (POP_JUMP_IF_TRUE, 0, true),
        ] {
            let (units, ret) = compare_then_jump(op);
            let mut frame = Frame::new(1, 4);
            let (lasti, calls) = run_with(&units, 0, vec![result], &mut frame, &s);
            assert_eq!(lasti, ret);
            assert_eq!(ids(&calls), [HelperId::CompareFast]);
            assert_eq!(frame.stack_top, 0);
            let body_ran = frame.data[0] == s.five.ptr() as usize;
            assert_eq!(body_ran, expect_body);
        }
    }

    #[test]
    fn compare_fused_miss_uses_generic_compare_and_pop_helper() {
        let (units, ret) = compare_then_jump(POP_JUMP_IF_FALSE);
        let (lasti, calls) = run(&units, 0, vec![FAST_PATH_MISS, 0, 1]);
        assert_eq!(lasti, ret);
        assert_eq!(
            ids(&calls),
            [
                HelperId::CompareFast,
                HelperId::CompareOp,
                HelperId::PopIsTrue
            ]
        );
    }

    #[test]
    fn entering_at_a_fused_jump_exits_to_the_interpreter() {
        let (units, _) = compare_then_jump(POP_JUMP_IF_FALSE);
        let pj = units
            .iter()
            .position(|c| matches!(c.op, Instruction::PopJumpIfFalse { .. }))
            .unwrap();
        let (lasti, calls) = run(&units, pj, vec![]);
        assert_eq!(lasti, pj as u32);
        assert!(calls.is_empty());
    }

    #[test]
    fn specialized_int_binary_op_uses_fast_helper_then_generic_on_miss() {
        let mut units = vec![u(LOAD_SMALL_INT, 5), u(LOAD_SMALL_INT, 5)];
        units.extend(with_caches(
            Instruction::BinaryOpAddInt,
            BinaryOperator::Add as u8,
        ));
        units.push(u(LOAD_SMALL_INT, 5));
        units.extend(with_caches(
            Instruction::BinaryOpSubtractInt,
            BinaryOperator::Subtract as u8,
        ));
        units.push(u(RETURN, 0));
        let mut frame = Frame::new(1, 4);
        let s = singletons();
        let (_, calls) = run_with(&units, 0, vec![0, FAST_PATH_MISS], &mut frame, &s);
        assert_eq!(
            ids(&calls),
            [
                HelperId::BinaryOpIntFast,
                HelperId::BinaryOpIntFast,
                HelperId::BinaryOp
            ]
        );
        assert_eq!(frame.stack_top, 1);
    }

    #[test]
    #[should_panic]
    fn entry_out_of_range_panics() {
        let units = [u(RETURN, 0)];
        run(&units, 1, vec![]);
    }
}
