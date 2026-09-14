use crate::asm::{Assembler, Cond, Label, Reg};
use crate::exec::{ARG0, ARG1, ExecError, ExecMemory, SHADOW_SPACE};
use core::ffi::c_void;
use rustpython_compiler_core::bytecode::{CodeUnit, Instruction, OpArgState};

#[repr(C)]
pub struct JitContext {
    pub user: *mut c_void,
}

impl JitContext {
    pub fn new(user: *mut c_void) -> Self {
        Self { user }
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
}

impl HelperId {
    pub const COUNT: usize = HelperId::EvalBreaker as usize + 1;
}

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
    /// `ctx` must stay valid for the whole run and every helper in the table
    /// used at compile time must accept it.
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

    pub fn code_size(&self) -> usize {
        self.mem.size()
    }
}

const CTX: Reg = Reg::Rbx;

struct Emitter<'a> {
    a: Assembler,
    helpers: &'a HelperTable,
    labels: Vec<Label>,
    epilogue: Label,
    len: usize,
}

impl Emitter<'_> {
    fn exit(&mut self, lasti: usize) {
        self.a.mov_ri(Reg::Rax, lasti as u64);
        self.a.jmp(self.epilogue);
    }

    fn call(&mut self, id: HelperId, arg: u64, idx: usize) {
        self.a.mov_rr(ARG0, CTX);
        self.a.mov_ri(ARG1, arg);
        self.a.mov_ri(Reg::Rax, self.helpers.address(id));
        self.a.call_r(Reg::Rax);
        self.a.test_rr(Reg::Rax, Reg::Rax);
        let ok = self.a.new_label();
        self.a.jcc(Cond::NS, ok);
        self.exit(idx + 1);
        self.a.bind(ok);
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
}

pub fn compile(units: &[CodeUnit], helpers: &HelperTable) -> Result<Compiled, CompileError> {
    let mut a = Assembler::new();
    let labels: Vec<Label> = (0..units.len()).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();

    a.push(CTX);
    if SHADOW_SPACE > 0 {
        a.sub_ri(Reg::Rsp, SHADOW_SPACE);
    }
    a.mov_rr(CTX, ARG0);
    a.jmp_r(ARG1);

    let mut e = Emitter {
        a,
        helpers,
        labels,
        epilogue,
        len: units.len(),
    };

    let mut arg_state = OpArgState::default();
    let mut idx = 0;
    while idx < units.len() {
        e.a.bind(e.labels[idx]);
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
            | Instruction::LoadFastCheck { .. } => e.call(HelperId::LoadFast, argv.into(), idx),
            Instruction::LoadFastLoadFast { .. }
            | Instruction::LoadFastBorrowLoadFastBorrow { .. } => {
                e.call(HelperId::LoadFastLoadFast, argv.into(), idx)
            }
            Instruction::StoreFast { .. } => e.call(HelperId::StoreFast, argv.into(), idx),
            Instruction::StoreFastLoadFast { .. } => {
                e.call(HelperId::StoreFastLoadFast, argv.into(), idx)
            }
            Instruction::StoreFastStoreFast { .. } => {
                e.call(HelperId::StoreFastStoreFast, argv.into(), idx)
            }
            Instruction::LoadConst { .. } => e.call(HelperId::LoadConst, argv.into(), idx),
            Instruction::LoadSmallInt { .. } => e.call(HelperId::LoadSmallInt, argv.into(), idx),
            Instruction::LoadGlobal { .. } => e.call(HelperId::LoadGlobal, argv.into(), idx),
            Instruction::BinaryOp { .. } => e.call(HelperId::BinaryOp, argv.into(), idx),
            Instruction::CompareOp { .. } => e.call(HelperId::CompareOp, argv.into(), idx),
            Instruction::ToBool => e.call(HelperId::ToBool, 0, idx),
            Instruction::UnaryNot => e.call(HelperId::UnaryNot, 0, idx),
            Instruction::UnaryNegative => e.call(HelperId::UnaryNegative, 0, idx),
            Instruction::UnaryInvert => e.call(HelperId::UnaryInvert, 0, idx),
            Instruction::PopTop | Instruction::EndFor | Instruction::PopIter => {
                e.call(HelperId::PopTop, 0, idx)
            }
            Instruction::Copy { .. } => e.call(HelperId::Copy, argv.into(), idx),
            Instruction::Swap { .. } => e.call(HelperId::Swap, argv.into(), idx),
            Instruction::PopJumpIfFalse { .. } => {
                e.call(HelperId::PopIsTrue, 0, idx);
                e.branch_if(Cond::NE, forward(argv));
            }
            Instruction::PopJumpIfTrue { .. } => {
                e.call(HelperId::PopIsTrue, 0, idx);
                e.branch_if(Cond::E, forward(argv));
            }
            Instruction::PopJumpIfNone { .. } => {
                e.call(HelperId::PopIsNone, 0, idx);
                e.branch_if(Cond::E, forward(argv));
            }
            Instruction::PopJumpIfNotNone { .. } => {
                e.call(HelperId::PopIsNone, 0, idx);
                e.branch_if(Cond::NE, forward(argv));
            }
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
                e.call(HelperId::EvalBreaker, 0, idx);
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

    e.a.bind(e.epilogue);
    if SHADOW_SPACE > 0 {
        e.a.add_ri(Reg::Rsp, SHADOW_SPACE);
    }
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
    use rustpython_compiler_core::bytecode::{
        Arg, BinaryOperator, CodeUnit, ComparisonOperator, Instruction, OpArgByte,
    };

    struct Recorder {
        calls: Vec<(HelperId, u64)>,
        script: Vec<i64>,
    }

    thread_local! {
        static REC: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    }

    macro_rules! fake {
        ($name:ident, $id:expr) => {
            extern "C" fn $name(_ctx: *mut JitContext, arg: u64) -> i64 {
                REC.with(|r| {
                    let mut r = r.borrow_mut();
                    let rec = r.as_mut().unwrap();
                    rec.calls.push(($id, arg));
                    if rec.script.is_empty() {
                        0
                    } else {
                        rec.script.remove(0)
                    }
                })
            }
        };
    }

    fake!(f_load_fast, HelperId::LoadFast);
    fake!(f_store_fast, HelperId::StoreFast);
    fake!(f_load_small_int, HelperId::LoadSmallInt);
    fake!(f_load_const, HelperId::LoadConst);
    fake!(f_binary_op, HelperId::BinaryOp);
    fake!(f_compare_op, HelperId::CompareOp);
    fake!(f_pop_is_true, HelperId::PopIsTrue);
    fake!(f_pop_is_none, HelperId::PopIsNone);
    fake!(f_pop_top, HelperId::PopTop);
    fake!(f_for_iter, HelperId::ForIter);
    fake!(f_eval_breaker, HelperId::EvalBreaker);
    fake!(f_generic, HelperId::LoadGlobal);

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
        t
    }

    fn run(units: &[CodeUnit], entry: usize, script: Vec<i64>) -> (u32, Vec<(HelperId, u64)>) {
        REC.with(|r| {
            *r.borrow_mut() = Some(Recorder {
                calls: vec![],
                script,
            })
        });
        let compiled = compile(units, &table()).expect("compile");
        let mut ctx = JitContext::new(core::ptr::null_mut());
        let lasti = unsafe { compiled.run(&mut ctx, entry) };
        let calls = REC.with(|r| r.borrow_mut().take().unwrap().calls);
        (lasti, calls)
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
    const POP_JUMP_IF_NONE: Instruction = Instruction::PopJumpIfNone {
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

    fn straight_line() -> Vec<CodeUnit> {
        let mut v = vec![u(LOAD_FAST, 0), u(LOAD_SMALL_INT, 1)];
        v.extend(with_caches(BINARY_OP, BinaryOperator::Add as u8));
        v.push(u(STORE_FAST, 0));
        v.push(u(Instruction::ReturnValue, 0));
        v
    }

    #[test]
    fn straight_line_calls_helpers_and_exits_at_return() {
        let units = straight_line();
        let ret_idx = units.len() as u32 - 1;
        let (lasti, calls) = run(&units, 0, vec![]);
        assert_eq!(lasti, ret_idx);
        assert_eq!(
            calls,
            [
                (HelperId::LoadFast, 0),
                (HelperId::LoadSmallInt, 1),
                (HelperId::BinaryOp, BinaryOperator::Add as u64),
                (HelperId::StoreFast, 0),
            ]
        );
    }

    #[test]
    fn helper_error_exits_just_after_faulting_instruction() {
        let units = straight_line();
        let (lasti, calls) = run(&units, 0, vec![0, 0, -1]);
        assert_eq!(lasti, 3);
        assert_eq!(calls.len(), 3);
    }

    fn branchy() -> (Vec<CodeUnit>, u32) {
        let mut v = vec![u(LOAD_FAST, 0)];
        let pj = v.len();
        v.extend(with_caches(POP_JUMP_IF_FALSE, 0));
        v.push(u(LOAD_SMALL_INT, 1));
        v.push(u(Instruction::PopTop, 0));
        let ret = v.len();
        v.push(u(Instruction::ReturnValue, 0));
        let after = pj + 1 + caches(POP_JUMP_IF_FALSE) as usize;
        v[pj].arg = OpArgByte::new((ret - after) as u8);
        (v, ret as u32)
    }

    #[test]
    fn pop_jump_if_false_taken_when_helper_reports_false() {
        let (units, ret) = branchy();
        let (lasti, calls) = run(&units, 0, vec![0, 0]);
        assert_eq!(lasti, ret);
        assert_eq!(calls, [(HelperId::LoadFast, 0), (HelperId::PopIsTrue, 0)]);
    }

    #[test]
    fn pop_jump_if_false_falls_through_when_helper_reports_true() {
        let (units, ret) = branchy();
        let (lasti, calls) = run(&units, 0, vec![0, 1]);
        assert_eq!(lasti, ret);
        assert_eq!(
            calls,
            [
                (HelperId::LoadFast, 0),
                (HelperId::PopIsTrue, 0),
                (HelperId::LoadSmallInt, 1),
                (HelperId::PopTop, 0),
            ]
        );
    }

    #[test]
    fn pop_jump_if_none_branches_on_helper_result() {
        let mut v = vec![u(LOAD_FAST, 0)];
        v.extend(with_caches(POP_JUMP_IF_NONE, 1));
        v.push(u(Instruction::PopTop, 0));
        v.push(u(Instruction::ReturnValue, 0));
        let (lasti, calls) = run(&v, 0, vec![0, 1]);
        assert_eq!(lasti, v.len() as u32 - 1);
        assert_eq!(calls, [(HelperId::LoadFast, 0), (HelperId::PopIsNone, 0)]);
        let (_, calls) = run(&v, 0, vec![0, 0]);
        assert_eq!(calls.len(), 3);
    }

    fn while_loop() -> (Vec<CodeUnit>, u32, usize) {
        let mut v = vec![u(LOAD_FAST, 0)];
        let pj = v.len();
        v.extend(with_caches(POP_JUMP_IF_FALSE, 0));
        let body = v.len();
        v.push(u(LOAD_SMALL_INT, 1));
        v.push(u(Instruction::PopTop, 0));
        let jb = v.len();
        let jb_after = jb + 1 + caches(JUMP_BACKWARD) as usize;
        v.extend(with_caches(JUMP_BACKWARD, jb_after as u8));
        let ret = v.len();
        v.push(u(Instruction::ReturnValue, 0));
        let after = pj + 1 + caches(POP_JUMP_IF_FALSE) as usize;
        v[pj].arg = OpArgByte::new((ret - after) as u8);
        (v, ret as u32, body)
    }

    #[test]
    fn loop_runs_until_condition_false_and_checks_eval_breaker() {
        let (units, ret, _) = while_loop();
        let (lasti, calls) = run(&units, 0, vec![0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(lasti, ret);
        let ids: Vec<HelperId> = calls.iter().map(|c| c.0).collect();
        use HelperId::*;
        assert_eq!(
            ids,
            [
                LoadFast,
                PopIsTrue,
                LoadSmallInt,
                PopTop,
                EvalBreaker,
                LoadFast,
                PopIsTrue,
                LoadSmallInt,
                PopTop,
                EvalBreaker,
                LoadFast,
                PopIsTrue
            ]
        );
    }

    #[test]
    fn eval_breaker_error_exits_after_jump_instruction() {
        let (units, _, _) = while_loop();
        let (lasti, calls) = run(&units, 0, vec![0, 1, 0, 0, -1]);
        let jb = units
            .iter()
            .position(|c| matches!(c.op, Instruction::JumpBackward { .. }))
            .unwrap();
        assert_eq!(lasti, jb as u32 + 1);
        assert_eq!(calls.len(), 5);
    }

    #[test]
    fn can_enter_in_the_middle_of_the_code() {
        let (units, ret, body) = while_loop();
        let (lasti, calls) = run(&units, body, vec![0, 0, 0, 0, 0]);
        assert_eq!(lasti, ret);
        use HelperId::*;
        let ids: Vec<HelperId> = calls.iter().map(|c| c.0).collect();
        assert_eq!(
            ids,
            [LoadSmallInt, PopTop, EvalBreaker, LoadFast, PopIsTrue]
        );
    }

    #[test]
    fn extended_arg_widens_the_operand() {
        let v = vec![
            u(Instruction::ExtendedArg, 1),
            u(LOAD_FAST, 4),
            u(Instruction::ReturnValue, 0),
        ];
        let (lasti, calls) = run(&v, 0, vec![]);
        assert_eq!(lasti, 2);
        assert_eq!(calls, [(HelperId::LoadFast, 0x104)]);
    }

    #[test]
    fn unsupported_instruction_exits_before_it_runs() {
        let mut v = vec![u(LOAD_FAST, 0)];
        v.extend(with_caches(CALL, 0));
        v.push(u(Instruction::ReturnValue, 0));
        let (lasti, calls) = run(&v, 0, vec![]);
        assert_eq!(lasti, 1);
        assert_eq!(calls, [(HelperId::LoadFast, 0)]);
    }

    #[test]
    fn specialized_ops_are_compiled_as_their_base_op() {
        let mut v = vec![u(LOAD_FAST, 0), u(LOAD_FAST, 1)];
        v.extend(with_caches(
            Instruction::BinaryOpAddInt,
            BinaryOperator::Add as u8,
        ));
        v.extend(with_caches(
            Instruction::CompareOpInt,
            ComparisonOperator::Less as u8,
        ));
        v.push(u(Instruction::ReturnValue, 0));
        let (_, calls) = run(&v, 0, vec![]);
        assert_eq!(calls[2], (HelperId::BinaryOp, BinaryOperator::Add as u64));
        assert_eq!(
            calls[3],
            (HelperId::CompareOp, ComparisonOperator::Less as u64)
        );
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
        v.push(u(Instruction::ReturnValue, 0));
        let after = fi + 1 + caches(FOR_ITER) as usize;
        v[fi].arg = OpArgByte::new((end_for - after) as u8);
        let (lasti, calls) = run(&v, 0, vec![0, 0, 0, 1]);
        assert_eq!(lasti, v.len() as u32 - 1);
        use HelperId::*;
        let ids: Vec<HelperId> = calls.iter().map(|c| c.0).collect();
        assert_eq!(ids, [ForIter, StoreFast, EvalBreaker, ForIter, PopTop]);
        assert_eq!(calls[0].1, end_for as u64);
    }

    #[test]
    #[should_panic]
    fn entry_out_of_range_panics() {
        let units = straight_line();
        run(&units, units.len(), vec![]);
    }

    #[test]
    fn same_compiled_code_can_be_entered_at_several_points() {
        let (units, ret, body) = while_loop();
        let compiled = compile(&units, &table()).expect("compile");
        for entry in [0, body] {
            REC.with(|r| {
                *r.borrow_mut() = Some(Recorder {
                    calls: vec![],
                    script: vec![0, 0],
                })
            });
            let mut ctx = JitContext::new(core::ptr::null_mut());
            assert_eq!(unsafe { compiled.run(&mut ctx, entry) }, ret);
        }
    }
}
