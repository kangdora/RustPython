#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Reg {
    const fn low(self) -> u8 {
        (self as u8) & 7
    }

    const fn ext(self) -> bool {
        (self as u8) >= 8
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    O,
    NO,
    B,
    AE,
    E,
    NE,
    BE,
    A,
    S,
    NS,
    L,
    GE,
    LE,
    G,
}

impl Cond {
    const fn code(self) -> u8 {
        match self {
            Self::O => 0x0,
            Self::NO => 0x1,
            Self::B => 0x2,
            Self::AE => 0x3,
            Self::E => 0x4,
            Self::NE => 0x5,
            Self::BE => 0x6,
            Self::A => 0x7,
            Self::S => 0x8,
            Self::NS => 0x9,
            Self::L => 0xC,
            Self::GE => 0xD,
            Self::LE => 0xE,
            Self::G => 0xF,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Default)]
pub struct Assembler {
    buf: Vec<u8>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label)>,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn finish(mut self) -> Vec<u8> {
        for (at, label) in core::mem::take(&mut self.fixups) {
            let target = self.labels[label.0].expect("jump to unbound label");
            let rel = target as i64 - (at as i64 + 4);
            let rel = i32::try_from(rel).expect("jump displacement out of range");
            self.buf[at..at + 4].copy_from_slice(&rel.to_le_bytes());
        }
        self.buf
    }

    pub fn new_label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        assert!(self.labels[label.0].is_none(), "label bound twice");
        self.labels[label.0] = Some(self.buf.len());
    }

    pub fn bind_new(&mut self) -> Label {
        let l = self.new_label();
        self.bind(l);
        l
    }

    fn byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    fn rex(&mut self, w: bool, reg: Reg, rm: Reg) {
        let mut rex = 0x40;
        if w {
            rex |= 0x08;
        }
        if reg.ext() {
            rex |= 0x04;
        }
        if rm.ext() {
            rex |= 0x01;
        }
        if rex != 0x40 {
            self.byte(rex);
        }
    }

    fn rex_w(&mut self, reg: Reg, rm: Reg) {
        self.rex(true, reg, rm);
    }

    fn modrm_rr(&mut self, reg: u8, rm: Reg) {
        self.byte(0xC0 | (reg << 3) | rm.low());
    }

    fn modrm_mem(&mut self, reg: u8, base: Reg, disp: i32) {
        self.byte(0x80 | (reg << 3) | base.low());
        if base.low() == 4 {
            self.byte(0x24);
        }
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    pub fn mov_rr(&mut self, dst: Reg, src: Reg) {
        self.rex_w(src, dst);
        self.byte(0x89);
        self.modrm_rr(src.low(), dst);
    }

    pub fn mov_rm(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.rex_w(dst, base);
        self.byte(0x8B);
        self.modrm_mem(dst.low(), base, disp);
    }

    pub fn mov_mr(&mut self, base: Reg, disp: i32, src: Reg) {
        self.rex_w(src, base);
        self.byte(0x89);
        self.modrm_mem(src.low(), base, disp);
    }

    pub fn mov_ri(&mut self, dst: Reg, imm: u64) {
        self.rex_w(Reg::Rax, dst);
        self.byte(0xB8 | dst.low());
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn mov_rm32(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.rex(false, dst, base);
        self.byte(0x8B);
        self.modrm_mem(dst.low(), base, disp);
    }

    pub fn mov_mr32(&mut self, base: Reg, disp: i32, src: Reg) {
        self.rex(false, src, base);
        self.byte(0x89);
        self.modrm_mem(src.low(), base, disp);
    }

    fn rex_idx(&mut self, reg: Reg, index: Reg, base: Reg) {
        let mut rex = 0x48;
        if reg.ext() {
            rex |= 0x04;
        }
        if index.ext() {
            rex |= 0x02;
        }
        if base.ext() {
            rex |= 0x01;
        }
        self.byte(rex);
    }

    fn modrm_idx8(&mut self, reg: u8, base: Reg, index: Reg, disp: i8) {
        self.byte(0x44 | (reg << 3));
        self.byte(0xC0 | (index.low() << 3) | base.low());
        self.byte(disp as u8);
    }

    pub fn mov_rm_idx8(&mut self, dst: Reg, base: Reg, index: Reg, disp: i8) {
        self.rex_idx(dst, index, base);
        self.byte(0x8B);
        self.modrm_idx8(dst.low(), base, index, disp);
    }

    pub fn mov_mr_idx8(&mut self, base: Reg, index: Reg, disp: i8, src: Reg) {
        self.rex_idx(src, index, base);
        self.byte(0x89);
        self.modrm_idx8(src.low(), base, index, disp);
    }

    pub fn lock_xadd(&mut self, base: Reg, disp: i32, src: Reg) {
        self.byte(0xF0);
        self.rex_w(src, base);
        self.byte(0x0F);
        self.byte(0xC1);
        self.modrm_mem(src.low(), base, disp);
    }

    pub fn lock_add_mi(&mut self, base: Reg, disp: i32, imm: i32) {
        self.byte(0xF0);
        self.rex_w(Reg::Rax, base);
        self.byte(0x81);
        self.modrm_mem(0, base, disp);
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn mov_mi32(&mut self, base: Reg, disp: i32, imm: i32) {
        self.rex_w(Reg::Rax, base);
        self.byte(0xC7);
        self.modrm_mem(0, base, disp);
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    fn alu_rr(&mut self, opcode: u8, dst: Reg, src: Reg) {
        self.rex_w(src, dst);
        self.byte(opcode);
        self.modrm_rr(src.low(), dst);
    }

    fn alu_ri(&mut self, ext: u8, dst: Reg, imm: i32) {
        self.rex_w(Reg::Rax, dst);
        self.byte(0x81);
        self.modrm_rr(ext, dst);
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn add_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x01, dst, src);
    }

    pub fn sub_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x29, dst, src);
    }

    pub fn and_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x21, dst, src);
    }

    pub fn cmp_rr(&mut self, a: Reg, b: Reg) {
        self.alu_rr(0x39, a, b);
    }

    pub fn test_rr(&mut self, a: Reg, b: Reg) {
        self.alu_rr(0x85, a, b);
    }

    pub fn add_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(0, dst, imm);
    }

    pub fn or_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(1, dst, imm);
    }

    pub fn and_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(4, dst, imm);
    }

    pub fn sub_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(5, dst, imm);
    }

    pub fn cmp_ri(&mut self, a: Reg, imm: i32) {
        self.alu_ri(7, a, imm);
    }

    fn shift_ri(&mut self, ext: u8, dst: Reg, imm: u8) {
        self.rex_w(Reg::Rax, dst);
        self.byte(0xC1);
        self.modrm_rr(ext, dst);
        self.byte(imm);
    }

    pub fn shl_ri(&mut self, dst: Reg, imm: u8) {
        self.shift_ri(4, dst, imm);
    }

    pub fn shr_ri(&mut self, dst: Reg, imm: u8) {
        self.shift_ri(5, dst, imm);
    }

    pub fn sar_ri(&mut self, dst: Reg, imm: u8) {
        self.shift_ri(7, dst, imm);
    }

    pub fn imul_rr(&mut self, dst: Reg, src: Reg) {
        self.rex_w(dst, src);
        self.byte(0x0F);
        self.byte(0xAF);
        self.modrm_rr(dst.low(), src);
    }

    pub fn push(&mut self, r: Reg) {
        self.rex(false, Reg::Rax, r);
        self.byte(0x50 | r.low());
    }

    pub fn pop(&mut self, r: Reg) {
        self.rex(false, Reg::Rax, r);
        self.byte(0x58 | r.low());
    }

    pub fn call_r(&mut self, r: Reg) {
        self.rex(false, Reg::Rax, r);
        self.byte(0xFF);
        self.modrm_rr(2, r);
    }

    pub fn jmp_r(&mut self, r: Reg) {
        self.rex(false, Reg::Rax, r);
        self.byte(0xFF);
        self.modrm_rr(4, r);
    }

    pub fn ret(&mut self) {
        self.byte(0xC3);
    }

    pub fn label_offset(&self, label: Label) -> Option<usize> {
        self.labels[label.0]
    }

    pub fn jmp(&mut self, label: Label) {
        self.byte(0xE9);
        self.fixups.push((self.buf.len(), label));
        self.buf.extend_from_slice(&[0; 4]);
    }

    pub fn jcc(&mut self, cond: Cond, label: Label) {
        self.byte(0x0F);
        self.byte(0x80 | cond.code());
        self.fixups.push((self.buf.len(), label));
        self.buf.extend_from_slice(&[0; 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(f: impl FnOnce(&mut Assembler)) -> Vec<u8> {
        let mut a = Assembler::new();
        f(&mut a);
        a.finish()
    }

    #[test]
    fn mov_reg_reg() {
        assert_eq!(asm(|a| a.mov_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x89, 0xC8]);
        assert_eq!(asm(|a| a.mov_rr(Reg::R12, Reg::Rbx)), [0x49, 0x89, 0xDC]);
        assert_eq!(asm(|a| a.mov_rr(Reg::Rbx, Reg::R12)), [0x4C, 0x89, 0xE3]);
    }

    #[test]
    fn mov_load_store_disp32() {
        assert_eq!(
            asm(|a| a.mov_rm(Reg::Rax, Reg::Rcx, 8)),
            [0x48, 0x8B, 0x81, 8, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_mr(Reg::Rcx, 8, Reg::Rax)),
            [0x48, 0x89, 0x81, 8, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_rm(Reg::R9, Reg::R12, -16)),
            [0x4D, 0x8B, 0x8C, 0x24, 0xF0, 0xFF, 0xFF, 0xFF]
        );
        assert_eq!(
            asm(|a| a.mov_rm(Reg::Rax, Reg::Rbp, 0)),
            [0x48, 0x8B, 0x85, 0, 0, 0, 0]
        );
    }

    #[test]
    fn mov_imm() {
        assert_eq!(
            asm(|a| a.mov_ri(Reg::Rax, 0x1122334455667788)),
            [0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        assert_eq!(
            asm(|a| a.mov_ri(Reg::R10, 1)),
            [0x49, 0xBA, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_mi32(Reg::Rbx, 4, -1)),
            [0x48, 0xC7, 0x83, 4, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]
        );
    }

    #[test]
    fn arith() {
        assert_eq!(asm(|a| a.add_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x01, 0xC8]);
        assert_eq!(asm(|a| a.sub_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x29, 0xC8]);
        assert_eq!(
            asm(|a| a.add_ri(Reg::Rsp, 32)),
            [0x48, 0x81, 0xC4, 32, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.sub_ri(Reg::Rsp, 32)),
            [0x48, 0x81, 0xEC, 32, 0, 0, 0]
        );
        assert_eq!(asm(|a| a.cmp_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x39, 0xC8]);
        assert_eq!(
            asm(|a| a.cmp_ri(Reg::Rax, 7)),
            [0x48, 0x81, 0xF8, 7, 0, 0, 0]
        );
        assert_eq!(asm(|a| a.test_rr(Reg::Rax, Reg::Rax)), [0x48, 0x85, 0xC0]);
        assert_eq!(
            asm(|a| a.and_ri(Reg::Rax, 1)),
            [0x48, 0x81, 0xE0, 1, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.or_ri(Reg::Rax, 1)),
            [0x48, 0x81, 0xC8, 1, 0, 0, 0]
        );
        assert_eq!(asm(|a| a.sar_ri(Reg::Rax, 1)), [0x48, 0xC1, 0xF8, 1]);
        assert_eq!(asm(|a| a.shl_ri(Reg::Rax, 1)), [0x48, 0xC1, 0xE0, 1]);
        assert_eq!(
            asm(|a| a.imul_rr(Reg::Rax, Reg::Rcx)),
            [0x48, 0x0F, 0xAF, 0xC1]
        );
    }

    #[test]
    fn thirty_two_bit_moves_and_indexed_moves() {
        assert_eq!(
            asm(|a| a.mov_rm32(Reg::R12, Reg::Rax, 0)),
            [0x44, 0x8B, 0xA0, 0, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_mr32(Reg::Rax, 0, Reg::R12)),
            [0x44, 0x89, 0xA0, 0, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_rm32(Reg::Rax, Reg::Rcx, 4)),
            [0x8B, 0x81, 4, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.mov_rm_idx8(Reg::Rax, Reg::R14, Reg::R12, 0)),
            [0x4B, 0x8B, 0x44, 0xE6, 0x00]
        );
        assert_eq!(
            asm(|a| a.mov_rm_idx8(Reg::Rax, Reg::R14, Reg::R12, -8)),
            [0x4B, 0x8B, 0x44, 0xE6, 0xF8]
        );
        assert_eq!(
            asm(|a| a.mov_mr_idx8(Reg::R14, Reg::R12, 0, Reg::Rax)),
            [0x4B, 0x89, 0x44, 0xE6, 0x00]
        );
    }

    #[test]
    fn locked_refcount_ops_and_bit_ops() {
        assert_eq!(
            asm(|a| a.lock_xadd(Reg::Rax, 0, Reg::Rdx)),
            [0xF0, 0x48, 0x0F, 0xC1, 0x90, 0, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| a.lock_add_mi(Reg::Rax, 0, 1)),
            [0xF0, 0x48, 0x81, 0x80, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(asm(|a| a.and_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x21, 0xC8]);
        assert_eq!(asm(|a| a.shr_ri(Reg::Rcx, 61)), [0x48, 0xC1, 0xE9, 61]);
    }

    #[test]
    fn stack_and_calls() {
        assert_eq!(asm(|a| a.push(Reg::Rbx)), [0x53]);
        assert_eq!(asm(|a| a.push(Reg::R12)), [0x41, 0x54]);
        assert_eq!(asm(|a| a.pop(Reg::Rbx)), [0x5B]);
        assert_eq!(asm(|a| a.pop(Reg::R12)), [0x41, 0x5C]);
        assert_eq!(asm(|a| a.call_r(Reg::Rax)), [0xFF, 0xD0]);
        assert_eq!(asm(|a| a.call_r(Reg::R11)), [0x41, 0xFF, 0xD3]);
        assert_eq!(asm(|a| a.ret()), [0xC3]);
        assert_eq!(asm(|a| a.jmp_r(Reg::Rax)), [0xFF, 0xE0]);
        assert_eq!(asm(|a| a.jmp_r(Reg::R11)), [0x41, 0xFF, 0xE3]);
    }

    #[test]
    fn forward_and_backward_jumps() {
        let code = asm(|a| {
            let top = a.bind_new();
            let out = a.new_label();
            a.jcc(Cond::E, out);
            a.jmp(top);
            a.bind(out);
            a.ret();
        });
        assert_eq!(
            code,
            [0x0F, 0x84, 5, 0, 0, 0, 0xE9, 0xF5, 0xFF, 0xFF, 0xFF, 0xC3]
        );
    }

    #[test]
    fn conditions_encode_distinct_opcodes() {
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::NE, l);
                a.bind(l);
            })[1],
            0x85
        );
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::L, l);
                a.bind(l);
            })[1],
            0x8C
        );
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::GE, l);
                a.bind(l);
            })[1],
            0x8D
        );
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::LE, l);
                a.bind(l);
            })[1],
            0x8E
        );
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::G, l);
                a.bind(l);
            })[1],
            0x8F
        );
        assert_eq!(
            asm(|a| {
                let l = a.new_label();
                a.jcc(Cond::O, l);
                a.bind(l);
            })[1],
            0x80
        );
    }

    #[test]
    #[should_panic]
    fn unbound_label_panics() {
        asm(|a| {
            let l = a.new_label();
            a.jmp(l);
        });
    }
}
