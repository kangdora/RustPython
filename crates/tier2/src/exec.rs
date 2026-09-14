use crate::asm::Reg;

#[cfg(windows)]
pub const ARG0: Reg = Reg::Rcx;
#[cfg(windows)]
pub const ARG1: Reg = Reg::Rdx;
#[cfg(windows)]
pub const ARG2: Reg = Reg::R8;
#[cfg(windows)]
pub const SHADOW_SPACE: i32 = 32;

#[cfg(not(windows))]
pub const ARG0: Reg = Reg::Rdi;
#[cfg(not(windows))]
pub const ARG1: Reg = Reg::Rsi;
#[cfg(not(windows))]
pub const ARG2: Reg = Reg::Rdx;
#[cfg(not(windows))]
pub const SHADOW_SPACE: i32 = 0;

#[derive(Debug)]
pub enum ExecError {
    Empty,
    Os(i32),
}

impl core::fmt::Display for ExecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "no code to map"),
            Self::Os(code) => write!(f, "os error {code}"),
        }
    }
}

impl std::error::Error for ExecError {}

pub struct ExecMemory {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for ExecMemory {}
unsafe impl Sync for ExecMemory {}

impl ExecMemory {
    pub fn new(code: &[u8]) -> Result<Self, ExecError> {
        if code.is_empty() {
            return Err(ExecError::Empty);
        }
        let size = code.len().div_ceil(4096) * 4096;
        let ptr = sys::alloc_rw(size)?;
        unsafe { core::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
        if let Err(e) = sys::make_rx(ptr, size) {
            sys::free(ptr, size);
            return Err(e);
        }
        Ok(Self { ptr, size })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for ExecMemory {
    fn drop(&mut self) {
        sys::free(self.ptr, self.size);
    }
}

#[cfg(windows)]
mod sys {
    use super::ExecError;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect,
    };

    pub fn alloc_rw(size: usize) -> Result<*mut u8, ExecError> {
        let p = unsafe {
            VirtualAlloc(
                core::ptr::null(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if p.is_null() {
            Err(ExecError::Os(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ))
        } else {
            Ok(p.cast())
        }
    }

    pub fn make_rx(p: *mut u8, size: usize) -> Result<(), ExecError> {
        let mut old = 0;
        let ok = unsafe { VirtualProtect(p.cast(), size, PAGE_EXECUTE_READ, &mut old) };
        if ok == 0 {
            Err(ExecError::Os(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ))
        } else {
            Ok(())
        }
    }

    pub fn free(p: *mut u8, _size: usize) {
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(unix)]
mod sys {
    use super::ExecError;

    pub fn alloc_rw(size: usize) -> Result<*mut u8, ExecError> {
        let p = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            Err(ExecError::Os(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ))
        } else {
            Ok(p.cast())
        }
    }

    pub fn make_rx(p: *mut u8, size: usize) -> Result<(), ExecError> {
        let rc = unsafe { libc::mprotect(p.cast(), size, libc::PROT_READ | libc::PROT_EXEC) };
        if rc != 0 {
            Err(ExecError::Os(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            ))
        } else {
            Ok(())
        }
    }

    pub fn free(p: *mut u8, size: usize) {
        unsafe { libc::munmap(p.cast(), size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{Assembler, Reg};

    #[test]
    fn runs_return_constant() {
        let mut a = Assembler::new();
        a.mov_ri(Reg::Rax, 42);
        a.ret();
        let mem = ExecMemory::new(&a.finish()).unwrap();
        let f: extern "C" fn() -> u64 = unsafe { core::mem::transmute(mem.as_ptr()) };
        assert_eq!(f(), 42);
    }

    #[test]
    fn first_argument_arrives_in_abi_register() {
        let mut a = Assembler::new();
        a.mov_rr(Reg::Rax, ARG0);
        a.add_ri(Reg::Rax, 1);
        a.ret();
        let mem = ExecMemory::new(&a.finish()).unwrap();
        let f: extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(mem.as_ptr()) };
        assert_eq!(f(41), 42);
    }

    extern "C" fn helper(x: u64) -> u64 {
        x * 3
    }

    #[test]
    fn calls_rust_helper_with_shadow_space() {
        let mut a = Assembler::new();
        a.push(Reg::Rbx);
        a.sub_ri(Reg::Rsp, SHADOW_SPACE);
        a.mov_rr(Reg::Rbx, ARG0);
        a.mov_rr(ARG0, Reg::Rbx);
        a.mov_ri(Reg::Rax, helper as *const () as u64);
        a.call_r(Reg::Rax);
        a.add_rr(Reg::Rax, Reg::Rbx);
        a.add_ri(Reg::Rsp, SHADOW_SPACE);
        a.pop(Reg::Rbx);
        a.ret();
        let mem = ExecMemory::new(&a.finish()).unwrap();
        let f: extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(mem.as_ptr()) };
        assert_eq!(f(10), 40);
    }

    #[test]
    fn empty_code_is_rejected() {
        assert!(ExecMemory::new(&[]).is_err());
    }
}
