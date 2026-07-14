//! Backend-neutral expression IR for `parallel-metal` kernels.

mod expr;
mod kernel;
mod statement;
mod types;

pub use expr::Expr;
pub use kernel::{ElementKernel, KernelInput, ScalarParam};
pub use statement::{AssignOp, DeviceBlock, Statement};
pub use types::{BinaryOp, ScalarType, UnaryOp};
