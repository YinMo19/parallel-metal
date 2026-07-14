use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    NoMetalDevice,
    EmptyExtent,
    ExtentOverflow,
    TensorTooLarge { elements: usize },
    ShapeMismatch { expected: usize, actual: usize },
    ShaderCompile(String),
    FunctionLookup(String),
    PipelineCreation(String),
    CommandFailed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMetalDevice => formatter.write_str("no Metal device is available"),
            Self::EmptyExtent => formatter.write_str("zero-sized tensors are not supported yet"),
            Self::ExtentOverflow => formatter.write_str("tensor extent or byte size overflowed"),
            Self::TensorTooLarge { elements } => write!(
                formatter,
                "tensor has {elements} elements; the first runtime slice supports at most u32::MAX"
            ),
            Self::ShapeMismatch { expected, actual } => write!(
                formatter,
                "shape mismatch: expected {expected} elements, got {actual}"
            ),
            Self::ShaderCompile(error) => {
                write!(formatter, "Metal shader compilation failed: {error}")
            }
            Self::FunctionLookup(error) => write!(formatter, "Metal kernel lookup failed: {error}"),
            Self::PipelineCreation(error) => {
                write!(formatter, "Metal pipeline creation failed: {error}")
            }
            Self::CommandFailed(status) => write!(formatter, "Metal command failed: {status}"),
        }
    }
}

impl std::error::Error for Error {}
