use std::{fmt, marker::PhantomData, ptr};

use metal::Buffer;

use crate::{Error, Extent, Point, Result, runtime};

/// A plain scalar with an identical, explicitly known Metal representation.
///
/// # Safety
///
/// Implementors must be `Copy`, contain no references or drop glue, have the
/// same size/alignment as their MSL counterpart, and permit every bit pattern
/// that a generated shader can write.
pub unsafe trait MetalElement: Copy + Send + Sync + 'static {
    const MSL_NAME: &'static str;
}

macro_rules! metal_elements {
    ($($rust:ty => $msl:literal),+ $(,)?) => {$(
        // SAFETY: Rust and MSL define these scalar types with matching sizes,
        // alignment, and plain-value semantics on Metal targets.
        unsafe impl MetalElement for $rust {
            const MSL_NAME: &'static str = $msl;
        }
    )+};
}

metal_elements! {
    u8 => "uchar",
    u16 => "ushort",
    u32 => "uint",
    u64 => "ulong",
    i8 => "char",
    i16 => "short",
    i32 => "int",
    i64 => "long",
    f32 => "float",
}

/// An owned, contiguous rank-`D` tensor backed by one shared Metal buffer.
pub struct Tensor<T: MetalElement, const D: usize> {
    pub(crate) buffer: Buffer,
    extent: Extent<D>,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: MetalElement, const D: usize> Tensor<T, D> {
    pub fn from_fn(extent: Extent<D>, mut make: impl FnMut(Point<D>) -> T) -> Result<Self> {
        let len = extent.element_count()?;
        let buffer = runtime::allocate_shared::<T>(len)?;
        let destination = buffer.contents().cast::<T>();

        for linear in 0..len {
            // SAFETY: `allocate_shared` reserved `len * size_of::<T>()` bytes and
            // each element is written exactly once before the tensor is exposed.
            unsafe {
                destination
                    .add(linear)
                    .write(make(extent.point_from_linear(linear)))
            };
        }

        Ok(Self {
            buffer,
            extent,
            len,
            _element: PhantomData,
        })
    }

    pub fn from_slice(extent: Extent<D>, values: &[T]) -> Result<Self> {
        let len = extent.element_count()?;
        if values.len() != len {
            return Err(Error::ShapeMismatch {
                expected: len,
                actual: values.len(),
            });
        }

        let buffer = runtime::allocate_shared::<T>(len)?;
        // SAFETY: source and destination are valid for `len` elements and cannot
        // overlap because the destination is a new Metal allocation.
        unsafe {
            ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast::<T>(), len);
        }
        Ok(Self {
            buffer,
            extent,
            len,
            _element: PhantomData,
        })
    }

    pub const fn extent(&self) -> Extent<D> {
        self.extent
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the buffer contains `len` initialized `T` values. GPU entry
        // points are synchronous in this implementation, so none is in flight.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<T>(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same allocation argument as `as_slice`; `&mut self` guarantees
        // exclusive CPU access and all generated GPU calls are synchronous.
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents().cast::<T>(), self.len) }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    pub fn cpu_address(&self) -> *mut std::ffi::c_void {
        self.buffer.contents()
    }

    pub(crate) unsafe fn from_gpu_buffer(buffer: Buffer, extent: Extent<D>, len: usize) -> Self {
        Self {
            buffer,
            extent,
            len,
            _element: PhantomData,
        }
    }
}

impl<T: MetalElement, const D: usize> fmt::Debug for Tensor<T, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("element", &std::any::type_name::<T>())
            .field("extent", &self.extent)
            .field("len", &self.len)
            .field("cpu_address", &self.cpu_address())
            .finish()
    }
}
