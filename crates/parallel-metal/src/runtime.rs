use std::{cell::RefCell, collections::HashMap, ffi::c_void, mem::size_of};

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use crate::{Error, Extent, MetalElement, Result, Tensor};

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
}

struct Runtime {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl Runtime {
    fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(Error::NoMetalDevice)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines: HashMap::new(),
        })
    }

    fn pipeline(&mut self, source: &str, kernel_name: &str) -> Result<&ComputePipelineState> {
        if !self.pipelines.contains_key(source) {
            let library = self
                .device
                .new_library_with_source(source, &CompileOptions::new())
                .map_err(Error::ShaderCompile)?;
            let function = library
                .get_function(kernel_name, None)
                .map_err(Error::FunctionLookup)?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            self.pipelines.insert(source.to_owned(), pipeline);
        }

        Ok(self
            .pipelines
            .get(source)
            .expect("pipeline was inserted above"))
    }
}

fn with_runtime<T>(operation: impl FnOnce(&mut Runtime) -> Result<T>) -> Result<T> {
    RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Runtime::new()?);
        }
        operation(slot.as_mut().expect("runtime was initialized above"))
    })
}

pub(crate) fn allocate_shared<T: MetalElement>(len: usize) -> Result<Buffer> {
    let byte_len = len
        .checked_mul(size_of::<T>())
        .ok_or(Error::ExtentOverflow)?;
    if byte_len == 0 {
        return Err(Error::EmptyExtent);
    }

    objc::rc::autoreleasepool(|| {
        with_runtime(|runtime| {
            Ok(runtime
                .device
                .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared))
        })
    })
}

#[doc(hidden)]
pub struct BufferBinding<'a> {
    buffer: &'a Buffer,
    elements: usize,
    extent: Vec<usize>,
    shape_preserving: bool,
}

impl<'a> BufferBinding<'a> {
    pub fn source<T: MetalElement, const D: usize>(tensor: &'a Tensor<T, D>) -> Self {
        Self {
            buffer: &tensor.buffer,
            elements: tensor.len(),
            extent: tensor.extent().axes().to_vec(),
            shape_preserving: true,
        }
    }

    pub fn capture<T: MetalElement, const D: usize>(tensor: &'a Tensor<T, D>) -> Self {
        Self {
            buffer: &tensor.buffer,
            elements: tensor.len(),
            extent: tensor.extent().axes().to_vec(),
            shape_preserving: false,
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct ScalarBinding<'a> {
    bytes: *const c_void,
    byte_len: u64,
    _lifetime: std::marker::PhantomData<&'a c_void>,
}

impl<'a> ScalarBinding<'a> {
    pub fn new<T: MetalElement>(value: &'a T) -> Self {
        Self {
            bytes: (value as *const T).cast(),
            byte_len: size_of::<T>() as u64,
            _lifetime: std::marker::PhantomData,
        }
    }
}

#[doc(hidden)]
pub fn execute_elementwise<T: MetalElement, const D: usize>(
    source: &str,
    kernel_name: &str,
    extent: Extent<D>,
    inputs: &[BufferBinding<'_>],
    scalars: &[ScalarBinding<'_>],
) -> Result<Tensor<T, D>> {
    let elements = extent.element_count()?;
    let count = u32::try_from(elements).map_err(|_| Error::TensorTooLarge { elements })?;
    let mut metal_extent = [0u32; D];
    for (destination, axis) in metal_extent.iter_mut().zip(extent.axes()) {
        *destination = u32::try_from(axis).map_err(|_| Error::TensorTooLarge { elements })?;
    }

    for input in inputs {
        if input.shape_preserving && input.elements != elements {
            return Err(Error::ShapeMismatch {
                expected: elements,
                actual: input.elements,
            });
        }
    }
    let input_extents = inputs
        .iter()
        .map(|input| {
            input
                .extent
                .iter()
                .map(|&axis| {
                    u32::try_from(axis).map_err(|_| Error::TensorTooLarge {
                        elements: input.elements,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    objc::rc::autoreleasepool(|| {
        with_runtime(|runtime| {
            let output = runtime.device.new_buffer(
                elements
                    .checked_mul(size_of::<T>())
                    .ok_or(Error::ExtentOverflow)? as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Cloning a Metal foreign object only retains the Objective-C object;
            // it lets us release the mutable cache borrow before using the queue.
            let pipeline = runtime.pipeline(source, kernel_name)?.to_owned();

            let command_buffer = runtime.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&output), 0);

            for (index, input) in inputs.iter().enumerate() {
                encoder.set_buffer(index as u64 + 1, Some(input.buffer), 0);
            }

            let input_extent_start = inputs.len() as u64 + 1;
            for (index, extent) in input_extents.iter().enumerate() {
                encoder.set_bytes(
                    input_extent_start + index as u64,
                    size_of_val(extent.as_slice()) as u64,
                    extent.as_ptr().cast(),
                );
            }

            let scalar_start = inputs.len() as u64 * 2 + 1;
            for (index, scalar) in scalars.iter().enumerate() {
                encoder.set_bytes(scalar_start + index as u64, scalar.byte_len, scalar.bytes);
            }

            encoder.set_bytes(
                scalar_start + scalars.len() as u64,
                size_of::<u32>() as u64,
                (&raw const count).cast(),
            );
            encoder.set_bytes(
                scalar_start + scalars.len() as u64 + 1,
                size_of_val(&metal_extent) as u64,
                metal_extent.as_ptr().cast(),
            );

            let grid = MTLSize::new(elements as u64, 1, 1);
            let group = MTLSize::new(
                pipeline
                    .max_total_threads_per_threadgroup()
                    .min(elements as u64),
                1,
                1,
            );
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::CommandFailed(format!(
                    "{:?}",
                    command_buffer.status()
                )));
            }

            // SAFETY: every logical element was written by exactly one GPU thread
            // and the command completed before the tensor becomes CPU-visible.
            Ok(unsafe { Tensor::from_gpu_buffer(output, extent, elements) })
        })
    })
}
