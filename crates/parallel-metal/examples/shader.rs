use std::{error::Error, path::PathBuf};

use parallel_metal::{Extent, Tensor, parallel};

/// A scalarized Rust translation of a compact ShaderToy-style fragment shader.
///
/// Pixels are packed as 0xAABBGGRR so the CPU can export them without another
/// GPU readback allocation.
#[parallel]
fn shader(extent: Extent<2>, time: f32) -> Tensor<u32, 2> {
    extent
        .parallel_iter()
        .map(|point| {
            // p = (frag_coord * 2.0 - resolution) / resolution.y
            let px: f32 = ((point[1] as f32 + 0.5) * 2.0 - extent[1] as f32) / extent[0] as f32;
            let py: f32 = (((extent[0] - 1 - point[0]) as f32 + 0.5) * 2.0 - extent[0] as f32)
                / extent[0] as f32;

            let l: f32 = abs(0.7 - (px * px + py * py));
            let mut vx: f32 = px * (1.0 - l) / 0.2;
            let mut vy: f32 = py * (1.0 - l) / 0.2;
            let mut ox: f32 = 0.0;
            let mut oy: f32 = 0.0;
            let mut oa: f32 = 0.0;

            // Original golfed loop: i++ < 8, update v, then accumulate o.
            for iteration in 1..=8 {
                let i: f32 = iteration as f32;
                let dx: f32 = cos(vy * i + time) / i + 0.7;
                let dy: f32 = cos(vx * i + i + time) / i + 0.7;
                vx += dx;
                vy += dy;

                let distance: f32 = abs(vx - vy) * 0.2;
                ox += (sin(vx) + 1.0) * distance;
                oy += (sin(vy) + 1.0) * distance;
                oa += (sin(vx) + 1.0) * distance;
            }

            let attenuation: f32 = exp(-4.0 * l);
            let red: f32 = tanh(exp(py) * attenuation / ox);
            let green: f32 = tanh(exp(-py) * attenuation / oy);
            let blue: f32 = tanh(exp(-2.0 * py) * attenuation / oy);
            let alpha: f32 = tanh(attenuation / oa);

            (red * 255.0) as u32
                | (((green * 255.0) as u32) << 8)
                | (((blue * 255.0) as u32) << 16)
                | (((alpha * 255.0) as u32) << 24)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let extent = Extent::new([256, 512]);
    let pixels = shader(extent, 1.0);
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/parallel-metal-shader.ppm"));

    // PPM keeps the example dependency-free. This loop is ordinary CPU work
    // reading the exact shared allocation that the GPU just wrote.
    let mut ppm = format!("P6\n{} {}\n255\n", extent[1], extent[0]).into_bytes();
    ppm.reserve(pixels.len() * 3);
    for &pixel in pixels.as_slice() {
        ppm.extend_from_slice(&[
            (pixel & 0xff) as u8,
            ((pixel >> 8) & 0xff) as u8,
            ((pixel >> 16) & 0xff) as u8,
        ]);
    }
    std::fs::write(&output, ppm)?;

    println!("shader extent: {:?}", pixels.extent());
    println!("shared CPU address: {:p}", pixels.cpu_address());
    println!("wrote {}", output.display());
    Ok(())
}
