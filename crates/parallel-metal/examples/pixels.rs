use parallel_metal::{Extent, Tensor, parallel};

#[parallel]
fn render(extent: Extent<2>, time: f32) -> Tensor<u32, 2> {
    extent
        .parallel_iter()
        .map(|(x, y)| x as u32 * 0x0001_0000 + y as u32 * 0x0000_0100 + time as u32)
        .collect()
}

fn main() {
    let extent = Extent::new([1920, 1080]);
    let pixels = render(extent, 42.0);

    println!("extent: {:?}", pixels.extent());
    println!("shared CPU address: {:p}", pixels.cpu_address());
    println!("top-left packed pixel: 0x{:08x}", pixels.as_slice()[0]);
    println!(
        "bottom-right packed pixel: 0x{:08x}",
        pixels.as_slice()[pixels.len() - 1]
    );
}
