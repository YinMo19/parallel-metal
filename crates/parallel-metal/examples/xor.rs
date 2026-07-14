use parallel_metal::{Extent, Tensor, parallel};

#[parallel]
fn xor(left: &Tensor<u64, 1>, right: &Tensor<u64, 1>) -> Tensor<u64, 1> {
    left.parallel_iter()
        .zip(right.parallel_iter())
        .map(|(left, right)| *left ^ *right)
        .collect()
}

fn main() -> parallel_metal::Result<()> {
    let extent = Extent::new([2 << 20]);
    let left = Tensor::from_fn(extent, |point| point[0] as u64)?;
    let right = Tensor::from_fn(extent, |point| {
        (point[0] as u64).rotate_left(17) ^ 0xfeed_face_cafe_beef
    })?;

    let output = xor(&left, &right);
    println!("extent: {:?}", output.extent());
    println!("shared CPU address: {:p}", output.cpu_address());
    println!("first 8 values: {:?}", &output.as_slice()[..8]);

    for (index, &actual) in output.as_slice().iter().enumerate() {
        let expected = (index as u64) ^ ((index as u64).rotate_left(17) ^ 0xfeed_face_cafe_beef);
        assert_eq!(actual, expected);
    }
    println!("verification: PASS");
    Ok(())
}
