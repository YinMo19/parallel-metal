use parallel_metal::{Extent, Tensor, parallel};

#[parallel]
fn xor(left: &Tensor<u64, 1>, right: &Tensor<u64, 1>) -> Tensor<u64, 1> {
    left.parallel_iter()
        .zip(right.parallel_iter())
        .map(|(left, right)| *left ^ *right)
        .collect()
}

#[parallel]
fn affine_4d(input: &Tensor<f32, 4>, scale: f32, bias: f32) -> Tensor<f32, 4> {
    input
        .parallel_iter()
        .map(|value| *value * scale + bias)
        .collect()
}

#[parallel]
fn coordinates_2d(extent: Extent<2>) -> Tensor<u32, 2> {
    extent
        .parallel_iter()
        .map(|point| point[0] as u32 * 10_000 + point[1] as u32)
        .collect()
}

#[parallel]
fn coordinates_5d(extent: Extent<5>) -> Tensor<u32, 5> {
    extent
        .parallel_iter()
        .map(|point| {
            point[0] as u32 * 10_000
                + point[1] as u32 * 1_000
                + point[2] as u32 * 100
                + point[3] as u32 * 10
                + point[4] as u32
        })
        .collect()
}

#[parallel]
fn indexed_3d(input: &Tensor<u32, 3>, scale: u32) -> Tensor<u32, 3> {
    input
        .indexed_parallel_iter()
        .map(|(point, value)| {
            *value + point[0] as u32 * scale + point[1] as u32 * 10 + point[2] as u32
        })
        .collect()
}

#[parallel]
fn wave(extent: Extent<1>, time: f32) -> Tensor<f32, 1> {
    extent
        .parallel_iter()
        .map(|point| {
            let mut value: f32 = 0.0;
            for iteration in 1..=4 {
                let divisor: f32 = iteration as f32;
                value += sin(point[0] as f32 / divisor + time);
            }
            tanh(value)
        })
        .collect()
}

#[test]
fn gpu_zip_xor_matches_cpu() {
    let extent = Extent::new([1_000_003]);
    let left = Tensor::from_fn(extent, |point| point[0] as u64 * 17).unwrap();
    let right = Tensor::from_fn(extent, |point| {
        (point[0] as u64).rotate_left(13) ^ 0xa5a5_a5a5_a5a5_a5a5
    })
    .unwrap();

    let output = xor(&left, &right);

    for (index, &actual) in output.as_slice().iter().enumerate() {
        let expected =
            (index as u64 * 17) ^ ((index as u64).rotate_left(13) ^ 0xa5a5_a5a5_a5a5_a5a5);
        assert_eq!(actual, expected, "mismatch at element {index}");
    }
}

#[test]
fn cpu_write_then_gpu_map_preserves_four_dimensional_shape() {
    let extent = Extent::new([2, 3, 5, 7]);
    let mut input = Tensor::from_fn(extent, |point| {
        (point[0] * 1000 + point[1] * 100 + point[2] * 10 + point[3]) as f32
    })
    .unwrap();
    let address = input.cpu_address();

    for value in input.as_mut_slice() {
        *value += 0.5;
    }
    let output = affine_4d(&input, 2.0, -1.0);

    assert_eq!(input.cpu_address(), address);
    assert_eq!(output.extent(), extent);
    for (input, output) in input.as_slice().iter().zip(output.as_slice()) {
        assert_eq!(*output, *input * 2.0 - 1.0);
    }
}

#[test]
fn extent_iterator_reconstructs_two_dimensional_points() {
    let extent = Extent::new([37, 53]);
    let output = coordinates_2d(extent);

    assert_eq!(output.extent(), extent);
    for y in 0..extent[0] {
        for x in 0..extent[1] {
            assert_eq!(
                output.as_slice()[y * extent[1] + x],
                (y * 10_000 + x) as u32
            );
        }
    }
}

#[test]
fn indexed_iterator_reconstructs_three_dimensional_points() {
    let extent = Extent::new([5, 7, 11]);
    let input = Tensor::from_fn(extent, |_| 100u32).unwrap();
    let output = indexed_3d(&input, 100);

    for z in 0..extent[0] {
        for y in 0..extent[1] {
            for x in 0..extent[2] {
                let linear = (z * extent[1] + y) * extent[2] + x;
                assert_eq!(
                    output.as_slice()[linear],
                    100 + z as u32 * 100 + y as u32 * 10 + x as u32
                );
            }
        }
    }
}

#[test]
fn extent_iterator_flattens_and_reconstructs_rank_five() {
    let extent = Extent::new([2, 3, 4, 5, 6]);
    let output = coordinates_5d(extent);

    assert_eq!(output.extent(), extent);
    for (linear, &actual) in output.as_slice().iter().enumerate() {
        let point = extent.point_from_linear(linear);
        let expected = point[0] as u32 * 10_000
            + point[1] as u32 * 1_000
            + point[2] as u32 * 100
            + point[3] as u32 * 10
            + point[4] as u32;
        assert_eq!(actual, expected, "mismatch at point {point:?}");
    }
}

#[test]
fn device_locals_loop_and_math_intrinsics_match_cpu() {
    let extent = Extent::new([257]);
    let time = 0.75f32;
    let output = wave(extent, time);

    for (index, &actual) in output.as_slice().iter().enumerate() {
        let mut expected = 0.0f32;
        for iteration in 1..=4 {
            expected += (index as f32 / iteration as f32 + time).sin();
        }
        expected = expected.tanh();
        assert!(
            (actual - expected).abs() < 1e-5,
            "mismatch at {index}: GPU={actual}, CPU={expected}"
        );
    }
}
