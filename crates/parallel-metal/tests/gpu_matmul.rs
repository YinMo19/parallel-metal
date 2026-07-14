use parallel_metal::{Extent, Tensor, parallel};

#[parallel]
fn matmul(left: &Tensor<f32, 2>, right: &Tensor<f32, 2>) -> Tensor<f32, 2> {
    let left_extent = left.extent();
    let right_extent = right.extent();
    assert_eq!(left_extent[0], right_extent[1], "incompatible matrices");
    let output = Extent::new([right_extent[0], left_extent[1]]);

    output
        .parallel_iter()
        .map(|(x, y)| {
            (0..left.extent()[0])
                .map(|k| left[(k, y)] * right[(x, k)])
                .sum()
        })
        .collect()
}

#[test]
fn gpu_matrix_multiplication_matches_cpu() {
    let rows = 7;
    let inner = 11;
    let columns = 5;
    let left = Tensor::from_fn(Extent::new([inner, rows]), |point| {
        point[0] as f32 * 0.25 + point[1] as f32
    })
    .unwrap();
    let right = Tensor::from_fn(Extent::new([columns, inner]), |point| {
        point[0] as f32 - point[1] as f32 * 0.125
    })
    .unwrap();

    let output = matmul(&left, &right);
    assert_eq!(output.extent(), Extent::new([columns, rows]));
    for y in 0..rows {
        for x in 0..columns {
            let expected = (0..inner)
                .map(|k| left.as_slice()[k + inner * y] * right.as_slice()[x + columns * k])
                .sum::<f32>();
            let actual = output.as_slice()[x + columns * y];
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}
