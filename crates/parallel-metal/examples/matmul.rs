use parallel_metal::{Extent, Tensor, parallel};

/// Multiplies `[inner, rows]` by `[columns, inner]` using x-first coordinates.
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

fn main() -> parallel_metal::Result<()> {
    let rows = 3;
    let inner = 3;
    let columns = 3;
    let left = Tensor::from_slice(
        Extent::new([inner, rows]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )?;
    let right = Tensor::from_slice(
        Extent::new([columns, inner]),
        &[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )?;

    let output = matmul(&left, &right);
    assert_eq!(output.extent(), Extent::new([columns, rows]));

    for y in 0..rows {
        for x in 0..columns {
            let expected = (0..inner)
                .map(|k| left.as_slice()[k + inner * y] * right.as_slice()[x + columns * k])
                .sum::<f32>();
            let actual = output.as_slice()[x + columns * y];
            assert!(
                (actual - expected).abs() < 1e-3,
                "mismatch at ({x}, {y}): GPU={actual}, CPU={expected}"
            );
        }
    }

    let expected = [30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0];
    assert_eq!(output.as_slice(), &expected);

    println!("matmul extent: {:?}", output.extent());
    println!("shared CPU address: {:p}", output.cpu_address());
    print_matrix("left", &left);
    print_matrix("right", &right);
    print_matrix("left × right", &output);
    println!("verification: PASS");
    Ok(())
}

fn print_matrix(name: &str, matrix: &Tensor<f32, 2>) {
    let extent = matrix.extent();
    println!("{name}:");
    for y in 0..extent[1] {
        println!(
            "  {:?}",
            &matrix.as_slice()[y * extent[0]..(y + 1) * extent[0]]
        );
    }
}
