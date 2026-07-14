use std::fmt::Write;

use crate::{DeviceBlock, ScalarType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarParam {
    pub name: String,
    pub ty: ScalarType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelInput {
    pub ty: ScalarType,
    pub rank: usize,
}

/// A single shape-preserving element kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementKernel {
    pub name: String,
    pub inputs: Vec<KernelInput>,
    pub scalars: Vec<ScalarParam>,
    pub output: ScalarType,
    /// Rank to reconstruct inside the kernel when the expression observes a point.
    pub logical_rank: Option<usize>,
    pub body: DeviceBlock,
}

impl ElementKernel {
    pub fn to_msl(&self) -> String {
        let mut source =
            String::from("#include <metal_stdlib>\n\nusing namespace metal;\n\nkernel void ");
        source.push_str(&self.name);
        source.push_str("(\n    device ");
        source.push_str(self.output.msl_name());
        source.push_str("* __pm_out [[buffer(0)]],\n");

        for (index, input) in self.inputs.iter().enumerate() {
            writeln!(
                source,
                "    const device {}* __pm_in{} [[buffer({})]],",
                input.ty.msl_name(),
                index,
                index + 1
            )
            .unwrap();
        }

        let input_extent_start = self.inputs.len() + 1;
        for (index, _) in self.inputs.iter().enumerate() {
            writeln!(
                source,
                "    constant uint* __pm_in{index}_extent [[buffer({})]],",
                input_extent_start + index
            )
            .unwrap();
        }

        let scalar_start = self.inputs.len() * 2 + 1;
        for (index, scalar) in self.scalars.iter().enumerate() {
            writeln!(
                source,
                "    constant {}& {} [[buffer({})]],",
                scalar.ty.msl_name(),
                scalar.name,
                scalar_start + index
            )
            .unwrap();
        }

        writeln!(
            source,
            "    constant uint& __pm_count [[buffer({})]],",
            scalar_start + self.scalars.len()
        )
        .unwrap();
        writeln!(
            source,
            "    constant uint* __pm_extent [[buffer({})]],",
            scalar_start + self.scalars.len() + 1
        )
        .unwrap();
        source.push_str("    uint __pm_linear [[thread_position_in_grid]])\n{\n");
        source.push_str("    if (__pm_linear < __pm_count) {\n");
        if let Some(rank) = self.logical_rank {
            source.push_str("        uint __pm_remaining = __pm_linear;\n");
            for axis in 0..rank {
                writeln!(
                    source,
                    "        uint __pm_point{axis} = __pm_remaining % __pm_extent[{axis}];"
                )
                .unwrap();
                if axis + 1 != rank {
                    writeln!(source, "        __pm_remaining /= __pm_extent[{axis}];").unwrap();
                }
            }
        }
        for statement in &self.body.statements {
            statement.write_msl(&mut source, 2);
        }
        source.push_str("        __pm_out[__pm_linear] = ");
        self.body.result.write_msl(&mut source);
        source.push_str(";\n    }\n}\n");
        source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssignOp, BinaryOp, DeviceBlock, Expr, Statement};

    #[test]
    fn emits_zip_xor_kernel() {
        let kernel = ElementKernel {
            name: "xor".into(),
            inputs: vec![
                KernelInput {
                    ty: ScalarType::U64,
                    rank: 1,
                },
                KernelInput {
                    ty: ScalarType::U64,
                    rank: 1,
                },
            ],
            scalars: vec![],
            output: ScalarType::U64,
            logical_rank: None,
            body: DeviceBlock {
                statements: vec![],
                result: Expr::Binary {
                    op: BinaryOp::BitXor,
                    left: Box::new(Expr::Input(0)),
                    right: Box::new(Expr::Input(1)),
                },
            },
        };

        let source = kernel.to_msl();
        assert!(source.contains("const device ulong* __pm_in1 [[buffer(2)]]"));
        assert!(source.contains("__pm_in0[__pm_linear] ^ __pm_in1[__pm_linear]"));
        assert!(source.contains("constant uint* __pm_in1_extent [[buffer(4)]]"));
        assert!(source.contains("constant uint& __pm_count [[buffer(5)]]"));
        assert!(source.contains("constant uint* __pm_extent [[buffer(6)]]"));
    }

    #[test]
    fn emits_logical_point_reconstruction() {
        let kernel = ElementKernel {
            name: "coordinates".into(),
            inputs: vec![],
            scalars: vec![],
            output: ScalarType::U32,
            logical_rank: Some(3),
            body: DeviceBlock {
                statements: vec![],
                result: Expr::PointAxis(1),
            },
        };

        let source = kernel.to_msl();
        assert!(source.contains("uint __pm_point0 = __pm_remaining % __pm_extent[0]"));
        assert!(source.contains("uint __pm_point1 = __pm_remaining % __pm_extent[1]"));
        assert!(source.contains("__pm_remaining /= __pm_extent[0]"));
        assert!(source.contains("__pm_out[__pm_linear] = __pm_point1"));
    }

    #[test]
    fn emits_indexed_input_and_dynamic_sum_loop() {
        let inputs = vec![
            KernelInput {
                ty: ScalarType::F32,
                rank: 2,
            },
            KernelInput {
                ty: ScalarType::F32,
                rank: 2,
            },
        ];
        let kernel = ElementKernel {
            name: "matmul".into(),
            inputs,
            scalars: vec![],
            output: ScalarType::F32,
            logical_rank: Some(2),
            body: DeviceBlock {
                statements: vec![
                    Statement::Let {
                        name: "sum".into(),
                        ty: ScalarType::F32,
                        value: Expr::Literal("0.0f".into()),
                    },
                    Statement::ForRange {
                        variable: "k".into(),
                        start: Expr::Literal("0".into()),
                        end: Expr::InputExtentAxis { input: 0, axis: 0 },
                        inclusive: false,
                        body: vec![Statement::Assign {
                            name: "sum".into(),
                            op: AssignOp::Add,
                            value: Expr::Binary {
                                op: BinaryOp::Mul,
                                left: Box::new(Expr::InputAt {
                                    input: 0,
                                    coordinates: vec![Expr::Local("k".into()), Expr::PointAxis(1)],
                                }),
                                right: Box::new(Expr::InputAt {
                                    input: 1,
                                    coordinates: vec![Expr::PointAxis(0), Expr::Local("k".into())],
                                }),
                            },
                        }],
                    },
                ],
                result: Expr::Local("sum".into()),
            },
        };

        let source = kernel.to_msl();
        assert!(source.contains("for (uint k = 0; k < __pm_in0_extent[0]; ++k)"));
        assert!(source.contains("__pm_in0[k + __pm_in0_extent[0] * (__pm_point1)]"));
        assert!(source.contains("__pm_in1[__pm_point0 + __pm_in1_extent[0] * (k)]"));
    }
}
