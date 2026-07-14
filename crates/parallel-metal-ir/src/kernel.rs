use std::fmt::Write;

use crate::{DeviceBlock, ScalarType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarParam {
    pub name: String,
    pub ty: ScalarType,
}

/// A single shape-preserving element kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementKernel {
    pub name: String,
    pub inputs: Vec<ScalarType>,
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

        for (index, ty) in self.inputs.iter().enumerate() {
            writeln!(
                source,
                "    const device {}* __pm_in{} [[buffer({})]],",
                ty.msl_name(),
                index,
                index + 1
            )
            .unwrap();
        }

        let scalar_start = self.inputs.len() + 1;
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
    use crate::{BinaryOp, DeviceBlock, Expr};

    #[test]
    fn emits_zip_xor_kernel() {
        let kernel = ElementKernel {
            name: "xor".into(),
            inputs: vec![ScalarType::U64, ScalarType::U64],
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
        assert!(source.contains("constant uint& __pm_count [[buffer(3)]]"));
        assert!(source.contains("constant uint* __pm_extent [[buffer(4)]]"));
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
}
