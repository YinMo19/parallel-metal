use std::fmt::Write;

use crate::{BinaryOp, ScalarType, UnaryOp};

/// An expression evaluated independently for each logical element.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Input(usize),
    InputAt {
        input: usize,
        coordinates: Vec<Self>,
    },
    InputExtentAxis {
        input: usize,
        axis: usize,
    },
    PointAxis(usize),
    ExtentAxis(usize),
    Scalar(String),
    Local(String),
    Literal(String),
    Call {
        function: String,
        arguments: Vec<Self>,
    },
    Unary {
        op: UnaryOp,
        value: Box<Self>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Self>,
        right: Box<Self>,
    },
    Cast {
        value: Box<Self>,
        ty: ScalarType,
    },
    Select {
        condition: Box<Self>,
        when_true: Box<Self>,
        when_false: Box<Self>,
    },
}

impl Expr {
    pub(crate) fn write_msl(&self, output: &mut String) {
        match self {
            Self::Input(index) => {
                write!(output, "__pm_in{index}[__pm_linear]").unwrap();
            }
            Self::InputAt { input, coordinates } => {
                write!(output, "__pm_in{input}[").unwrap();
                write_input_offset(output, *input, coordinates, 0);
                output.push(']');
            }
            Self::InputExtentAxis { input, axis } => {
                write!(output, "__pm_in{input}_extent[{axis}]").unwrap();
            }
            Self::PointAxis(axis) => {
                write!(output, "__pm_point{axis}").unwrap();
            }
            Self::ExtentAxis(axis) => {
                write!(output, "__pm_extent[{axis}]").unwrap();
            }
            Self::Scalar(name) | Self::Local(name) | Self::Literal(name) => output.push_str(name),
            Self::Call {
                function,
                arguments,
            } => {
                output.push_str(function);
                output.push('(');
                for (index, argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    argument.write_msl(output);
                }
                output.push(')');
            }
            Self::Unary { op, value } => {
                output.push('(');
                output.push_str(op.msl());
                value.write_msl(output);
                output.push(')');
            }
            Self::Binary { op, left, right } => {
                output.push('(');
                left.write_msl(output);
                output.push(' ');
                output.push_str(op.msl());
                output.push(' ');
                right.write_msl(output);
                output.push(')');
            }
            Self::Cast { value, ty } => {
                output.push_str(ty.msl_name());
                output.push('(');
                value.write_msl(output);
                output.push(')');
            }
            Self::Select {
                condition,
                when_true,
                when_false,
            } => {
                output.push('(');
                condition.write_msl(output);
                output.push_str(" ? ");
                when_true.write_msl(output);
                output.push_str(" : ");
                when_false.write_msl(output);
                output.push(')');
            }
        }
    }
}

fn write_input_offset(output: &mut String, input: usize, coordinates: &[Expr], axis: usize) {
    coordinates[axis].write_msl(output);
    if axis + 1 < coordinates.len() {
        write!(output, " + __pm_in{input}_extent[{axis}] * (").unwrap();
        write_input_offset(output, input, coordinates, axis + 1);
        output.push(')');
    }
}
