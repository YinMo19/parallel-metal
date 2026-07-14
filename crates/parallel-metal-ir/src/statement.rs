use std::fmt::Write;

use crate::{Expr, ScalarType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
}

impl AssignOp {
    const fn msl(self) -> &'static str {
        match self {
            Self::Set => "=",
            Self::Add => "+=",
            Self::Sub => "-=",
            Self::Mul => "*=",
            Self::Div => "/=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Let {
        name: String,
        ty: ScalarType,
        value: Expr,
    },
    Assign {
        name: String,
        op: AssignOp,
        value: Expr,
    },
    ForRange {
        variable: String,
        start: u32,
        end: u32,
        inclusive: bool,
        body: Vec<Self>,
    },
}

impl Statement {
    pub(crate) fn write_msl(&self, output: &mut String, indentation: usize) {
        let indent = "    ".repeat(indentation);
        match self {
            Self::Let { name, ty, value } => {
                write!(output, "{indent}{} {name} = ", ty.msl_name()).unwrap();
                value.write_msl(output);
                output.push_str(";\n");
            }
            Self::Assign { name, op, value } => {
                write!(output, "{indent}{name} {} ", op.msl()).unwrap();
                value.write_msl(output);
                output.push_str(";\n");
            }
            Self::ForRange {
                variable,
                start,
                end,
                inclusive,
                body,
            } => {
                let comparison = if *inclusive { "<=" } else { "<" };
                writeln!(
                    output,
                    "{indent}for (uint {variable} = {start}; {variable} {comparison} {end}; ++{variable}) {{"
                )
                .unwrap();
                for statement in body {
                    statement.write_msl(output, indentation + 1);
                }
                writeln!(output, "{indent}}}").unwrap();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBlock {
    pub statements: Vec<Statement>,
    pub result: Expr,
}
