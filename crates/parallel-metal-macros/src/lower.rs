use std::collections::HashMap;

use parallel_metal_ir::{
    AssignOp, BinaryOp, DeviceBlock, Expr as IrExpr, ScalarType, Statement, UnaryOp,
};
use syn::{
    BinOp, Block, Expr, ExprClosure, ExprForLoop, ExprRange, Lit, Local, Pat, RangeLimits, Stmt,
};

use crate::syntax::{only_tail_expression, parse_scalar_type, simple_pat_ident};

#[derive(Clone)]
struct LocalInfo {
    msl_name: String,
    mutable: bool,
}

pub(crate) struct DeviceContext<'a> {
    pub(crate) bindings: &'a HashMap<String, usize>,
    pub(crate) tensor_bindings: &'a HashMap<String, (usize, usize)>,
    pub(crate) point_binding: Option<&'a str>,
    pub(crate) point_axes: &'a HashMap<String, usize>,
    pub(crate) extent_names: &'a [String],
    pub(crate) logical_rank: Option<usize>,
    pub(crate) scalars: &'a HashMap<String, ScalarType>,
    pub(crate) result_type: ScalarType,
}

pub(crate) fn lower_device_body(
    expression: &Expr,
    context: DeviceContext<'_>,
) -> syn::Result<DeviceBlock> {
    let mut locals = HashMap::new();

    if let Expr::Block(block) = expression {
        let (statements, result) =
            lower_block_parts(&block.block, &context, &mut locals, context.result_type)?;
        let result = result.ok_or_else(|| {
            syn::Error::new_spanned(
                &block.block,
                "a device map block must end with its output expression",
            )
        })?;
        Ok(DeviceBlock { statements, result })
    } else {
        let (statements, result) =
            lower_result_expression(expression, &context, &locals, context.result_type)?;
        Ok(DeviceBlock { statements, result })
    }
}

fn lower_block_parts(
    block: &Block,
    context: &DeviceContext<'_>,
    locals: &mut HashMap<String, LocalInfo>,
    result_type: ScalarType,
) -> syn::Result<(Vec<Statement>, Option<IrExpr>)> {
    let mut statements = Vec::new();
    let mut result = None;

    for (index, statement) in block.stmts.iter().enumerate() {
        let is_last = index + 1 == block.stmts.len();
        match statement {
            Stmt::Local(local) => statements.push(lower_local(local, context, locals)?),
            Stmt::Expr(expression, semicolon) => {
                if semicolon.is_none() && is_last && !matches!(expression, Expr::ForLoop(_)) {
                    let (prefix, expression) =
                        lower_result_expression(expression, context, locals, result_type)?;
                    statements.extend(prefix);
                    result = Some(expression);
                } else {
                    statements.push(lower_statement(expression, context, locals)?);
                }
            }
            Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(
                    item,
                    "items are not supported inside a device map block",
                ));
            }
            Stmt::Macro(statement) => {
                return Err(syn::Error::new_spanned(
                    statement,
                    "macros are not supported inside a device map block",
                ));
            }
        }
    }

    Ok((statements, result))
}

fn lower_result_expression(
    expression: &Expr,
    context: &DeviceContext<'_>,
    locals: &HashMap<String, LocalInfo>,
    result_type: ScalarType,
) -> syn::Result<(Vec<Statement>, IrExpr)> {
    if let Some((range, closure)) = range_map_sum(expression)? {
        let (start, end, inclusive) = lower_range(range, context, locals)?;
        let variable = simple_pat_ident(&closure.inputs[0])?;
        let rust_name = variable.to_string();
        let msl_name = loop_msl_identifier(&rust_name);
        let mut loop_locals = locals.clone();
        loop_locals.insert(
            rust_name,
            LocalInfo {
                msl_name: msl_name.clone(),
                mutable: false,
            },
        );
        let value = if let Expr::Block(block) = closure.body.as_ref() {
            lower_pure_block_expression(&block.block, context, &loop_locals)?
        } else {
            lower_expression(&closure.body, context, &loop_locals)?
        };
        let sum_name = "__pm_range_sum".to_owned();
        let zero = if result_type == ScalarType::F32 {
            "0.0f"
        } else {
            "0"
        };
        Ok((
            vec![
                Statement::Let {
                    name: sum_name.clone(),
                    ty: result_type,
                    value: IrExpr::Literal(zero.to_owned()),
                },
                Statement::ForRange {
                    variable: msl_name,
                    start,
                    end,
                    inclusive,
                    body: vec![Statement::Assign {
                        name: sum_name.clone(),
                        op: AssignOp::Add,
                        value,
                    }],
                },
            ],
            IrExpr::Local(sum_name),
        ))
    } else {
        Ok((vec![], lower_expression(expression, context, locals)?))
    }
}

fn range_map_sum(expression: &Expr) -> syn::Result<Option<(&Expr, &ExprClosure)>> {
    let Expr::MethodCall(sum) = expression else {
        return Ok(None);
    };
    if sum.method != "sum" || !sum.args.is_empty() {
        return Ok(None);
    }
    let Expr::MethodCall(map) = sum.receiver.as_ref() else {
        return Err(syn::Error::new_spanned(
            &sum.receiver,
            "device sum currently requires `(range).map(...).sum()`",
        ));
    };
    if map.method != "map" || map.args.len() != 1 {
        return Err(syn::Error::new_spanned(
            map,
            "device sum currently requires exactly one range map closure",
        ));
    }
    let Expr::Closure(closure) = &map.args[0] else {
        return Err(syn::Error::new_spanned(
            &map.args[0],
            "device range map requires an inline closure",
        ));
    };
    if closure.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &closure.inputs,
            "device range map closure must have exactly one argument",
        ));
    }
    Ok(Some((&map.receiver, closure)))
}

fn lower_local(
    local: &Local,
    context: &DeviceContext<'_>,
    locals: &mut HashMap<String, LocalInfo>,
) -> syn::Result<Statement> {
    let Pat::Type(typed) = &local.pat else {
        return Err(syn::Error::new_spanned(
            &local.pat,
            "device local variables require an explicit scalar type, e.g. let mut x: f32",
        ));
    };
    let ident = simple_pat_ident(&typed.pat)?;
    let rust_name = ident.to_string();
    if locals.contains_key(&rust_name) {
        return Err(syn::Error::new_spanned(
            ident,
            "shadowing device local variables is not supported yet",
        ));
    }
    let ty = parse_scalar_type(&typed.ty)?.ok_or_else(|| {
        syn::Error::new_spanned(
            &typed.ty,
            "device locals currently require a primitive scalar type",
        )
    })?;
    let initializer = local.init.as_ref().ok_or_else(|| {
        syn::Error::new_spanned(local, "device local variables require an initializer")
    })?;
    if initializer.diverge.is_some() {
        return Err(syn::Error::new_spanned(
            &initializer.expr,
            "let-else is not supported in device code",
        ));
    }
    let value = lower_expression(&initializer.expr, context, locals)?;
    let msl_name = local_msl_identifier(&rust_name);
    let mutable = typed_pat_is_mutable(&typed.pat);
    locals.insert(
        rust_name,
        LocalInfo {
            msl_name: msl_name.clone(),
            mutable,
        },
    );
    Ok(Statement::Let {
        name: msl_name,
        ty,
        value,
    })
}

fn lower_statement(
    expression: &Expr,
    context: &DeviceContext<'_>,
    locals: &HashMap<String, LocalInfo>,
) -> syn::Result<Statement> {
    match expression {
        Expr::Assign(assign) => {
            let name = assignment_target(&assign.left, locals)?;
            Ok(Statement::Assign {
                name,
                op: AssignOp::Set,
                value: lower_expression(&assign.right, context, locals)?,
            })
        }
        Expr::Binary(binary) => {
            let op = match binary.op {
                BinOp::AddAssign(_) => AssignOp::Add,
                BinOp::SubAssign(_) => AssignOp::Sub,
                BinOp::MulAssign(_) => AssignOp::Mul,
                BinOp::DivAssign(_) => AssignOp::Div,
                _ => {
                    return Err(syn::Error::new_spanned(
                        binary,
                        "only assignment expressions and bounded device loops may be statements",
                    ));
                }
            };
            let name = assignment_target(&binary.left, locals)?;
            Ok(Statement::Assign {
                name,
                op,
                value: lower_expression(&binary.right, context, locals)?,
            })
        }
        Expr::ForLoop(loop_expression) => lower_for_loop(loop_expression, context, locals),
        Expr::MethodCall(call) if call.method == "for_each" => {
            lower_for_each(&call.receiver, &call.args, context, locals)
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "only local assignment and bounded for/for_each loops are device statements",
        )),
    }
}

fn lower_for_loop(
    loop_expression: &ExprForLoop,
    context: &DeviceContext<'_>,
    outer_locals: &HashMap<String, LocalInfo>,
) -> syn::Result<Statement> {
    let variable = simple_pat_ident(&loop_expression.pat)?;
    let (start, end, inclusive) = lower_range(&loop_expression.expr, context, outer_locals)?;

    let rust_name = variable.to_string();
    let msl_name = loop_msl_identifier(&rust_name);
    let mut locals = outer_locals.clone();
    locals.insert(
        rust_name,
        LocalInfo {
            msl_name: msl_name.clone(),
            mutable: false,
        },
    );
    let body = lower_unit_block(&loop_expression.body, context, &mut locals)?;
    Ok(Statement::ForRange {
        variable: msl_name,
        start,
        end,
        inclusive,
        body,
    })
}

fn lower_for_each(
    receiver: &Expr,
    arguments: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    context: &DeviceContext<'_>,
    outer_locals: &HashMap<String, LocalInfo>,
) -> syn::Result<Statement> {
    if arguments.len() != 1 {
        return Err(syn::Error::new_spanned(
            arguments,
            "device range for_each expects exactly one closure",
        ));
    }
    let Expr::Closure(closure) = &arguments[0] else {
        return Err(syn::Error::new_spanned(
            &arguments[0],
            "device range for_each requires an inline closure",
        ));
    };
    if closure.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &closure.inputs,
            "device range for_each closure must have exactly one argument",
        ));
    }

    let variable = simple_pat_ident(&closure.inputs[0])?;
    let (start, end, inclusive) = lower_range(receiver, context, outer_locals)?;
    let rust_name = variable.to_string();
    let msl_name = loop_msl_identifier(&rust_name);
    let mut locals = outer_locals.clone();
    locals.insert(
        rust_name,
        LocalInfo {
            msl_name: msl_name.clone(),
            mutable: false,
        },
    );
    let body = lower_loop_closure_body(closure, context, &mut locals)?;

    Ok(Statement::ForRange {
        variable: msl_name,
        start,
        end,
        inclusive,
        body,
    })
}

fn lower_loop_closure_body(
    closure: &ExprClosure,
    context: &DeviceContext<'_>,
    locals: &mut HashMap<String, LocalInfo>,
) -> syn::Result<Vec<Statement>> {
    if let Expr::Block(block) = closure.body.as_ref() {
        lower_unit_block(&block.block, context, locals)
    } else {
        Ok(vec![lower_statement(&closure.body, context, locals)?])
    }
}

fn lower_unit_block(
    block: &Block,
    context: &DeviceContext<'_>,
    locals: &mut HashMap<String, LocalInfo>,
) -> syn::Result<Vec<Statement>> {
    let mut statements = Vec::new();
    for statement in &block.stmts {
        match statement {
            Stmt::Local(local) => statements.push(lower_local(local, context, locals)?),
            Stmt::Expr(expression, _) => {
                statements.push(lower_statement(expression, context, locals)?);
            }
            Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(
                    item,
                    "items are not supported inside a device loop",
                ));
            }
            Stmt::Macro(statement) => {
                return Err(syn::Error::new_spanned(
                    statement,
                    "macros are not supported inside a device loop",
                ));
            }
        }
    }
    Ok(statements)
}

fn lower_range(
    expression: &Expr,
    context: &DeviceContext<'_>,
    locals: &HashMap<String, LocalInfo>,
) -> syn::Result<(IrExpr, IrExpr, bool)> {
    let expression = strip_grouping(expression);
    let Expr::Range(ExprRange {
        start: Some(start),
        limits,
        end: Some(end),
        ..
    }) = expression
    else {
        return Err(syn::Error::new_spanned(
            expression,
            "device loops require a bounded range such as 0..count or 1..=8",
        ));
    };
    let start = lower_expression(start, context, locals)?;
    let end_expression = end;
    let end = lower_expression(end_expression, context, locals)?;
    let inclusive = matches!(limits, RangeLimits::Closed(_));
    if inclusive && integer_literal(end_expression) == Some(u32::MAX) {
        return Err(syn::Error::new_spanned(
            expression,
            "an inclusive device loop cannot end at u32::MAX",
        ));
    }
    Ok((start, end, inclusive))
}

fn strip_grouping(mut expression: &Expr) -> &Expr {
    loop {
        expression = match expression {
            Expr::Paren(paren) => &paren.expr,
            Expr::Group(group) => &group.expr,
            _ => return expression,
        };
    }
}

fn assignment_target(
    expression: &Expr,
    locals: &HashMap<String, LocalInfo>,
) -> syn::Result<String> {
    let Expr::Path(path) = expression else {
        return Err(syn::Error::new_spanned(
            expression,
            "device assignment target must be a local variable",
        ));
    };
    if path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            path,
            "device assignment target must be a local variable",
        ));
    }
    let name = path.path.segments[0].ident.to_string();
    let local = locals.get(&name).ok_or_else(|| {
        syn::Error::new_spanned(path, "device assignment target is not a local variable")
    })?;
    if !local.mutable {
        return Err(syn::Error::new_spanned(
            path,
            "device assignment requires a `let mut` local",
        ));
    }
    Ok(local.msl_name.clone())
}

fn lower_expression(
    expression: &Expr,
    context: &DeviceContext<'_>,
    locals: &HashMap<String, LocalInfo>,
) -> syn::Result<IrExpr> {
    match expression {
        Expr::Index(index) => {
            if let Expr::MethodCall(extent) = index.expr.as_ref()
                && extent.method == "extent"
                && extent.args.is_empty()
                && let Expr::Path(tensor) = extent.receiver.as_ref()
                && tensor.path.segments.len() == 1
            {
                let tensor = tensor.path.segments[0].ident.to_string();
                if let Some(&(input, rank)) = context.tensor_bindings.get(&tensor) {
                    let axis = index_literal(&index.index)?;
                    if axis >= rank {
                        return Err(syn::Error::new_spanned(
                            &index.index,
                            format!("axis {axis} is out of bounds for rank {rank}"),
                        ));
                    }
                    return Ok(IrExpr::InputExtentAxis { input, axis });
                }
            }

            let Expr::Path(base) = index.expr.as_ref() else {
                return Err(syn::Error::new_spanned(
                    index,
                    "device indexing requires a simple Tensor, Point, or Extent identifier",
                ));
            };
            if base.path.segments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    base,
                    "device indexing requires a simple identifier",
                ));
            }
            let base = base.path.segments[0].ident.to_string();
            if let Some(&(input, rank)) = context.tensor_bindings.get(&base) {
                let coordinates = tensor_coordinates(&index.index, rank)?
                    .iter()
                    .map(|coordinate| lower_expression(coordinate, context, locals))
                    .collect::<syn::Result<Vec<_>>>()?;
                return Ok(IrExpr::InputAt { input, coordinates });
            }

            let axis = index_literal(&index.index)?;
            let rank = context.logical_rank.ok_or_else(|| {
                syn::Error::new_spanned(index, "this iterator does not expose logical points")
            })?;
            if axis >= rank {
                return Err(syn::Error::new_spanned(
                    &index.index,
                    format!("axis {axis} is out of bounds for rank {rank}"),
                ));
            }
            if context.point_binding == Some(base.as_str()) {
                Ok(IrExpr::PointAxis(axis))
            } else if context.extent_names.contains(&base) {
                Ok(IrExpr::ExtentAxis(axis))
            } else {
                Err(syn::Error::new_spanned(
                    index,
                    "device indexing currently supports only the iterator Point or source Extent",
                ))
            }
        }
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            let name = path.path.segments[0].ident.to_string();
            if let Some(index) = context.bindings.get(&name) {
                Ok(IrExpr::Input(*index))
            } else if let Some(axis) = context.point_axes.get(&name) {
                Ok(IrExpr::PointAxis(*axis))
            } else if context.scalars.contains_key(&name) {
                Ok(IrExpr::Scalar(scalar_msl_identifier(&name)))
            } else if let Some(local) = locals.get(&name) {
                Ok(IrExpr::Local(local.msl_name.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    path,
                    "device expression references an unknown value",
                ))
            }
        }
        Expr::Call(call) => {
            let Expr::Path(function) = call.func.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &call.func,
                    "device intrinsic must be a simple function call",
                ));
            };
            if function.path.segments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    function,
                    "device intrinsic must be a simple function call",
                ));
            }
            let function = function.path.segments[0].ident.to_string();
            if !matches!(function.as_str(), "sin" | "cos" | "abs" | "exp" | "tanh") {
                return Err(syn::Error::new_spanned(
                    call,
                    "supported device math intrinsics are sin, cos, abs, exp, and tanh",
                ));
            }
            let arguments = call
                .args
                .iter()
                .map(|argument| lower_expression(argument, context, locals))
                .collect::<syn::Result<Vec<_>>>()?;
            if arguments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "this device math intrinsic expects one argument",
                ));
            }
            Ok(IrExpr::Call {
                function,
                arguments,
            })
        }
        Expr::Lit(literal) => match &literal.lit {
            Lit::Int(value) => Ok(IrExpr::Literal(value.base10_digits().to_owned())),
            Lit::Float(value) => Ok(IrExpr::Literal(format!("{}f", value.base10_digits()))),
            Lit::Bool(value) => Ok(IrExpr::Literal(value.value.to_string())),
            other => Err(syn::Error::new_spanned(
                other,
                "this literal is not supported in a device expression",
            )),
        },
        Expr::Paren(expression) => lower_expression(&expression.expr, context, locals),
        Expr::Group(expression) => lower_expression(&expression.expr, context, locals),
        Expr::Unary(expression) => match expression.op {
            syn::UnOp::Deref(_) => lower_expression(&expression.expr, context, locals),
            syn::UnOp::Neg(_) => Ok(IrExpr::Unary {
                op: UnaryOp::Neg,
                value: Box::new(lower_expression(&expression.expr, context, locals)?),
            }),
            _ => Err(syn::Error::new_spanned(
                expression,
                "only dereference and numeric negation are supported unary operators",
            )),
        },
        Expr::Binary(expression) => Ok(IrExpr::Binary {
            op: lower_binary_operator(&expression.op)?,
            left: Box::new(lower_expression(&expression.left, context, locals)?),
            right: Box::new(lower_expression(&expression.right, context, locals)?),
        }),
        Expr::Cast(expression) => {
            let ty = parse_scalar_type(&expression.ty)?.ok_or_else(|| {
                syn::Error::new_spanned(&expression.ty, "unsupported device cast target")
            })?;
            Ok(IrExpr::Cast {
                value: Box::new(lower_expression(&expression.expr, context, locals)?),
                ty,
            })
        }
        Expr::If(expression) => {
            let (_, when_false) = expression.else_branch.as_ref().ok_or_else(|| {
                syn::Error::new_spanned(expression, "device if expressions require an else branch")
            })?;
            Ok(IrExpr::Select {
                condition: Box::new(lower_expression(&expression.cond, context, locals)?),
                when_true: Box::new(lower_pure_block_expression(
                    &expression.then_branch,
                    context,
                    locals,
                )?),
                when_false: Box::new(lower_expression(when_false, context, locals)?),
            })
        }
        Expr::Block(expression) => lower_pure_block_expression(&expression.block, context, locals),
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported device expression in #[parallel] map",
        )),
    }
}

fn tensor_coordinates(expression: &Expr, rank: usize) -> syn::Result<Vec<&Expr>> {
    if rank == 1 && !matches!(expression, Expr::Tuple(_)) {
        return Ok(vec![expression]);
    }
    let Expr::Tuple(tuple) = expression else {
        return Err(syn::Error::new_spanned(
            expression,
            format!("rank-{rank} tensor indexing requires a {rank}-coordinate tuple"),
        ));
    };
    if tuple.elems.len() != rank {
        return Err(syn::Error::new_spanned(
            tuple,
            format!(
                "tensor index has {} coordinates but the tensor rank is {rank}",
                tuple.elems.len()
            ),
        ));
    }
    Ok(tuple.elems.iter().collect())
}

fn lower_pure_block_expression(
    block: &Block,
    context: &DeviceContext<'_>,
    locals: &HashMap<String, LocalInfo>,
) -> syn::Result<IrExpr> {
    let expression = only_tail_expression(block)?;
    lower_expression(expression, context, locals)
}

fn index_literal(expression: &Expr) -> syn::Result<usize> {
    let Expr::Lit(literal) = expression else {
        return Err(syn::Error::new_spanned(
            expression,
            "Point/Extent axis must be an integer literal",
        ));
    };
    let Lit::Int(index) = &literal.lit else {
        return Err(syn::Error::new_spanned(
            literal,
            "Point/Extent axis must be an integer literal",
        ));
    };
    index.base10_parse::<usize>()
}

fn integer_literal(expression: &Expr) -> Option<u32> {
    let Expr::Lit(literal) = expression else {
        return None;
    };
    let Lit::Int(value) = &literal.lit else {
        return None;
    };
    value.base10_parse::<u32>().ok()
}

fn lower_binary_operator(operator: &BinOp) -> syn::Result<BinaryOp> {
    Ok(match operator {
        BinOp::Add(_) => BinaryOp::Add,
        BinOp::Sub(_) => BinaryOp::Sub,
        BinOp::Mul(_) => BinaryOp::Mul,
        BinOp::Div(_) => BinaryOp::Div,
        BinOp::Rem(_) => BinaryOp::Rem,
        BinOp::BitXor(_) => BinaryOp::BitXor,
        BinOp::BitAnd(_) => BinaryOp::BitAnd,
        BinOp::BitOr(_) => BinaryOp::BitOr,
        BinOp::Shl(_) => BinaryOp::Shl,
        BinOp::Shr(_) => BinaryOp::Shr,
        BinOp::Eq(_) => BinaryOp::Eq,
        BinOp::Ne(_) => BinaryOp::Ne,
        BinOp::Lt(_) => BinaryOp::Lt,
        BinOp::Le(_) => BinaryOp::Le,
        BinOp::Gt(_) => BinaryOp::Gt,
        BinOp::Ge(_) => BinaryOp::Ge,
        BinOp::And(_) => BinaryOp::And,
        BinOp::Or(_) => BinaryOp::Or,
        _ => {
            return Err(syn::Error::new_spanned(
                operator,
                "assignment operators are valid only as device statements",
            ));
        }
    })
}

fn typed_pat_is_mutable(pattern: &Pat) -> bool {
    match pattern {
        Pat::Ident(pattern) => pattern.mutability.is_some(),
        Pat::Paren(pattern) => typed_pat_is_mutable(&pattern.pat),
        Pat::Reference(pattern) => typed_pat_is_mutable(&pattern.pat),
        Pat::Type(pattern) => typed_pat_is_mutable(&pattern.pat),
        _ => false,
    }
}

pub(crate) fn sanitize_identifier(identifier: &str) -> String {
    identifier
        .trim_start_matches("r#")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn scalar_msl_identifier(identifier: &str) -> String {
    format!("__pm_scalar_{}", sanitize_identifier(identifier))
}

fn local_msl_identifier(identifier: &str) -> String {
    format!("__pm_local_{}", sanitize_identifier(identifier))
}

fn loop_msl_identifier(identifier: &str) -> String {
    format!("__pm_loop_{}", sanitize_identifier(identifier))
}
