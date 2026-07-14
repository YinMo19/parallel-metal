use std::collections::HashMap;

use parallel_metal_ir::ScalarType;
use syn::{
    AngleBracketedGenericArguments, Block, Expr, ExprClosure, ExprMethodCall, GenericArgument,
    Ident, Lit, Pat, PathArguments, Stmt, Type, TypePath,
};

#[derive(Clone)]
pub(crate) struct TensorType {
    pub(crate) element_tokens: Type,
    pub(crate) element: ScalarType,
    pub(crate) rank: Expr,
}

pub(crate) fn only_tail_expression(block: &Block) -> syn::Result<&Expr> {
    match block.stmts.as_slice() {
        [Stmt::Expr(expression, None)] => Ok(expression),
        _ => Err(syn::Error::new_spanned(
            block,
            "the first #[parallel] slice requires the function body to be one iterator chain",
        )),
    }
}

pub(crate) fn expect_method<'a>(
    expression: &'a Expr,
    name: &str,
    argument_count: usize,
) -> syn::Result<&'a ExprMethodCall> {
    let Expr::MethodCall(call) = expression else {
        return Err(syn::Error::new_spanned(
            expression,
            format!("expected .{name}() in the parallel iterator chain"),
        ));
    };
    if call.method != name || call.args.len() != argument_count {
        return Err(syn::Error::new_spanned(
            call,
            format!("expected .{name}() with {argument_count} argument(s)"),
        ));
    }
    Ok(call)
}

#[derive(Clone)]
pub(crate) struct IteratorSource {
    pub(crate) ident: Ident,
    pub(crate) indexed: bool,
}

pub(crate) fn parse_sources(expression: &Expr) -> syn::Result<Vec<IteratorSource>> {
    let Expr::MethodCall(call) = expression else {
        return Err(syn::Error::new_spanned(
            expression,
            "expected parallel_iter(), copied(), or zip()",
        ));
    };

    match call.method.to_string().as_str() {
        "parallel_iter" | "indexed_parallel_iter" if call.args.is_empty() => {
            let Expr::Path(path) = call.receiver.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &call.receiver,
                    "parallel_iter() receiver must be a function parameter",
                ));
            };
            if path.path.segments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    path,
                    "parallel_iter() receiver must be a simple function parameter",
                ));
            }
            Ok(vec![IteratorSource {
                ident: path.path.segments[0].ident.clone(),
                indexed: call.method == "indexed_parallel_iter",
            }])
        }
        "copied" if call.args.is_empty() => parse_sources(&call.receiver),
        "zip" if call.args.len() == 1 => {
            let mut sources = parse_sources(&call.receiver)?;
            sources.extend(parse_sources(&call.args[0])?);
            Ok(sources)
        }
        _ => Err(syn::Error::new_spanned(
            call,
            "supported source adapters are parallel_iter(), indexed_parallel_iter(), copied(), and one zip()",
        )),
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ClosureMode {
    Values(usize),
    Indexed,
    Points,
}

pub(crate) struct ClosureBindings {
    pub(crate) values: HashMap<String, usize>,
    pub(crate) point: Option<String>,
    pub(crate) point_axes: HashMap<String, usize>,
}

pub(crate) fn closure_bindings(
    closure: &ExprClosure,
    mode: ClosureMode,
    point_rank: Option<usize>,
) -> syn::Result<ClosureBindings> {
    if closure.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &closure.inputs,
            "map closure must have exactly one argument",
        ));
    }

    let mut bindings = HashMap::new();
    let mut point_axes = HashMap::new();
    let pattern = &closure.inputs[0];
    let point = match mode {
        ClosureMode::Values(1) => {
            let ident = simple_pat_ident(pattern)?;
            bindings.insert(ident.to_string(), 0);
            None
        }
        ClosureMode::Values(input_count) => {
            if input_count == 0 {
                unreachable!("a values source always has an input");
            }
            let Pat::Tuple(tuple) = pattern else {
                return Err(syn::Error::new_spanned(
                    pattern,
                    "map after zip must destructure its argument as |(left, right)|",
                ));
            };
            if tuple.elems.len() != input_count {
                return Err(syn::Error::new_spanned(
                    tuple,
                    "zip tuple pattern does not match the number of inputs",
                ));
            }
            for (index, pattern) in tuple.elems.iter().enumerate() {
                bindings.insert(simple_pat_ident(pattern)?.to_string(), index);
            }
            None
        }
        ClosureMode::Indexed => {
            let Pat::Tuple(tuple) = pattern else {
                return Err(syn::Error::new_spanned(
                    pattern,
                    "indexed_parallel_iter() map must use |(point, value)|",
                ));
            };
            if tuple.elems.len() != 2 {
                return Err(syn::Error::new_spanned(
                    tuple,
                    "indexed_parallel_iter() map must use |(point, value)|",
                ));
            }
            let point = bind_point_pattern(
                &tuple.elems[0],
                point_rank.expect("indexed iteration has a point rank"),
                &mut point_axes,
            )?;
            let value = simple_pat_ident(&tuple.elems[1])?.to_string();
            bindings.insert(value, 0);
            point
        }
        ClosureMode::Points => bind_point_pattern(
            pattern,
            point_rank.expect("extent iteration has a point rank"),
            &mut point_axes,
        )?,
    };
    Ok(ClosureBindings {
        values: bindings,
        point,
        point_axes,
    })
}

fn bind_point_pattern(
    pattern: &Pat,
    rank: usize,
    axes: &mut HashMap<String, usize>,
) -> syn::Result<Option<String>> {
    if let Ok(ident) = simple_pat_ident(pattern) {
        return Ok(Some(ident.to_string()));
    }

    let Pat::Tuple(tuple) = pattern else {
        return Err(syn::Error::new_spanned(
            pattern,
            "point must be bound as `point` or destructured as `(axis0, axis1, ...)`",
        ));
    };
    if tuple.elems.len() != rank {
        return Err(syn::Error::new_spanned(
            tuple,
            format!(
                "point pattern has {} axes but this iterator has rank {rank}",
                tuple.elems.len()
            ),
        ));
    }

    for (axis, pattern) in tuple.elems.iter().enumerate() {
        if matches!(pattern, Pat::Wild(_)) {
            continue;
        }
        let ident = simple_pat_ident(pattern)?;
        if axes.insert(ident.to_string(), axis).is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "a point destructuring pattern cannot bind the same name twice",
            ));
        }
    }
    Ok(None)
}

pub(crate) fn simple_pat_ident(pattern: &Pat) -> syn::Result<&Ident> {
    match pattern {
        Pat::Ident(pattern) if pattern.subpat.is_none() => Ok(&pattern.ident),
        Pat::Reference(pattern) => simple_pat_ident(&pattern.pat),
        Pat::Type(pattern) => simple_pat_ident(&pattern.pat),
        Pat::Paren(pattern) => simple_pat_ident(&pattern.pat),
        _ => Err(syn::Error::new_spanned(
            pattern,
            "expected a simple identifier pattern",
        )),
    }
}

pub(crate) fn parse_tensor_type(ty: &Type) -> syn::Result<Option<TensorType>> {
    let ty = match ty {
        Type::Reference(reference) => reference.elem.as_ref(),
        other => other,
    };
    let Type::Path(path) = ty else {
        return Ok(None);
    };
    let Some(segment) = path.path.segments.last() else {
        return Ok(None);
    };
    if segment.ident != "Tensor" && segment.ident != "UnifiedTensor" {
        return Ok(None);
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(syn::Error::new_spanned(
            segment,
            "Tensor requires element and rank arguments",
        ));
    };
    parse_tensor_arguments(arguments).map(Some)
}

pub(crate) fn parse_extent_type(ty: &Type) -> syn::Result<Option<Expr>> {
    let Type::Path(path) = ty else {
        return Ok(None);
    };
    let Some(segment) = path.path.segments.last() else {
        return Ok(None);
    };
    if segment.ident != "Extent" {
        return Ok(None);
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(syn::Error::new_spanned(
            segment,
            "Extent requires one const rank argument",
        ));
    };
    let mut items = arguments.args.iter();
    let Some(GenericArgument::Const(rank)) = items.next() else {
        return Err(syn::Error::new_spanned(
            arguments,
            "Extent requires one const rank argument",
        ));
    };
    if items.next().is_some() {
        return Err(syn::Error::new_spanned(
            arguments,
            "Extent expects exactly one const rank argument",
        ));
    }
    Ok(Some(rank.clone()))
}

pub(crate) fn rank_literal(rank: &Expr) -> syn::Result<usize> {
    let Expr::Lit(literal) = rank else {
        return Err(syn::Error::new_spanned(
            rank,
            "coordinate-aware iteration currently requires a literal tensor rank",
        ));
    };
    let Lit::Int(rank) = &literal.lit else {
        return Err(syn::Error::new_spanned(
            literal,
            "coordinate-aware iteration currently requires a literal tensor rank",
        ));
    };
    rank.base10_parse::<usize>()
}

fn parse_tensor_arguments(arguments: &AngleBracketedGenericArguments) -> syn::Result<TensorType> {
    let mut items = arguments.args.iter();
    let Some(GenericArgument::Type(element_tokens)) = items.next() else {
        return Err(syn::Error::new_spanned(
            arguments,
            "Tensor's first argument must be an element type",
        ));
    };
    let element = parse_scalar_type(element_tokens)?.ok_or_else(|| {
        syn::Error::new_spanned(
            element_tokens,
            "the first implementation slice supports primitive Metal scalar elements",
        )
    })?;
    let Some(GenericArgument::Const(rank)) = items.next() else {
        return Err(syn::Error::new_spanned(
            arguments,
            "Tensor's second argument must be a const rank",
        ));
    };
    if items.next().is_some() {
        return Err(syn::Error::new_spanned(
            arguments,
            "Tensor expects exactly two generic arguments",
        ));
    }
    Ok(TensorType {
        element_tokens: element_tokens.clone(),
        element,
        rank: rank.clone(),
    })
}

pub(crate) fn parse_scalar_type(ty: &Type) -> syn::Result<Option<ScalarType>> {
    let Type::Path(TypePath { qself: None, path }) = ty else {
        return Ok(None);
    };
    let Some(segment) = path.segments.last() else {
        return Ok(None);
    };
    if !matches!(segment.arguments, PathArguments::None) {
        return Ok(None);
    }
    Ok(ScalarType::from_rust_name(&segment.ident.to_string()))
}
