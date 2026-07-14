use std::collections::{HashMap, HashSet};

use parallel_metal_ir::{ElementKernel, KernelInput, ScalarParam as IrScalarParam, ScalarType};
use quote::{ToTokens, quote};
use syn::{Expr, FnArg, Ident, ItemFn, Pat, ReturnType, Stmt};

use crate::lower::{DeviceContext, lower_device_body, sanitize_identifier, scalar_msl_identifier};
use crate::syntax::{
    ClosureMode, TensorType, closure_bindings, expect_method, parse_extent_type, parse_scalar_type,
    parse_sources, parse_tensor_type, rank_literal, simple_pat_ident, split_tail_expression,
};

#[derive(Clone)]
struct TensorParam {
    ident: Ident,
    ty: TensorType,
}

#[derive(Clone)]
struct ExtentParam {
    ident: Ident,
    rank: Expr,
}

#[derive(Clone)]
struct ScalarParam {
    ident: Ident,
    ty: ScalarType,
}

pub(crate) fn parallel(function: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    if function.sig.constness.is_some() {
        return Err(syn::Error::new_spanned(
            function.sig.constness,
            "GPU entry points cannot be const fn",
        ));
    }
    if function.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            function.sig.asyncness,
            "async #[parallel] functions are not implemented yet",
        ));
    }

    let mut tensors = HashMap::<String, TensorParam>::new();
    let mut tensor_params = Vec::<TensorParam>::new();
    let mut extents = HashMap::<String, ExtentParam>::new();
    let mut scalars = Vec::<ScalarParam>::new();

    for argument in &function.sig.inputs {
        let FnArg::Typed(argument) = argument else {
            return Err(syn::Error::new_spanned(
                argument,
                "#[parallel] is supported only on free functions",
            ));
        };
        let ident = simple_pat_ident(&argument.pat)?.clone();

        if let Some(ty) = parse_tensor_type(&argument.ty)? {
            let parameter = TensorParam { ident, ty };
            tensor_params.push(parameter.clone());
            tensors.insert(parameter.ident.to_string(), parameter);
        } else if let Some(rank) = parse_extent_type(&argument.ty)? {
            extents.insert(ident.to_string(), ExtentParam { ident, rank });
        } else {
            let ty = parse_scalar_type(&argument.ty)?.ok_or_else(|| {
                syn::Error::new_spanned(
                    &argument.ty,
                    "only Tensor, Extent, and primitive Metal scalar parameters are supported",
                )
            })?;
            scalars.push(ScalarParam { ident, ty });
        }
    }

    let output = match &function.sig.output {
        ReturnType::Type(_, ty) => parse_tensor_type(ty)?.ok_or_else(|| {
            syn::Error::new_spanned(
                ty,
                "#[parallel] currently requires a Tensor<T, D> return type",
            )
        })?,
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &function.sig,
                "#[parallel] currently requires a Tensor<T, D> return type",
            ));
        }
    };

    let (host_statements, chain) = split_tail_expression(&function.block)?;
    for statement in host_statements {
        let Stmt::Local(local) = statement else {
            continue;
        };
        let Ok(ident) = simple_pat_ident(&local.pat).cloned() else {
            continue;
        };
        if let Pat::Type(typed) = &local.pat {
            if let Some(rank) = parse_extent_type(&typed.ty)? {
                let parameter = ExtentParam { ident, rank };
                extents.insert(parameter.ident.to_string(), parameter);
            } else if let Some(ty) = parse_scalar_type(&typed.ty)? {
                scalars.push(ScalarParam { ident, ty });
            }
        } else if local
            .init
            .as_ref()
            .is_some_and(|initializer| is_extent_constructor(&initializer.expr))
        {
            let parameter = ExtentParam {
                ident,
                rank: output.rank.clone(),
            };
            extents.insert(parameter.ident.to_string(), parameter);
        }
    }
    let collect = expect_method(chain, "collect", 0)?;
    let map = expect_method(&collect.receiver, "map", 1)?;
    let closure = match &map.args[0] {
        Expr::Closure(closure) => closure,
        expression => {
            return Err(syn::Error::new_spanned(
                expression,
                "map currently requires an inline closure",
            ));
        }
    };

    let sources = parse_sources(&map.receiver)?;
    if sources.is_empty() || sources.len() > 2 {
        return Err(syn::Error::new_spanned(
            &map.receiver,
            "the first implementation slice supports one input or one zip of two inputs",
        ));
    }

    let mut input_params = Vec::with_capacity(sources.len());
    let mut extent_source = None::<ExtentParam>;
    for source in &sources {
        if let Some(parameter) = tensors.get(&source.ident.to_string()) {
            input_params.push(parameter.clone());
        } else if let Some(parameter) = extents.get(&source.ident.to_string()) {
            if sources.len() != 1 || source.indexed {
                return Err(syn::Error::new_spanned(
                    &source.ident,
                    "Extent supports only a direct parallel_iter() source",
                ));
            }
            extent_source = Some(parameter.clone());
        } else {
            return Err(syn::Error::new_spanned(
                &source.ident,
                "parallel iteration must start from a Tensor or Extent function parameter",
            ));
        }
    }
    if extent_source.is_some() && !input_params.is_empty() {
        return Err(syn::Error::new_spanned(
            &map.receiver,
            "an Extent source cannot be zipped with a tensor in this implementation slice",
        ));
    }

    let indexed_source = sources.iter().any(|source| source.indexed);
    if indexed_source && (sources.len() != 1 || extent_source.is_some()) {
        return Err(syn::Error::new_spanned(
            &map.receiver,
            "indexed_parallel_iter() currently supports one tensor source without zip",
        ));
    }

    let source_rank = if let Some(input) = input_params.first() {
        &input.ty.rank
    } else {
        &extent_source
            .as_ref()
            .expect("sources were validated above")
            .rank
    };
    let expected_rank = source_rank.to_token_stream().to_string();
    if output.rank.to_token_stream().to_string() != expected_rank {
        return Err(syn::Error::new_spanned(
            &function.sig.output,
            "shape-preserving collect must return the same tensor rank as its input",
        ));
    }
    for input in input_params.iter().skip(1) {
        if input.ty.rank.to_token_stream().to_string() != expected_rank {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "zip inputs must have the same tensor rank",
            ));
        }
    }

    let closure_mode = if extent_source.is_some() {
        ClosureMode::Points
    } else if indexed_source {
        ClosureMode::Indexed
    } else {
        ClosureMode::Values(sources.len())
    };
    let logical_rank = if matches!(closure_mode, ClosureMode::Points | ClosureMode::Indexed) {
        Some(rank_literal(source_rank)?)
    } else {
        None
    };
    let mut bindings = closure_bindings(closure, closure_mode, logical_rank)?;
    let tensor_indices = tensor_params
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.ident.to_string(), index))
        .collect::<HashMap<_, _>>();
    let source_tensor_indices = input_params
        .iter()
        .map(|input| tensor_indices[&input.ident.to_string()])
        .collect::<Vec<_>>();
    for input in bindings.values.values_mut() {
        *input = source_tensor_indices[*input];
    }
    let tensor_bindings = tensor_params
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            Ok((
                tensor.ident.to_string(),
                (index, rank_literal(&tensor.ty.rank)?),
            ))
        })
        .collect::<syn::Result<HashMap<_, _>>>()?;
    let scalar_types = scalars
        .iter()
        .map(|scalar| (scalar.ident.to_string(), scalar.ty))
        .collect::<HashMap<_, _>>();
    let extent_names = extent_source
        .iter()
        .map(|extent| extent.ident.to_string())
        .collect::<Vec<_>>();
    let body = lower_device_body(
        &closure.body,
        DeviceContext {
            bindings: &bindings.values,
            tensor_bindings: &tensor_bindings,
            point_binding: bindings.point.as_deref(),
            point_axes: &bindings.point_axes,
            extent_names: &extent_names,
            logical_rank,
            scalars: &scalar_types,
            result_type: output.element,
        },
    )?;

    let function_name = function.sig.ident.to_string();
    let kernel_name = format!("__pm_kernel_{}", sanitize_identifier(&function_name));
    let kernel = ElementKernel {
        name: kernel_name.clone(),
        inputs: tensor_params
            .iter()
            .map(|input| {
                Ok(KernelInput {
                    ty: input.ty.element,
                    rank: rank_literal(&input.ty.rank)?,
                })
            })
            .collect::<syn::Result<Vec<_>>>()?,
        scalars: scalars
            .iter()
            .map(|scalar| IrScalarParam {
                name: scalar_msl_identifier(&scalar.ident.to_string()),
                ty: scalar.ty,
            })
            .collect(),
        output: output.element,
        logical_rank,
        body,
    };
    let source = kernel.to_msl();
    let source = syn::LitStr::new(&source, function.sig.ident.span());
    let kernel_name = syn::LitStr::new(&kernel_name, function.sig.ident.span());

    let extent_expression = if let Some(input) = input_params.first() {
        let ident = &input.ident;
        quote!(#ident.extent())
    } else {
        let ident = &extent_source
            .as_ref()
            .expect("sources were validated above")
            .ident;
        quote!(#ident)
    };
    let other_inputs = input_params.iter().skip(1).map(|input| &input.ident);
    let source_tensor_names = input_params
        .iter()
        .map(|input| input.ident.to_string())
        .collect::<HashSet<_>>();
    let input_bindings = tensor_params.iter().map(|input| {
        let ident = &input.ident;
        if source_tensor_names.contains(&ident.to_string()) {
            quote!(::parallel_metal::__private::BufferBinding::source(#ident))
        } else {
            quote!(::parallel_metal::__private::BufferBinding::capture(#ident))
        }
    });
    let scalar_bindings = scalars.iter().map(|scalar| {
        let ident = &scalar.ident;
        quote!(::parallel_metal::__private::ScalarBinding::new(&#ident))
    });
    let output_element = &output.element_tokens;
    let output_rank = &output.rank;

    let attributes = &function.attrs;
    let visibility = &function.vis;
    let signature = &function.sig;

    Ok(quote! {
        #(#attributes)*
        #visibility #signature {
            #(#host_statements)*
            let __pm_extent = #extent_expression;
            #(
                assert_eq!(
                    #other_inputs.extent(),
                    __pm_extent,
                    "parallel zip inputs must have identical extents"
                );
            )*

            ::parallel_metal::__private::execute_elementwise::<#output_element, #output_rank>(
                #source,
                #kernel_name,
                __pm_extent,
                &[#(#input_bindings),*],
                &[#(#scalar_bindings),*],
            )
            .unwrap_or_else(|error| {
                panic!("generated Metal kernel {} failed: {}", #kernel_name, error)
            })
        }
    })
}

fn is_extent_constructor(expression: &Expr) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    path.path
        .segments
        .iter()
        .any(|segment| segment.ident == "Extent")
        && path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "new")
}
