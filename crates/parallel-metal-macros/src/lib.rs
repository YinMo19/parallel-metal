use proc_macro::TokenStream;
use syn::parse_macro_input;

mod expand;
mod lower;
mod syntax;

#[proc_macro_attribute]
/// Compile a supported `parallel_iter()` chain into a synchronous Metal kernel.
pub fn parallel(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[parallel] does not accept options in the first implementation slice",
        )
        .to_compile_error()
        .into();
    }

    let function = parse_macro_input!(item as syn::ItemFn);
    expand::parallel(function)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
