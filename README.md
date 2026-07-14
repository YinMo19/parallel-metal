# parallel-metal

> 状态：Phase 1 原型已经可以在 Apple Metal GPU 上运行；公开 API 仍会快速演进。

`parallel-metal` 是一个面向 Apple Silicon 和 Metal 的 Rust 并行计算语言实验。核心目标是
让 GPU 成为普通函数内部的实现细节：调用者不创建 command queue，不计算 thread group，
也不调用 `.dispatch()`。

```rust,ignore
#[parallel]
fn xor(left: &Tensor<u64, 1>, right: &Tensor<u64, 1>)
    -> Tensor<u64, 1>
{
    left.parallel_iter()
        .zip(right.parallel_iter())
        .map(|(left, right)| *left ^ *right)
        .collect()
}

let output = xor(&left, &right); // 普通函数调用
cpu_consume(output.as_slice());  // 返回后继续 CPU 计算
```

核心抽象不是“数组”或“图片”，而是任意逻辑维度的共享 tensor：

```rust,ignore
Tensor<T, 1> // 数组
Tensor<T, 2> // 图片、矩阵
Tensor<T, 3> // 体数据
Tensor<T, 4> // 四维数据，轴含义由应用定义
```

Metal 的物理线程网格最多是三维，但库的逻辑维度不受此限制。调度器负责把任意维度映射
到 Metal 的 1D/2D/3D grid，并在 kernel 中恢复逻辑坐标。

坐标采用 GPU/图形领域直觉：`Extent<2>` 写作 `[width, height]`，point 是 `[x, y]`；
`Extent<3>` 写作 `[width, height, depth]`，point 是 `[x, y, z]`。axis 0 连续，所以二维
offset 是 `x + width * y`，三维 offset 是 `x + width * (y + height * z)`。四维及以上
继续使用相同的“低编号轴更连续”规则，但轴的业务含义由应用定义。

公开容器名采用简短的 `Tensor<T, D>`。shared Metal storage 是它的类型契约，不再编码进
类型名；早期原型中的 `UnifiedTensor` 暂时作为 deprecated alias 保留。

## 设计原则

- GPU 计算区域写在 `#[parallel]` 函数内部；
- 对外保持普通函数、参数和返回值语义；
- 大块数据由 `MTLStorageModeShared` buffer 持有，实现 CPU/GPU zero-copy；
- scalar/uniform 等小参数允许复制；
- 公开 device 语言采用 Rust 风格的惰性 `parallel_iter()` 链；
- 计算语义与物理线程调度分离；
- 同一套语义生成 CPU fallback 和 Metal kernel；
- 第一阶段优先支持纯函数和 out-of-place 输出，避免数据竞争；
- 不宣称可以编译任意 Rust，只支持明确、可验证的 device 子集；
- compute pipeline 与 graphics pipeline 分开建模，共享类型系统和编译基础设施。

## 当前可以运行的语法

第一条纵向切片已经实现：

- `Tensor<T, D>` 的 `StorageModeShared` 存储；
- 运行期 `Extent<D>` 和 axis-0-contiguous `Point<D>`；
- `tensor.parallel_iter().map(...).collect()`；
- 两个相同 shape tensor 的 `.zip()`；
- `tensor.indexed_parallel_iter()` 的 N 维逻辑坐标；
- `extent.parallel_iter()` 的无输入 N 维生成；
- 坐标既可写成 `|point| point[0]`，也可按 rank 解构成 `|(x, y)|`；
- parallel chain 前可以执行普通 host 端 shape 检查和输出 extent 推导；
- device closure 可以用 `tensor[(x, y, ...)]` 只读访问输入 tensor；
- primitive integer 和 `f32` 的算术、比较、位运算、cast 和简单 `if` 表达式；
- 显式类型的 scalar local、`let mut`/赋值、有界范围 `for`/`for_each`，以及局部
  `(range).map(...).sum()`；
- `sin`、`cos`、`abs`、`exp` 和 `tanh` device math intrinsic；
- MSL 生成、首次调用编译、thread-local pipeline cache 和同步执行。

例如直接在 GPU 上生成二维 packed pixels：

```rust,ignore
#[parallel]
fn render(extent: Extent<2>, time: f32) -> Tensor<u32, 2> {
    extent
        .parallel_iter()
        .map(|(x, y)| x as u32 * 65_536 + y as u32 * 256 + time as u32)
        .collect()
}
```

在 Apple Silicon Mac 上运行：

```sh
cargo test --workspace
cargo run --release -p parallel-metal --example xor
cargo run --release -p parallel-metal --example pixels
cargo run --release -p parallel-metal --example shader
cargo run --release -p parallel-metal --example matmul
```

`shader` example 是一段 ShaderToy 风格流体光球公式的标量化 Rust 翻译。GPU 逐像素执行，
CPU zero-copy 读取结果并写到 `/tmp/parallel-metal-shader.ppm`。

`matmul` example 用普通函数式 Rust 形状表达朴素矩阵乘法：host 端检查 shape，GPU closure
中按坐标读取两个输入，并用 `(0..left.extent()[0]).map(...).sum()` 计算 dot product。它是
语义案例，不是高性能 tiled GEMM。

## 当前明确限制

- 一个 chain 目前只支持一个 `map`，输入为一个 tensor、两个 tensor 的一次 zip，或一个
  `Extent`；
- 局部 iterator 目前只实现 range `for`/`for_each` 和直接作为 map 结果的
  `(range).map(...).sum()`；
- tensor 坐标读取尚未自动插入 bounds check 或证明索引安全，当前由 host shape 检查负责；
- `reduce`、`scan`、`filter`、`flat_map`、in-place kernel 和跨 chain fusion 尚未实现；
- 矩阵乘法还是逐输出元素的朴素 kernel，尚无 threadgroup tiling、向量化或专用调度；
- tensor element 和 scalar 目前只支持内置整数与 `f32`，`#[derive(MetalElement)]` 尚未实现；
- runtime 当前同步等待 GPU；
- GPU 初始化、MSL 编译或 command 执行失败时目前 panic，设计中的 CPU fallback 尚未接入；
- zero-sized tensor 和超过 `u32::MAX` element 的 dispatch 暂不支持；
- 项目当前只构建于 macOS。

完整目标和分阶段约束见 [DESIGN.md](DESIGN.md)。设计章节描述长期语义，本文的“当前明确
限制”描述此刻真实可用的实现范围。

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
