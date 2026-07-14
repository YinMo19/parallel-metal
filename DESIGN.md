# parallel-metal 设计草案

> 本文描述目标架构和冻结语义。Phase 1 的首个纵向原型已经开始实现；示例代码中超出
> [README 当前范围](README.md#当前可以运行的语法)的部分仍是设计，不代表 API 已经存在。

## 1. 目标与非目标

### 1.1 目标

`parallel-metal` 希望提供一种普通 Rust 函数式的 GPU 计算体验：

```rust,ignore
let output = compute(&input, params);
let result = cpu_process(output.as_slice());
```

所有 Metal 细节都封装在被 `#[parallel]` 标记的函数中。宏和 runtime 负责：

- 识别显式并行区域；
- 生成 CPU 实现和 Metal Shading Language；
- 验证 Rust/MSL 类型布局；
- 分配共享输出；
- 绑定参数和资源；
- 选择物理 launch；
- 缓存 compute pipeline；
- 提交、同步和处理 fallback。

核心需要覆盖：

- 1D 数组和序列；
- 2D 图像与矩阵；
- 3D 体数据；
- 4D 及更高维 tensor；
- map、zip、stencil、reduce、scan 等不同并行算法；
- CPU 读写和 GPU 计算之间的 zero-copy 切换。

### 1.2 非目标

第一阶段不尝试：

- 编译任意 Rust 到 MSL；
- 隐藏普通 `[T; N]` 与 `MTLBuffer` 之间真实发生的复制；
- 自动把所有循环变成高性能 GPU kernel；
- 在没有约束的情况下允许共享可变别名；
- 用同一个抽象强行覆盖 compute、vertex、fragment 和完整图形管线；
- 在第一版实现跨终结操作/跨函数的 kernel fusion、异步 graph 或复杂编译优化。单条
  shape-preserving iterator chain 内的 adapter fusion 属于其基本语义。

## 2. 总体语言模型

一个 `#[parallel]` 函数仍然是 host 函数。函数体可以包含普通 host Rust，但 GPU 代码必须
位于一条显式的 `parallel_iter()` 链中：

```rust,ignore
#[parallel]
fn affine(input: &Tensor<f32, 1>, scale: f32, bias: f32)
    -> Tensor<f32, 1>
{
    input.parallel_iter()
        .map(|value| {
            // device Rust 子集
            *value * scale + bias
        })
        .collect()
}
```

这条 iterator chain 的边界非常重要：

- chain 外的代码由 CPU 执行；
- `map`、`zip`、`reduce` 等 combinator 的闭包降低为 device IR；
- 捕获的 scalar 变成 kernel 参数；
- 捕获的 tensor/view 变成 Metal buffer 参数；
- iterator 本身是惰性计划，只有 `collect`、`sum`、`reduce` 等终结操作才触发执行；
- `collect` 直接创建并返回一个新的共享输出 tensor。

这样既满足“GPU 只存在于函数内部”，也不需要宏猜测任意语句应该运行在 CPU 还是 GPU。

长期可以在一个函数中包含多个并行区域：

```rust,ignore
#[parallel]
fn normalize(input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
    let maximum = input
        .parallel_iter()
        .copied()
        .reduce(
            || f32::NEG_INFINITY,
            |left, right| left.max(right),
        );

    input.parallel_iter()
        .map(|value| *value / maximum)
        .collect()
}
```

这里 `reduce` 和 `collect` 是两个 kernel 阶段，但仍完全封装在一次普通函数调用中。

## 3. 统一的 N 维数据模型

### 3.1 核心类型

核心不为 array、image、volume 分别建立互不兼容的类型，而是使用：

```rust,ignore
pub struct Extent<const D: usize> {
    axes: [usize; D],
}

pub struct Point<const D: usize> {
    axes: [usize; D],
}

pub struct Strides<const D: usize> {
    elements: [usize; D],
}

pub struct Tensor<T, const D: usize> {
    // 拥有 MTLBuffer、extent、strides 和同步状态
}
```

`D` 是编译期已知的 rank，具体 extent 可以在运行期确定：

```rust,ignore
let vector_extent = Extent::<1>::new([2 << 20]);
let image_extent = Extent::<2>::new([1920, 1080]);
let volume_extent = Extent::<3>::new([256, 256, 128]);
let batch_extent = Extent::<4>::new([1920, 1080, 4, 8]);
```

使用运行期 extent 而不是把每条轴都放入 const generic，原因是：

- 任意 rank 的 API 保持一致；
- 窗口、图片、模型输入通常在运行期才知道尺寸；
- pipeline 不需要为每一个尺寸重新编译；
- shape 可以作为普通 kernel 参数传入；
- 避免为 1D、2D、3D 分别设计一套类型和宏。

编译期尺寸可以作为后续优化层，但不成为核心数据模型。

### 3.2 轴顺序和连续布局

默认采用 axis-0-contiguous 布局，第一条轴连续：

```text
extent  = [D0, D1, ..., Dn]
stride[0] = 1
stride[k] = extent[k - 1] * stride[k - 1]
offset(point) = Σ point[k] * stride[k]
```

因此：

- 二维图像使用 `[width, height]` 和 `[x, y]`；
- 三维体数据使用 `[width, height, depth]` 和 `[x, y, z]`；
- 四维及以上仍按 axis 0、axis 1、axis 2、axis 3 的顺序排列，业务含义由应用定义。

二维 offset 为 `x + width * y`，三维 offset 为
`x + width * (y + height * z)`。它直接匹配 Metal 的 `(x, y, z)` 坐标直觉；导入采用
NCHW 等其他布局的数据时，必须通过显式 view、transpose 或复制转换，不能暗中改变轴语义。

所有 extent 乘法和 byte size 计算必须检查溢出。`D == 0` 不作为 tensor；scalar 单独处理。

### 3.3 便利别名

下面的名称只是语义别名或轻量 view，不建立新的存储模型：

```rust,ignore
type Array<T> = Tensor<T, 1>;
type Image<T> = Tensor<T, 2>;
type Volume<T> = Tensor<T, 3>;
```

图像可以额外提供 `width()`、`height()`、`pixel(x, y)` 等便利方法，但底层仍是 rank-2
tensor。算法不能依赖这些别名才能工作。

### 3.4 View、切片和广播

一个完整的通用模型需要区分拥有者和 view：

```rust,ignore
TensorView<'a, T, D>       // 只读 offset + extent + strides
TensorViewMut<'a, T, D>    // 独占可写 view
```

view 不分配、不复制，只引用原始共享 buffer。transpose、crop、slice 和 reshape 优先生成
view。广播通过 stride 为 0 的只读 view 表示。

第一版输出保持连续布局；任意 strided output 会显著增加别名与写冲突分析，不进入首个
实现阶段。

## 4. 任意逻辑维度如何映射到 Metal

### 4.1 逻辑 domain 与物理 launch 分离

Metal 的 `MTLSize` 只有 width、height、depth，也就是最多三维物理 grid。这个限制不应该
进入用户算法。对逻辑 domain 调用 `parallel_iter()` 时，iterator item 是 `Point<D>`：

```rust,ignore
extent
    .parallel_iter()
    .map(|point: Point<D>| {
        // 与物理 thread_position_in_grid 的形状无关
    })
    .collect()
```

默认映射如下：

| 逻辑 rank | 默认物理映射 |
| --- | --- |
| 1 | axis 0 映射到 Metal x |
| 2 | axis 0、1 分别映射到 Metal x、y |
| 3 | axis 0、1、2 分别映射到 Metal x、y、z |
| 大于 3 | 生成 linear id，再根据 extent/strides 恢复 `Point<D>` |

即使 `D <= 3`，调度器也允许为了设备限制或 tiling 改用线性映射。算法只能依赖逻辑 point，
不能依赖具体的 Metal `uint3`。

### 4.2 高维还原

对连续 tensor，可从 linear id 还原任意维坐标：

```text
remaining = linear_id
for axis from 0 up to D-1:
    point[axis] = remaining % extent[axis]
    remaining   = remaining / extent[axis]
```

宏可以在 extent 为常量时消除部分除法；运行期 extent 则作为参数传入。对于只执行逐元素
map 且不读取坐标的 closure，编译器可以完全跳过坐标还原，直接使用 linear offset。

### 4.3 超大 domain 和设备限制

调度器负责：

- 查询每个物理维度和 threadgroup 的设备限制；
- 把逻辑 element count 分解到可用的 1D/2D/3D grid；
- 必要时提交多个 chunk，并给每个 chunk 传入 base linear offset；
- 使用 checked arithmetic 验证 element count 和 buffer byte size；
- 在可安全使用 `u32` 时生成较轻量索引，否则使用 `u64` 逻辑 offset。

这意味着高维支持不依赖 Metal 出现四维 thread grid。

## 5. Rust 风格的并行 iterator 语言

### 5.1 公开语法

公开 device 语言以惰性的 `parallel_iter()` 链为中心，而不是暴露 `generate`、kernel 或
dispatch：

```rust,ignore
input
    .parallel_iter()
    .map(|value| transform(*value))
    .collect()
```

它在阅读和组合方式上接近标准 `Iterator` 与 Rayon，但它不是任意 Rust `Iterator` 的自动
GPU 实现。只有库定义并被宏识别的 `ParallelIterator` combinator 可以进入 device IR。

数据源包括：

```rust,ignore
tensor.parallel_iter()          // item = &T，保留 tensor extent
tensor.indexed_parallel_iter()  // item = (Point<D>, &T)
extent.parallel_iter()          // item = Point<D>
view.parallel_iter()            // item = &T，保留 view extent/strides
```

固定 rank 的坐标允许直接按轴解构。tuple 顺序严格等于 `Extent` 的轴顺序；空间 domain 的
正式约定是 `[width, height, depth]` 对应 `(x, y, z)`：

```rust,ignore
Extent::new([width, height])
    .parallel_iter()
    .map(|(x, y)| shade(x, y))

volume
    .indexed_parallel_iter()
    .map(|((x, y, z), value)| transform(x, y, z, *value))
```

整体 `|point|` 与解构写法都属于正式语法。二维、三维 shader 通常用解构更易读；需要按
axis 编写通用算法、传递完整坐标或 rank 较高的时候，`point[axis]` 更稳定。`|(x, y)|`
不是 Rust 对 `Point<D>` 结构体本身的原生 pattern，而是 `#[parallel]` device DSL 在类型检查
前识别并降低的坐标 pattern。

`enumerate()` 保持 Rust 直觉，返回线性 `(usize, item)`；需要 N 维坐标时使用明确的
`indexed_parallel_iter()`，避免让 `enumerate()` 在不同 rank 下偷偷改变含义。

### 5.2 惰性与终结操作

`parallel_iter()`、`map()`、`zip()` 等只建立计划。真正执行发生在终结操作：

```rust,ignore
.collect()                 // 产生 Tensor
.sum()                     // 产生 scalar，多级 reduction
.reduce(identity, operation) // 产生 scalar，多级 reduction
.reduce_axis(axis, identity, operation) // 产生低一维 tensor
.any(predicate)
.all(predicate)
```

在 `#[parallel]` 函数中，宏收集整条 chain，做 shape/类型验证，降低为内部 primitive IR，
然后决定需要一个还是多个 Metal kernel。不存在一个可在函数外手动 `.dispatch()` 的 iterator
对象。

连续的 shape-preserving adapter 应当 fusion 成一个 kernel：

```rust,ignore
input.parallel_iter()
    .map(first)
    .map(second)
    .zip(other.parallel_iter())
    .map(combine)
    .collect()
```

这条 chain 默认只分配最终输出，不为每个 `map` 创建中间 tensor。只有 reduction、scan、
filter 等算法边界才会引入必要的多 pass 和中间 storage。

### 5.3 Shape 传播

parallel iterator 除 item type 外还携带逻辑 domain：

- `tensor.parallel_iter()` 的 domain 是 tensor extent；
- `map`、`copied`、`enumerate` 保持 extent；
- `zip` 要求双方 extent 相同，广播必须先显式创建 broadcast view；
- shape-preserving chain 的 `collect` 返回相同 rank/extent 的连续 `Tensor`；
- `extent.parallel_iter()` 允许按 point 生成一个全新的 tensor；
- `filter`、`flat_map` 等改变 cardinality，不能沿用原 extent。

因此二维 pixel 生成写成：

```rust,ignore
extent
    .parallel_iter()
    .map(|point| shade(point, params))
    .collect()
```

而两个 rank-4 tensor 的逐元素运算仍保持 rank-4 shape：

```rust,ignore
left.parallel_iter()
    .zip(right.parallel_iter())
    .map(|(left, right)| *left + *right)
    .collect()
```

### 5.4 Combinator 与内部 primitive

不同链式操作可能降低成完全不同的执行模型：

| iterator 操作 | 执行模型 | 设计要求 |
| --- | --- | --- |
| map/copied/zip | 单 pass、一个线程一个输出 | 第一阶段核心，可 fusion |
| indexed map/stencil | 单 pass 或 tiled pass | 输出必须与输入分离 |
| sum/min/max/reduce | 多级 reduction | 中间 buffer 和多次 dispatch |
| scan/prefix sum | 多 pass terminal/adapter | 专用算法 primitive |
| filter | predicate + scan + compact | 输出成为动态长度 rank-1 tensor |
| flat_map | 计数 + scan + 生成 | 动态输出，不能当作普通 map |
| gather | 单 pass | 输入只读，索引需边界策略 |
| scatter | 潜在写冲突 | 必须使用 atomic 或显式冲突策略 |
| sort | 多 pass | 库级 terminal，不从任意 closure 自动推导 |

`generate` 可以存在于编译器内部 IR，表示“一个 domain 的每个 point 生成一个输出”，但不
成为用户首先接触的公开语法。公开层统一是 chain，内部层统一是具有明确竞争与同步语义的
primitive。

不是每个标准 iterator 方法都应该立即支持。无法保证 shape、有限执行或 device 语义的
adapter 必须在宏展开时报清晰错误，不能悄悄回到 CPU 或改变复杂度。

### 5.5 两种 iterator 层级

device closure 内还可以有普通的、每线程局部的有限迭代。它和外层数据并行 chain 不属于
同一层：

```rust,ignore
extent.parallel_iter()                 // 跨 point 并行，一个 point 一个 GPU thread
    .map(|(x, y)| {
        let mut light: f32 = 0.0;
        (0..8).for_each(|sample| {      // 当前 GPU thread 内的局部循环
            light += trace(x, y, sample);
        });
        light
    })
    .collect()
```

`for sample in 0..8` 和 `(0..8).for_each(...)` 降到相同的 device loop IR，因此只是表达风格
不同，不会产生八次 dispatch。范围边界可以来自字面量、scalar 或 tensor extent，但必须是
启动 kernel 前已经确定的有限整数。局部 iterator 的 `map`、`fold`、`sum`、`any`、`all`
仍在单个 GPU thread 内执行，不能分配 heap 或产生动态长度集合。

外层 parallel iterator 按执行代价分组扩展：

| 类别 | 方法 | lowering 约束 |
| --- | --- | --- |
| 单 pass 可融合 | `map`、`zip`、`copied`、`enumerate`、`take`、`skip` | 尽量融合为一个 kernel；domain 变化必须静态可计算 |
| 邻域/重排 | `gather`、`windows`、`chunks`、view adapters | 单 pass，但必须携带 shape、stride 和边界策略 |
| reduction terminal | `fold`、`reduce`、`sum`、`min`、`max`、`any`、`all` | 多级 reduction，整数与浮点结合顺序需要明确 |
| 多 pass adapter | `scan`、`filter`、`flat_map` | 需要中间 buffer；动态 cardinality 不能伪装成普通 `map` |
| 冲突写入 | `scatter`、`for_each` 写外部 tensor | 默认拒绝；只有 atomic 或可证明无冲突时开放 |

目标是覆盖 GPU 上有明确语义的常用 iterator 组合，而不是声称任意 `std::iter::Iterator`
都能并行化。尤其外层 `for_each` 若允许任意 shared mutation，会破坏当前“一个 point 只写
一个输出”的无竞争模型；生成 tensor 仍优先用 `map(...).collect()`。

## 6. 1D、2D、3D 和更高维示例

### 6.1 一维 XOR

```rust,ignore
#[parallel]
fn xor(left: &Tensor<u64, 1>, right: &Tensor<u64, 1>)
    -> Tensor<u64, 1>
{
    assert_eq!(left.extent(), right.extent()); // host validation

    left.parallel_iter()
        .zip(right.parallel_iter())
        .map(|(left, right)| *left ^ *right)
        .collect()
}
```

### 6.2 二维像素生成

```rust,ignore
#[parallel]
fn render(extent: Extent<2>, params: Params) -> Tensor<Rgba8, 2> {
    extent
        .parallel_iter()
        .map(|(x, y)| {
            let uv = Vec2::new(
                x as f32 / extent[0] as f32,
                y as f32 / extent[1] as f32,
            );

            shade(uv, params)
        })
        .collect()
}
```

### 6.3 三维 stencil

```rust,ignore
#[parallel]
fn diffuse(input: &Tensor<f32, 3>) -> Tensor<f32, 3> {
    input
        .indexed_parallel_iter()
        .map(|(point, _value)| {
            let center = input.sample(point, Boundary::Clamp);
            let left   = input.sample(point.offset([-1, 0, 0]), Boundary::Clamp);
            let right  = input.sample(point.offset([ 1, 0, 0]), Boundary::Clamp);
            let up     = input.sample(point.offset([0, -1, 0]), Boundary::Clamp);
            let down   = input.sample(point.offset([0,  1, 0]), Boundary::Clamp);
            let front  = input.sample(point.offset([0, 0, -1]), Boundary::Clamp);
            let back   = input.sample(point.offset([0, 0,  1]), Boundary::Clamp);

            (center + left + right + up + down + front + back) / 7.0
        })
        .collect()
}
```

输入和输出分离，所以所有线程都只读取旧 volume，不会读到相邻线程刚写入的新值。

### 6.4 四维 tensor

```rust,ignore
#[parallel]
fn relu(input: &Tensor<f32, 4>) -> Tensor<f32, 4> {
    input
        .parallel_iter()
        .map(|value| (*value).max(0.0))
        .collect()
}
```

用户代码不需要知道这个 4D domain 被线性化还是分解到 Metal xyz grid。

### 6.5 矩阵乘法作为语义验收案例

矩阵使用 axis-0-contiguous 布局：左矩阵 shape 是 `[inner, rows]`，右矩阵是
`[columns, inner]`，输出是 `[columns, rows]`。目标代码保持普通 Rust 的 host 前处理与
函数式 dot product：

```rust,ignore
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
```

这个案例要求编译链明确区分三部分：chain 前的语句是普通 host Rust；外层 `parallel_iter`
建立输出 domain；内层 range iterator 是每个 GPU thread 内的顺序 reduction。输入 tensor
保持 zero-copy，只额外传递很小的 extent metadata。

当前纵向切片已经能执行这段朴素算法，但它也刻意暴露后续工作：

- 对坐标索引做静态 bounds proof，无法证明时插入可定义行为的边界策略；
- 把局部 `map/sum` 扩展成通用 `fold/min/max/any/all`，而不是特殊识别单一 chain；
- 为 GEMM 增加 tiled primitive、threadgroup memory 和向量化，同时保持上述公开函数不变；
- 支持 transpose/strided view 和 batch 维度，不要求用户为了布局先复制 tensor。

## 7. Zero-copy 内存模型

### 7.1 唯一底层存储

`Tensor` 使用 `MTLStorageModeShared` buffer 作为唯一数据存储：

```text
MTLBuffer allocation
├── CPU: Buffer::contents() 形成受控 slice
└── GPU: 同一个 MTLBuffer 绑定到 compute/graphics encoder
```

从 GPU 产生的新 tensor 由 runtime 直接分配 shared buffer，GPU 写入后把 tensor 返回给
调用者。这里没有“先生成普通数组，再转换为 Tensor”的步骤。

大块输入如果最初是普通 `[T]`、`Vec<T>` 或 `[T; N]`，转换到 `Tensor` 必须复制一次。
宏可以隐藏这次转换，但不能把它称为 zero-copy。真正端到端 zero-copy 的大块输入需要从
创建时就由 `Tensor` 或其他可证明兼容的 Metal storage 持有。

### 7.2 小参数

scalar、extent、strides 和小型 uniform 可以通过 `setBytes` 或 ring-buffer 复制。zero-copy
承诺针对大块 tensor 数据，不应为了几十字节参数引入复杂生命周期。

### 7.3 CPU/GPU 所有权状态

共享内存仍需要同步。概念状态机为：

```text
CPU-ready --submit GPU read/write--> GPU-in-flight
    ^                                  |
    |---------- wait/fence ------------|
```

安全规则：

- GPU-in-flight 时不能产生 CPU `&mut [T]`；
- CPU mutable guard 存在时不能提交访问同一 storage 的 GPU command；
- 只读输入可以被同一 command graph 中多个 kernel 共享；
- 可写 view 必须拥有不重叠的独占范围，或者使用显式 atomic 语义；
- CPU 写 guard 结束后，下一次 GPU submission 才能读取这些更改；
- GPU 写完成或 fence 满足后，CPU 才能读取结果。

第一版让 `#[parallel]` 函数同步返回，函数返回时结果是 CPU-ready。未来的异步 graph 可以
把 fence 保存在 tensor 内部，并在 CPU `read()` 时等待，但不能改变可观察结果和别名规则。

### 7.4 CPU 访问 API

建议正式的同步边界是 scoped guard：

```rust,ignore
let pixels = image.read();      // 必要时等待，得到只读 guard
let value = pixels[point];

let mut pixels = image.write(); // 必要时等待，得到独占 guard
pixels[point] = replacement;
```

同步实现阶段可以额外提供 `as_slice()`、`Index` 等便利接口。异步实现不能让一个看似廉价的
索引操作在无文档说明时产生不可见的长时间阻塞，因此 guard 是长期更稳定的核心协议。

## 8. 普通数组 bridge

为了调用便利，可以提供明确标注会复制的 bridge：

```rust,ignore
#[parallel(bridge)]
fn xor_native<const N: usize>(left: &[u64; N], right: &[u64; N]) -> [u64; N] {
    left.parallel_iter()
        .zip(right.parallel_iter())
        .map(|(left, right)| *left ^ *right)
        .collect()
}
```

概念流程为：

```text
native input --copy--> shared input
                         |
                         GPU
                         |
native output <--copy-- shared output
```

bridge 不能与核心 zero-copy API 混淆。对于 `[u64; 2 << 20]`，两个输入和一个输出的边界
复制总量是 48 MiB；像 XOR 这样的 memory-bound 工作可能因此不值得使用 GPU。

## 9. Device 语言和类型系统

### 9.1 支持的语法

第一阶段 device 闭包支持：

- 局部不可变/可变 POD 变量；
- scalar、vector、matrix 和 `Point<D>`；
- 算术、比较、位运算和显式转换；
- `if`、边界来自字面量/scalar/tensor extent 的有限循环；
- 只读 tensor/view 索引；
- 明确列入 device prelude 的数学函数；
- 调用被 `#[device]` 标记的纯辅助函数；
- 返回一个 `MetalElement` 作为当前 point 的输出。

第一阶段拒绝：

- heap allocation、`Vec`、`String`；
- trait object、动态 dispatch；
- recursion、panic、unwind；
- Rust 引用逃出 device 闭包；
- 任意标准库调用；
- 未声明语义的 shared mutation；
- 无法静态确定设备实现的泛型调用。

### 9.2 跨边界类型

所有进入 device 的值必须实现 sealed `MetalElement`。用户结构通过 derive 生成并验证：

```rust,ignore
#[derive(Clone, Copy, MetalElement)]
#[repr(C)]
struct Params {
    time: f32,
    scale: f32,
}
```

derive 必须验证：

- 所有字段都可映射到 MSL；
- Rust/MSL size、alignment 和 field offset 一致；
- 类型没有引用和析构逻辑；
- padding 被明确初始化或结构布局完全受控；
- `usize`、Rust `bool` 等 ABI 不稳定映射不会直接跨边界。

像 `Rgba8` 这样的公共类型宜使用明确布局，例如透明包装的 `u32`，避免 Rust `[u8; 4]`
与 MSL `uchar4` 的 alignment 差异。

### 9.3 中间表示

宏不应直接通过字符串拼接翻译 Rust 到 MSL。建议编译流程为：

```text
Rust syntax
   ↓ parse + supported-subset validation
device AST
   ↓ explicit type/layout resolution
backend-neutral device IR
   ├── CPU evaluator / fallback
   └── MSL code generator
```

即使项目只针对 Metal，中间 IR 也能统一处理：

- logical point 和 physical thread id 分离；
- bounds check；
- buffer bindings；
- 多维索引和 strides；
- constant folding；
- device helper 去重；
- 相同语义的 CPU/MSL 实现。

proc macro 无法访问 rustc 完整的类型检查结果，因此 device 类型需要在签名、intrinsic 和
derive metadata 中保持明确；不依赖宏猜测复杂的 trait inference。

## 10. 调度模型

### 10.1 语义不包含 threadgroup 大小

用户函数描述“每个逻辑 point 算什么”，调度器决定“物理线程怎样执行”。默认策略为
`Schedule::Auto`，根据 pipeline 和设备能力选择：

- grid 维度；
- threads per threadgroup；
- linear/native mapping；
- chunk 数量；
- 是否采用专用 tiled kernel。

公开函数调用永远不接收 launch config。

### 10.2 可选调度提示

复杂算法将来可以声明不改变语义的 hint：

```rust,ignore
#[parallel(schedule(tile = [8, 8, 4]))]
fn diffuse(...) -> ... { ... }
```

hint 只能影响性能，不能影响结果正确性。设备不支持时调度器可以选择其他合法配置。

### 10.3 Threadgroup memory

barrier 和 threadgroup memory 不能安全地伪装成普通 `parallel_iter().map()`。需要单独的高级
primitive 或受约束 kernel builder，并明确要求一个 threadgroup 内所有线程遵守相同的
barrier control flow。卷积、矩阵乘法和 block reduction 可以由库提供经过验证的实现，
而不是要求普通用户手写同步。

## 11. 同步、错误和 CPU fallback

### 11.1 第一阶段：同步函数

公开函数的第一版语义为：

1. 验证 shape 和参数；
2. 选择 CPU 或 GPU backend；
3. 如果选择 GPU，提交并等待 command buffer；
4. 成功后返回 CPU-ready tensor 或 scalar；
5. GPU 初始化/编译/执行失败时，对纯 out-of-place 计算执行 CPU fallback。

同步语义符合“普通函数返回值已经算好”的直觉，也让错误能够在函数返回前确定。

### 11.2 自动 backend 选择

每个 primitive 可以根据以下信息选择 CPU/GPU：

- element count；
- 估算的计算强度；
- pipeline 是否已经缓存；
- 输入当前由 CPU 还是 GPU 最近使用；
- GPU 是否可用；
- 用户或 benchmark profile 设置的阈值。

阈值是性能策略，不改变函数语义。

### 11.3 未来：异步 graph

为了让连续 GPU 函数避免每次等待，未来可以增加内部 command graph：

```text
GPU function A -> pending tensor -> GPU function B -> pending tensor
                                                    |
                                              first CPU read waits
```

异步模式必须先解决错误传播、CPU guard、跨线程访问和资源生命周期。它是同步模型之上的
优化层，不应迫使第一版公开 API 暴露 `.dispatch()`。

## 12. Compute 与图形渲染的边界

“每个 pixel 并行生成颜色”可以作为 rank-2 compute：

```rust,ignore
fn render(...) -> Tensor<Rgba8, 2>
```

这种结果可以由 CPU zero-copy 读取和修改。要显示到窗口，runtime 再用 GPU fullscreen pass
或 blit 把它送入 `CAMetalDrawable`；drawable 是窗口系统拥有的独立显示资源。

真正的 rasterization 不是普通 N 维 tensor 计算。vertex/fragment shader 还涉及：

- vertex assembly；
- interpolation；
- derivatives；
- depth/stencil；
- discard 和 blending；
- render attachments。

因此完整架构应该共享 device 类型和编译 IR，但使用不同入口：

```rust,ignore
#[parallel] // compute domain
fn compute(...) -> Tensor<...> { ... }

#[vertex]   // graphics stage，未来设计
fn vertex(...) -> VertexOutput { ... }

#[fragment] // graphics stage，未来设计
fn fragment(...) -> Color { ... }
```

不能因为二维 compute 也能生成 pixel，就把 fragment shader 的全部语义塞进 `Extent<2>`。

## 13. 所有权、别名和数据竞争

第一阶段采用最容易证明的规则：

- 输入 tensor/view 只读；
- shape-preserving chain 的 `collect` 总是分配独立输出；
- 每个 point 只写自己的一个输出元素；
- 输出不能在 device closure 中作为输入被读取；
- 不允许捕获可变 Rust 引用；
- stencil 必须读旧输入、写新输出；
- shape mismatch 在 launch 前由 host 检查。

以后允许的 in-place map 也必须限制为“线程只读写同一 linear index”，否则需要显式 atomic
或算法级同步。跨 threadgroup 不存在普通 barrier，宏不能暗示其存在。

## 14. Pipeline、缓存和构建

建议第一阶段在首次调用时生成/编译 MSL，并通过函数体 hash、类型布局和 device identity
缓存 pipeline。尺寸作为 runtime 参数，不应成为 cache key，除非选择了尺寸专用优化。

成熟后可增加 build-time `.metallib`：

- shader 错误提前到构建期；
- 减少首次调用延迟；
- 允许更完整的 reflection 和布局验证。

无论使用 runtime MSL 还是 `.metallib`，公开函数和 device IR 不改变。

## 15. 建议的工程边界

当前 workspace 按以下边界建立：

```text
parallel-metal/
├── parallel-metal          # 公开 tensor、runtime 和 prelude
├── parallel-metal-macros   # #[parallel]、#[device]、derive
├── parallel-metal-ir       # device IR、验证和 MSL codegen
└── examples                # 1D/2D/3D/4D 与 CPU/GPU 验证
```

宏 crate 不直接管理 Metal 对象；runtime 不解析 Rust syntax；IR/codegen 不持有应用资源。

## 16. 实现前冻结的核心决定

以下是本草案给出的明确建议：

1. 核心容器采用 `Tensor<T, const D: usize>`，具体 extent 运行期保存。
2. 采用 axis 0 连续的默认布局；空间坐标按 `(x, y, z)` 排列。
3. `Array/Image/Volume` 只是 rank 便利层，不是独立 backend。
4. 用户算法只看 `Point<D>`；物理 Metal grid 最多三维且由调度器决定。
5. `D > 3` 使用 linearization/unflatten，不限制逻辑 rank。
6. `parallel_iter()` chain 是公开语言；内部 generate/map/reduce primitive 不暴露 dispatch。
7. host/device 以 iterator combinator closure 为边界，不猜测任意 Rust 语句。
8. 大块数据只有 shared tensor 路径承诺 zero-copy；原生数组 bridge 明确发生复制。
9. 第一阶段只做纯 out-of-place 计算和同步函数语义。
10. CPU fallback 与 MSL 必须来自同一个 device IR 语义。
11. compute 与 graphics stage 分开建模，共享类型、IR 和 runtime 基础设施。
12. 调度策略与计算语义分离，为 tiling、多 pass 和未来 graph 留出空间。

## 17. 分阶段实现范围

### Phase 1：语义闭环

- `Tensor<T, D>` shared storage；
- `Extent<D>`、`Point<D>`、连续 strides；
- `parallel_iter()`、`map`、`zip`、`copied` 和 `collect`；
- scalar capture 和只读 tensor capture；
- CPU fallback；
- 1D/2D/3D/4D 相同结果验证；
- 同步返回。

### Phase 2：View 与通用索引

- slice、crop、transpose、reshape；
- 只读 strided view 和广播；
- 边界采样与 stencil；
- native/linear launch planner。

### Phase 3：算法 primitive

- reduce；
- scan；
- tiled convolution/matmul；
- threadgroup memory 的受约束模型；
- benchmark 驱动的 CPU/GPU backend 选择。

### Phase 4：执行与图形扩展

- 异步 command graph 和延迟 CPU 同步；
- kernel fusion；
- build-time metallib；
- 独立的 vertex/fragment stage 设计；
- 窗口 present 集成。

阶段划分是为了控制实现风险。Phase 1 已从 shape-preserving iterator 的纵向切片开始；
后续阶段不允许破坏前面冻结的 shape、storage、ownership 和 device IR 语义。

## 18. 设计验收问题

开始实现前，应能对以下问题给出确定答案：

- 一个 rank-5 tensor 如何映射到 Metal 三维 grid？
- 一个 transpose view 如何在线性 buffer 上寻址？
- 哪些类型可以安全跨越 Rust/MSL ABI？
- CPU 在什么时刻可以获得 `&mut [T]`？
- 哪些 primitive 天然 race-free，哪些需要 atomic 或多 pass？
- GPU 失败后，是否能够安全地重新执行 CPU 版本？
- pipeline cache 是否错误地依赖具体 extent？
- pixel compute 与 fragment shader 为什么不是同一个入口模型？
- 用户何时发生了真实复制，API 是否明确说明？
- 将来增加异步执行时，是否保持现有同步 API 的结果语义？

runtime 和 proc macro 的首个原型必须持续用这些问题作为验收标准；扩展到下一阶段前，相关
契约需要有实现与测试共同证明。
