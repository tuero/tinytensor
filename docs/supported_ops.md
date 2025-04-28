# Supported Tensor Operations

## Tensor Storage
For a nice description of PyTorch's underlying storage implementation,
which this library follows,
see this nice [blog post](https://blog.ezyang.com/2019/05/pytorch-internals/).

Tensors have an underlying storage pointer, with a shape, stride, and offset.
This allows for cheap operations like reshapes, indexing, etc. 
where the underlying data pointer remains the same,
while the shape/stride/offsets are modifyed. 
This also means that making inplace modifications to views of a tensor will also have those 
modifications reflected in the original tensor.

## Supported Operations
Most common operations are supported on Tensors,
and also have autograd support.
These are defined in [tensor.h](../include/tt/tensor.h)
- [Element-wise Binary](../include/tt/tensor.h?plain=1#L2388): `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `||`, `&&`, `|`, `&`, `^`, `<<`, `>>`, 
`maximum`, `minimum`, `pow`
- [Element-wise Unary](../include/tt/tensor.h?plain=1#L2713): `abs`, `negate`, `logical_not`, `sign`, `log`, `log10`, `log2`, `log1p`,
`exp`, `exp2`, `expm1`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, 
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `tgamma`, `lgamma`, `digamma`,
`ceil`, `floor`, `round`, `isinf`, `isnan`, `isfinite`
- [Matmul](../include/tt/tensor.h?plain=1#L2470): vector-vector, vector-matrix, matrix-vector, matrix-matrix, batched matrix-matrix
- [Activations](../include/tt/tensor.h?plain=1#L2985): `sigmoid`, `log_sigmoid`, `hardsigmoid`, `softplus`, `relu`, `relu6`, `leaky_relu`,
`elu`, `selu`, `silu`, `hardtanh`, `softsign`, `softmax`, `log_softmax`
- [Shape Modifications](../include/tt/tensor.h?plain=1#L2168): `broadcast_to`, `expand`, `squeeze`, `unsqueeze`, `reshape`,
`flatten`, `permute`, `repeat_interleave`, `repeat`, `gather`
- [Indexing](../include/tt/tensor.h?plain=1#L2321): `index`, `index_select`, `index_put`
- [Reduction](../include/tt/tensor.h?plain=1#L2473): `min`, `argmin`, `max`, `argmax`, `sum`, `mean`, `all`, `any`, `var`
- [Misc](../include/tt/tensor.h?plain=1#L3111): `where`, `isclose`, `allclose`, `clamp`

## Indexing
Tenors support indexing.
- Indexing with an integer will select the row for the given dimenson
- Indexing with a `indexing::Slice` will select a slice from start-stop-step of the given dimension
- These can be combined with a list

See [index.h](../include/tt/index.h) for the indexing structs.
```cpp
Tensor x = uniform_real(0, 1, {4, 3, 5});
Tensor t1 = x[1];                   // t has shape [3, 5]
Tensor t2 = x[Slice(1, 3)];         // t has shape [2, 3, 5]
Tensor t3 = x[{Slice(), 1}];        // t has shape [4, 5], similar to pytorch x[:,1]
```

## Inplace Operations
Most operations support inplace versions, with a `_` suffix.
For example, `tensor.exp_()` will apply `exp` inplace. 
In general, inplace operations are not supported on tensors which require graidents to be computed. 


## Util
- `current_memory_allocated`: Current memory allocated in bytes
- `make_dot`: Create a dot graphivz of the ops and tensors for the computation graph up to and including the tensor
