# Creating Tensors

Tensors have the following properties:
- `Device` device
- `ScalarType` dtype
- A flag for whether gradients should be saved for that particular tensor

## Device
Devices are a struct which contains a `backend` type and an `id`. 
Some backends ignore the device `id` (like `Backend::cpu`),
but are used in multi-device backends like `Backend::cuda`.
By default, `kCPU` is the device used which is a basic non-accelerated CPU backend implementation.
If building with CUDA support, the device `kCUDA` will be available,
which can be checked by `#ifdef TT_CUDA`.

The `cuda` backend supports multi-gpu. 
Like PyTorch, tensors can only perform operations against other tensors if on the 
same device, which includes both the `backend` and `id`,
so you will need to manually move tensors from one device to another.
```cpp
const Device cuda_device0{.backend = Backend::cuda, .id = 0};
const Device cuda_device1{.backend = Backend::cuda, .id = 1};
Tensor t1 = ones({4, 4}, TensorOptions().device(cuda_device0));
Tensor t2 = ones({4, 4}, TensorOptions().device(cuda_device1));

// Throws an exception because t1 and t2 are not on the same device 
//   (i.e. they are on separate GPUs)
// In function:
//         tinytensor::Tensor tinytensor::add(const Tensor&, const Tensor&)
// 
// TinyTensor Error: Expected inputs to be on same device, given devices cuda:0, cuda:1
Tensor result_throws = t1 + t2;

// Need to manually move so they are on the same GPU
Tensor t3 = t1.to(cuda_device1);
Tensor result = t3 + t2;
```

## ScalarType
The following scalar types are defined in [scalar.h](../include/tt/scalar.h)
```cpp
enum class ScalarType {
    bool8 = 0,    // 8 bit bool
    u8,           // 8 bit unsigned int
    i16,          // 16 bit int
    i32,          // 32 bit int
    i64,          // 64 bit int
    f32,          // 32 bit float
    f64,          // 64 bit float
};

// Shortnames
constexpr ScalarType kBool = ScalarType::bool8;
constexpr ScalarType kU8 = ScalarType::u8;
constexpr ScalarType kI16 = ScalarType::i16;
constexpr ScalarType kI32 = ScalarType::i32;
constexpr ScalarType kI64 = ScalarType::i64;
constexpr ScalarType kF32 = ScalarType::f32;
constexpr ScalarType kF64 = ScalarType::f64;

// Default scalar types
constexpr ScalarType kDefaultInt = ScalarType::i32;
constexpr ScalarType kDefaultFloat = ScalarType::f32;
```

When applying operations on tensors of different scalar types, 
the result will usually be a common type (usually following the C++ upcasting rules).
For example, the summation of a `kI32` tensor with a `kF32` tensor will be a `kF32`.

## TensorOptions
Most tensor construction methods take a [TensorOptions](../include/tt/tensor.h?plain=1#L54),
which lets you specify the device, scalar type, and if gradient tracking is required.

## Creating Tensors from External Data
Tensors can be created from vectors or initializer_lists.
The type of the Tensor will takeon the underlying type of the vector:
```cpp
// Create a tensor of type kF32 and shape (3, 2)
std::vector<float> v = {1, 2, 3, 4, 5, 6};
tinytensor::Tensor t(v, {3, 2}, kCPU);
```

## Tensor Creation Methods
The following tensor creation methods are supported, which follows from torch. 
See [here](../include/tt/tensor.h?plain=1#L1564) for full docs.
- `full`: Full of specified value
- `zeros`: Full of zeros
- `ones`: Full of ones
- `arange`: Values from range [0, N)
- `linspace`: Evenly spaced values over a range
- `eye`: 2D tensor of zeros, with ones on the main diagonal
- `one_hot`: 2d Tensor with zeros, except for ones at the specified indices

Tensors can also be created from supported distributions:
- `uniform_real`
- `uniform_int`
- `bernoulli`
- `binomial`
- `geometric`
- `poisson`
- `exponential`
- `normal`
- `cauchy`
- `lognormal`
- `weibull`
