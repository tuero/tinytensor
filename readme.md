# tinytensor
A C++ multi-dimensional array + automatic differentiation library with CUDA acceleration.
- The syntax and API design follows closely to PyTorch's.
- The underlying Tensor backends use strided data layouts, so many view operations will create new `Tensor`s 
which share the same underlying data, similar to PyTorch.
- The largest difference is that most operations do not support automatic broadcasting.
I usually find it error prone if operations can automatically broadcast your input Tensors, 
so you will need to `reshape()` or `broadcast()` your Tensors to ensure the shapes match what is expected for the operations.

> [!WARNING]
> This is a personal project to learn how automatic differentiation engines are implemented.
> I make no guarantee that the implementations are correct and/or efficient, but lots of effort has gone 
> into the implementation and testing.
>
> I may experiment with various implementations later down the road to see how to optimize things,
> but I expect in general the API should remain consistent.

## Documentation
The documentation will not be complete as this is a personal project and not meant for serious use. 
However, the API and design follows PyTorch so users familiar with that should be able to find their way.
- [Creating Tensors](docs/tensor.md)
- [Supported Tensor Operations](docs/supported_ops.md)
- [Creating Neural Networks](docs/neural_net.md)
- [Datasets and Dataloaders](docs/dataloader.md)
- [Reproducibility](docs/rand.md)
- [Custom Autograd-Aware Functions](docs/autograd_custom.md)
- [Adding A Backend](docs/backend.md)

## Roadmap For Things To Implement
- AMD backend
- Accelerated CPU backend (OpenMP)
- CUDNN backend
- JIT backend, which compiles kernels on the fly to reduce number of kernel invocations
- Additional custom kernels to see performance differences

## Usage
See the [examples](examples/) for complete examples.

```cpp
#include <tt/tinytensor.h>

using namespace tinytensor;

#ifdef TT_CUDA
constexpr Device device = kCUDA;
#else
constexpr Device device = kCPU;
#endif

class Net : public nn::Module {
public:
    Net() : linear1(4, 32), linear2(32, 10) {
        register_module(linear1);
        register_module(linear2);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Net";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = relu(linear1.forward(input));
        result = linear2.forward(result);
        return result;
    }

private:
    nn::Linear linear1;
    nn::Linear linear2;
};

int main() {
    double learning_rate = 3e-4;
    uint64_t seed = 0;
    Tensor inputs = uniform_real(0, 1, {100, 4});
    Tensor targets = uniform_real(0, 1, {100, 10});
    data::TensorDataset dataset(inputs, targets);


    auto [train_data, validate_data, test_data] = data::random_split(std::move(dataset), seed, 70, 20, 10);

    // Train loader shuffles its batches, validate/test loaders do not
    int batch_size = 4;
    auto train_loader = data::DataLoader(train_data, batch_size, true, seed);
    auto validate_loader = data::DataLoader(validate_data, batch_size, false);
    auto test_loader = data::DataLoader(test_data, batch_size, false);

    Net net;
    net.to(device);
    optim::Adam optimizer(net.parameters_for_optimizer(), learning_rate);

    // Gradients are automatically tracked for computations involving network params
    net.train();
    
    for (auto [inp, tar] : train_loader) {
        inp = inp.to(device);
        tar = tar.to(device);
        optimizer.zero_grad();
        Tensor y = net.forward(inp);
        Tensor loss = nn::mse_loss(y, tar);
        loss.backward();
        optimizer.step();
    }

    // Scope a gradient guard to disable gradients and prevent memory consumption when not needed
    {
        const autograd::NoGradGuard guard;
        net.eval();
        for (auto [inp, tar] : test_loader) {
            inp = inp.to(device);
            tar = tar.to(device);
            Tensor y = net.forward(inp);
            Tensor loss = nn::mse_loss(y, tar);
            double loss_value = loss.item<double>();
        }
    }
    // After this point, the guard goes out of scope, and gradients are resumed tracking
}
```

## Building and Include To Your Project
A C++20 compiler is required, and CUDA-12+ if building with CUDA support. 
The following configurations have been tested:
- Ubuntu 24.04: `g++` 13.3.0, `clang++` 18.1.8, `nvcc` 12.6
- macOS 15.2: `g++` 14.2.0, `clang++` 19.1.6

Add the following to your `CMakeLists.txt`
```shell
include(FetchContent)

# If building with CUDA support
enable_language(CUDA) 

message("Configuring TinyTensor")
FetchContent_Declare(tinytensor
    GIT_REPOSITORY https://github.com/tuero/tinytensor.git
    GIT_TAG master
)
FetchContent_MakeAvailable(tinytensor)

# Example application with a single source main.cpp
add_executable(main main.cpp)
target_link_libraries(main tinytensor)
```

> [!IMPORTANT]
> To build with CUDA support, you must set the cmake flag `TT_BUILD_CUDA`:
```shell
mkdir build && cd build
cmake -DTT_BUILD_CUDA=1 ..
make
```

## Building Tests and Examples
The `CMakePresets.json` defines build options for the tests and exampes. 
- To build tests, the cmake flag `TT_BUILD_TESTS` must be set
- To build examples, the cmake flag `TT_BUILD_EXAMPLES` must be set

`CTest` is used to facilitate the testing. 
Some tests are optionally built if [libtorch](https://pytorch.org/) is present,
with its locations set through `Torch_DIR`.

Using the supplied `CMakePresets.json`:
```shell
# Optionally set to enable test which rely on libtorch
export Torch_DIR=/usr/local/libtorch/share/cmake/Torch/

cmake --preset=gcc-testing-cuda
cmake --build --preset=gcc-testing-cuda -- -j8
ctest --preset=gcc-testing-cuda
```

