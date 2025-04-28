# Neural Networks

## Creating Neural Networks
Neural networks should inherit from [nn::Module](../include/tt/nn/module.h),
and must implement the following:
- `name()`: String name of the module
- `forward()`: The forward method, which can take in any parameters and return any parameters

Nested modules must be registered, which allows recursive traversal of modules for things like saving, loading, and moving to/from devices.

```cpp
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
```

## Saving and Loading
Models can be saved and loaded.
Only the underlying data is saved, but not the underlying device. 
The tensors will be loaded onto the current device the model resides on.

```cpp
Net n1, n2;
n1.to(device);

// Train model
n1.save("model.pt");

// Load later
// Moving can be done before or after loading
n2.to(device);
n2.load("model.pt");
```

## Training Using Optimziers
Optimizers take the models parameters, and will control zeroing the model parameter gradients and updating them 
after the gradients have been set through a backward pass of the computation graph.

Note that here we use `parameters_for_optimizer`, NOT `parameters`.
Because we wish to track the parameter tensors, even through things like moving on devices,
we need an additional layer of indirection.
Here, we use `reference_wrapper` instead of baking the indirection in the underlying tensor storage.

```cpp
Net net;
net.to(device);
optim::Adam optimizer(net.parameters_for_optimizer(), learning_rate);

// ...

optimizer.zero_grad();              // Zero the stored gradients in the model parameter tensors
Tensor y = net.forward(input);
Tensor loss = nn::mse_loss(y, target);
loss.backward();                    // Set the gradients by following the computation graph
optimizer.step();                   // Apply the optimizer update rule using the stored gradients
```

## Neural Network Computation Modes
Some neural modules behave differently depending if they are training or evaluation mode,
such as `Dropout`.
These can be controled by calling `.train()` or `.eval()`
```cpp
Net net;
net.train();
// ...
net.eval();
```

## Disabling Gradient Computations
By default, all computations on `Modules` have gradients tracked,
which consumes memory until the gradients are cleared and dealt with.
When gradients are not required, such as during validation and testing,
you can disable the gradients by using a scoped graident guard.

```cpp
// Everything before here has gradients being tracked
{
    // While the guard is alive, gradients are not tracked
    // Generally, you should scope it so that you control when to enable/disable with precision
    const autograd::NoGradGuard guard;
}
// Everything after this has gradients being tracked again
```

## Applying Function To All Modules In A Neural Net
Its common to want to apply a function to all modules in a neural net,
such as applying a custom weight initialization.
When doing so, you can use `Module.as<LayerType>()`,
which will return if successful return a pointer to your layer if its underlying type is `LayerType`,
or `nullptr` otherwise. 
The underlying parameters of the layer can then be accessed, so long as they are not private.
```cpp
void init(nn::Module &m) {
    if (auto *layer = m.as<nn::Linear>()) {
        std::cout << "linear" << std::endl;
        std::cout << layer->bias.has_value() << std::endl;
        std::cout << layer->weight << std::endl;
    }
    if (auto *layer = m.as<nn::Conv2d>()) {
        std::cout << "conv" << std::endl;
        std::cout << layer->bias.has_value();
        std::cout << layer->weight << std::endl;
    }
}

Net net;
net.apply(init);
```

## Registering Tensors Parameters
If your custom neural layer requires a custom neural tensor parameter, 
then it must be registered like Modules. 
However, the tensor must exist as a `std::shared_ptr<Tensor>`,
so that the internals can be moved/disengaged such as is the case when moving between devices.
See some of the implemented Modules for examples.

## ModuleList
If you want to use a list of modules, 
you can use [ModuleList](../include/tt/nn/module_list.h).
Modules must be added as rvalue references,
and are stored as a `shared_ptr`, 
and are returned as a reference to `Module`, so you will need to cast using `.as<>()` (see above),
or `.as_checked<>()` which will return a reference to the casted type if exists or throw an exception otherwise.

```cpp
class Net : public nn::Module {
public:
    Net(int body_size) : linear1(4, 32), linear2(32, 10) {
        assert(body_size > 0);
        for (int _ : std::views::iota(0, body_size)) {
            body.push_back(nn::Linear(32, 32));
        }
        register_module(linear1);
        register_module(body);
        register_module(linear2);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Net";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = relu(linear1.forward(input));
        for (auto &layer : body) {
            result = relu(layer->as_checked<nn::Linear>().forward(result));
        }
        result = linear2.forward(result);
        return result;
    }

private:
    nn::Linear linear1;
    nn::ModuleList body;
    nn::Linear linear2;
};
```


## Implemented Optimizers
- [Adagrad](../include/tt/optim/adagrad.h)
- [Adam](../include/tt/optim/adam.h)
- [AdamW](../include/tt/optim/adamw.h)
- [RMSprop](../include/tt/optim/rmsprop.h)
- [SGD](../include/tt/optim/sgd.h)

## Implemented Neural Modules
- [Activation Functions](../include/tt/nn/activation.h)
    - `Sigmoid`
    - `LogSigmoid` 
    - `HardSigmoid`
    - `Softplys`
    - `ReLU`
    - `ReLU6`
    - `LeakyReLU`
    - `ELU`
    - `SELU`
    - `SiLU`
    - `Tanh`
    - `HardTanh`
    - `SoftSign`
    - `Softmax`
- [Batch Normalization](../include/tt/nn/batchnorm.h)
    - `BatchNorm1d`
    - `BatchNorm2d`
- [Instance Normalization](../include/tt/nn/instancenorm.h)
    - `InstanceNorm1d`
    - `InstanceNorm2d`
- [LayerNorm](../include/tt/nn/layernorm.h)
- [Linear](../include/tt/nn/linear.h)
- [Conv2d](../include/tt/nn/conv2d.h)
- [Dropout](../include/tt/nn/dropout.h)
- [Embedding](../include/tt/nn/embedding.h)
- Recurrent
    - [RNN](../include/tt/nn/rnn.h)
    - [LSTM](../include/tt/nn/lstm.h)
    - [GRU](../include/tt/nn/gru.h)
- [Pooling](../include/tt/nn/pool2d.h)
    - `MinPool2d`
    - `MaxPool2d`
    - `AvgPool2d`
- [Loss Functions](../include/tt/nn/loss.h)
    - `L1Loss`
    - `MSELoss`
    - `CrossEntropyLoss`
    - `NLLLoss`
    - `KLDivLoss`
    - `BCELoss`
    - `BCELossWithLogits`
    - `HuberLoss`
    - `SmoothL1Loss`
    - `SoftMarginLoss`

