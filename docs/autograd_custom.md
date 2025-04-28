# Custom Autograd-Aware Functions
All of the operations defined are autograd-aware, so writing a function of a composition of these functions
will also have gradients propagated through the operations.

For performance reasons, you may want to define your own function and handle the gradient computation yourself. 
To do so, you must define a struct which extents the CRTP class [TensorFunction](../include/tt/autograd.h?plain=1#L142),
which defines the following
- `auto forward(AutogradStorage &storage, bool is_grad_required, ...) -> Tensor`,
which takes as input `storage` to store data to be used later during the backward computation step,
`is_grad_required` which signals if a gradient will be required for this current computation
(which you can use to test if you should store anything),
and any additional params are the params which your functions will use.
- `auto backward(AutogradStorage &storage, const Tensor &grad_output) -> TensorList`,
which takes as input the `storage` from the `forward` method,
and `grad_output` which is the gradient with respect to the output of `forward()`.
The return value is a list of gradients, one for each input (and is with respect to that input). 

A limitation with the implementation is that autograd functions can only return a single `Tensor`.

To see examples of how this is done, see the defined autograd ops [here](../tinytensor/autograd/).
