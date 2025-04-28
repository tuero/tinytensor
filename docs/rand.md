# Reproducibility

Tensor operations or structs that use randomness will require either a seed, 
a [Generator](../include/tt/random.h),
or will use the global `Generator`.

The global generator can be set through the use of a seed
```cpp
uint64_t seed = 0;
set_default_generator_seed(seed);
```

Generator objects can also be created, and passed along as needed
```cpp
Generator gen(seed);
```
