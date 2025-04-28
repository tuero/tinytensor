# Datasets and Dataloaders

## Datasets
Datasets must satisfy the [IsDataset](../include/tt/data/dataset.h?plain=1#L29), concept.
- A typedef'd `DataType`, which is a tuple of arbitrary types
- A `size()` method which returns the number of samples the dataset contains
- A `get(int idx)` method to return the sample for the given index in range [0, size())

```cpp
template <typename T>
concept IsDataset = requires(const T ct, int idx) {
    typename T::DataType;
    requires IsSpecialization<typename T::DataType, std::tuple>;
    { ct.size() } -> std::same_as<int>;
    { ct.get(idx) } -> std::same_as<typename T::DataType>;
};
```

If your dataset simply contains tensors (which are created externally),
you can reuse and instantiate [TensorDataset](../include/tt/data/dataset.h?plain=1#L39).

```cpp
Tensor inputs = uniform_real(0, 1, {100, 4});
Tensor targets = uniform_real(0, 1, {100, 4});
data::TensorDataset train_data(inputs, targets);
```

## DatasetView
Views are created over datasets, such as the case when creating train/validation/test partitions.
They are also required to use with Dataloaders.
- Views can be created directly by moving the dataset into it, as is the case if the dataset is not to be partitioned
```cpp
data::DatasetView train_data_view(std::move(train_data));
```
- Or using [random_split](../include/tt/data/dataset.h?plain=1#L157), 
which creates views of various partition sizes that must sum to the total size of the dataset
```cpp
int seed = 0;
// dataset has a size() = 100
auto [train_data, validate_data, test_data] = data::random_split(std::move(dataset), seed, 70, 20, 10);
```

## Dataloaders
A [Dataloader](../include/tt/data/dataset.h?plain=1#L269) takes a `DatasetView` and provides batching and shuffling.
It also supports range-based loops with internal iterators. 
```cpp
Tensor inputs = uniform_real(0, 1, {100, 4});
Tensor targets = uniform_real(0, 1, {100, 4});
data::TensorDataset dataset(inputs, targets);

auto [train_data, validate_data, test_data] = data::random_split(std::move(dataset), seed, 70, 20, 10);

// Train loader shuffles its batches, validate/test loaders do not
int batch_size = 4;
auto train_loader = data::DataLoader(train_data, batch_size, true, seed);
auto validate_loader = data::DataLoader(validate_data, batch_size, false);
auto test_loader = data::DataLoader(test_data, batch_size, false);

// Iterate over the dataloader
// Since we supplied two tensors to our dataset, the iterated value will be a tuple of two batched tensors
for (auto [inp, tar] : train_loader) {
    // inp has shape (batch_size, 4) unless the last batch is partial
    // tar has shape (batch_size, 4) unless the last batch is partial
}
```
