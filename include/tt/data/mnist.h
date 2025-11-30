// mnist.h
// MNIST dataset

#ifndef TINYTENSOR_DATAST_MNIST_H_
#define TINYTENSOR_DATAST_MNIST_H_

#include <tt/export.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include <string>
#include <tuple>

namespace tinytensor::data {

// IsDataset compliant wrapper for MNIST
// Users must download the *-images-idx3-ubyte and *-labels-idx1-ubyte data files
class TINYTENSOR_EXPORT MNISTDataset {
public:
    using DataType = std::tuple<Tensor, Tensor>;

    /**
     * Create a IsDataset compiant wrapper for MNIST
     * @param img_path The full path to the *-images-idx3-ubyte file
     * @param label_path The full path to the *-labels-idx1-ubyte file
     * @param normalize True to normalize the data, false to leave rescaled to [0, 1]
     */
    MNISTDataset(const std::string &img_path, const std::string &label_path, bool normalize = true);

    /**
     * Shape of each image
     */
    [[nodiscard]] auto img_shape() const -> Shape;

    /**
     * Number of samples in the dataset
     */
    [[nodiscard]] auto size() const -> int;

    /**
     * Return tuple of image and label
     * @note image has shape (1, rows, cols)
     */
    [[nodiscard]] auto get(int idx) const -> DataType;

private:
    MNISTDataset(const std::tuple<Tensor, Tensor, int> &);
    Tensor images;
    Tensor labels;
    int N;
};

}    // namespace tinytensor::data

#endif    // TINYTENSOR_DATAST_MNIST_H_
