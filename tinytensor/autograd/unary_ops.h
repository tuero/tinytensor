// unary_ops.h
// Unary operations with autograd support

#ifndef TINYTENSOR_AUTOGRAD_UNARY_OPS_H_
#define TINYTENSOR_AUTOGRAD_UNARY_OPS_H_

#include <tt/autograd.h>
#include <tt/device.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <string>

namespace tinytensor::autograd {

struct TensorClone : public TensorFunction<TensorClone> {
    static constexpr std::string name = "Clone";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorToScalarType : public TensorFunction<TensorToScalarType> {
    static constexpr std::string name = "ToScalarType";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, ScalarType dtype)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorToDevice : public TensorFunction<TensorToDevice> {
    static constexpr std::string name = "TensorToDevice";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, Device device) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIdentity : public TensorFunction<TensorIdentity> {
    static constexpr std::string name = "Identity";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorAbs : public TensorFunction<TensorAbs> {
    static constexpr std::string name = "Abs";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSign : public TensorFunction<TensorSign> {
    static constexpr std::string name = "Sign";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorNegate : public TensorFunction<TensorNegate> {
    static constexpr std::string name = "Negate";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLog : public TensorFunction<TensorLog> {
    static constexpr std::string name = "Log";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLog2 : public TensorFunction<TensorLog2> {
    static constexpr std::string name = "Log2";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLog10 : public TensorFunction<TensorLog10> {
    static constexpr std::string name = "Log10";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLog1p : public TensorFunction<TensorLog1p> {
    static constexpr std::string name = "Log1p";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorExp : public TensorFunction<TensorExp> {
    static constexpr std::string name = "Exp";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorExp2 : public TensorFunction<TensorExp2> {
    static constexpr std::string name = "Exp2";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorExpm1 : public TensorFunction<TensorExpm1> {
    static constexpr std::string name = "Expm1";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSqrt : public TensorFunction<TensorSqrt> {
    static constexpr std::string name = "Sqrt";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSin : public TensorFunction<TensorSin> {
    static constexpr std::string name = "Sin";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorCos : public TensorFunction<TensorCos> {
    static constexpr std::string name = "Cos";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorTan : public TensorFunction<TensorTan> {
    static constexpr std::string name = "Tan";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorASin : public TensorFunction<TensorASin> {
    static constexpr std::string name = "ASin";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorACos : public TensorFunction<TensorACos> {
    static constexpr std::string name = "ACos";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorATan : public TensorFunction<TensorATan> {
    static constexpr std::string name = "ATan";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSinh : public TensorFunction<TensorSinh> {
    static constexpr std::string name = "Sinh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorCosh : public TensorFunction<TensorCosh> {
    static constexpr std::string name = "Cosh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorTanh : public TensorFunction<TensorTanh> {
    static constexpr std::string name = "Tanh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorASinh : public TensorFunction<TensorASinh> {
    static constexpr std::string name = "ASinh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorACosh : public TensorFunction<TensorACosh> {
    static constexpr std::string name = "ACosh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorATanh : public TensorFunction<TensorATanh> {
    static constexpr std::string name = "ATanh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorErf : public TensorFunction<TensorErf> {
    static constexpr std::string name = "Erf";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorErfc : public TensorFunction<TensorErfc> {
    static constexpr std::string name = "Erfc";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorTGamma : public TensorFunction<TensorTGamma> {
    static constexpr std::string name = "TGamma";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLGamma : public TensorFunction<TensorLGamma> {
    static constexpr std::string name = "LGamma";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSigmoid : public TensorFunction<TensorSigmoid> {
    static constexpr std::string name = "Sigmoid";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLogSigmoid : public TensorFunction<TensorLogSigmoid> {
    static constexpr std::string name = "LogSigmoid";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorHardSigmoid : public TensorFunction<TensorHardSigmoid> {
    static constexpr std::string name = "HardSigmoid";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSoftplus : public TensorFunction<TensorSoftplus> {
    static constexpr std::string name = "Softplus";
    static auto
        forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double beta, double threshold)
            -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorRelu : public TensorFunction<TensorRelu> {
    static constexpr std::string name = "Relu";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorRelu6 : public TensorFunction<TensorRelu6> {
    static constexpr std::string name = "Relu6";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLeakyRelu : public TensorFunction<TensorLeakyRelu> {
    static constexpr std::string name = "LeakyRelu";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double negative_slope)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorElu : public TensorFunction<TensorElu> {
    static constexpr std::string name = "ELU";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double alpha) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSelu : public TensorFunction<TensorSelu> {
    static constexpr std::string name = "SELU";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSilu : public TensorFunction<TensorSilu> {
    static constexpr std::string name = "SiLU";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorHardtanh : public TensorFunction<TensorHardtanh> {
    static constexpr std::string name = "Hardtanh";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double min, double max)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSoftsign : public TensorFunction<TensorSoftsign> {
    static constexpr std::string name = "Softsign";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSoftmax : public TensorFunction<TensorSoftmax> {
    static constexpr std::string name = "Softmax";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorLogSoftmax : public TensorFunction<TensorLogSoftmax> {
    static constexpr std::string name = "LogSoftmax";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    // namespace tinytensor::autograd

#endif    // TINYTENSOR_AUTOGRAD_UNARY_OPS_H_
