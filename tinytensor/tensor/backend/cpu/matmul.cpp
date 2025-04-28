// matmul.cpp
// Matmul runner

#include "tensor/backend/cpu/matmul.h"

#include <tt/concepts.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "tensor/backend/common/span.h"
#include "tensor/backend/cpu/kernel/matmul.hpp"
#include "tensor/backend/cpu/storage_cpu.h"

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <variant>
#include <vector>

namespace tinytensor::cpu {

using namespace kernel::matmul;

auto batched_matmul_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    assert(lhs.dim() == 3 && rhs.dim() == 3);
    assert(lhs.size(2) == rhs.size(1));    // Inner dims match
    const int B = lhs.size(0);
    const int N = lhs.size(1);
    const int K = lhs.size(2);
    const int M = rhs.size(2);

    const auto res_shape = Shape({B, N, M});
    const auto res_device = lhs.device();

    // matmul requires contiguous data
    const auto lhs_cont = lhs.contiguous();
    const auto rhs_cont = rhs.contiguous();
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    // Storage type
            using T = template_parameter_t<DT>;                          // Underlying type

            // Allocate for result
            std::vector<T> result(static_cast<std::size_t>(B * N * M));

            // Data ptr to first element
            // Data can have an offset whilst still being contiguous
            const T *lhs_data_ptr = tensor_storage.data();
            const T *rhs_data_ptr = std::get<DT>(rhs_cont.template get_storage<StorageCPU>().storage).data();
            lhs_data_ptr += lhs_cont.offset();    // NOLINT(*-pointer-arithmetic)
            rhs_data_ptr += rhs_cont.offset();    // NOLINT(*-pointer-arithmetic)
            T *res_data_ptr = result.data();

            // Loop over batches
            for (int i = 0; i < B; ++i) {
                const HostSpan<const T> lhs_span{lhs_data_ptr, static_cast<std::size_t>(N * K)};
                const HostSpan<const T> rhs_span{rhs_data_ptr, static_cast<std::size_t>(K * M)};
                const HostSpan<T> res_span{res_data_ptr, static_cast<std::size_t>(N * M)};

                // Call kernel
                matmul_kernel(lhs_span, rhs_span, res_span, N, K, M);

                // Increment pointers to next batch
                lhs_data_ptr += (N * K);    // NOLINT(*-pointer-arithmetic)
                rhs_data_ptr += (K * M);    // NOLINT(*-pointer-arithmetic)
                res_data_ptr += (N * M);    // NOLINT(*-pointer-arithmetic)
            }

            return {std::make_unique<StorageCPU>(std::move(result)), lhs.dtype(), res_shape, res_device};
        },
        lhs_cont.template get_storage<StorageCPU>().storage
    );
}

}    // namespace tinytensor::cpu
