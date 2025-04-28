// test_matmul.cpp
// Test matmul methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

using namespace tinytensor;

namespace {
template <typename T>
auto dot_product(std::vector<T> &v1, std::vector<T> &v2) -> std::vector<T> {
    T result{};
    assert(v1.size() == v2.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result += static_cast<T>(v1[i] * v2[i]);
    }
    return {result};
}

template <typename T>
void matmul(const T *A, const T *B, T *C, int N, int K, int M) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            // Inner accumulate
            for (int i = 0; i < K; ++i) {
                const int idx_a = row * K + i;
                const int idx_b = i * M + col;
                const int idx_c = row * M + col;
                C[idx_c] += static_cast<T>(A[idx_a] * B[idx_b]);    // NOLINT
            }
        }
    }
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Dot product") {
    // disabled for bool
    auto test_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {12}, device);
        Tensor rhs = full(true, {12}, device);
        CHECK_THROWS_AS(std::ignore = matmul(lhs, rhs), TTException);
    };
    runner_boolean(test_bool);

    auto test_small = []<typename T>(Device device) {
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int i = 0; i < 12; ++i) {
            data_lhs.push_back(static_cast<T>(i));
            data_rhs.push_back(static_cast<T>(i + 10));
        }
        Tensor lhs(data_lhs, {12}, device);
        Tensor rhs(data_rhs, {12}, device);
        Tensor expected(dot_product(data_lhs, data_rhs), {1}, device);
        CHECK(all(isclose(tinytensor::matmul(lhs, rhs), expected)));
    };
    runner_all_except_bool(test_small);

    auto test_large = []<typename T>(Device device) {
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int i = 0; i < 2048; ++i) {
            data_lhs.push_back(static_cast<T>(i));
            data_rhs.push_back(static_cast<T>(i + 10));
        }
        Tensor lhs(data_lhs, {2048}, device);
        Tensor rhs(data_rhs, {2048}, device);
        Tensor expected(dot_product(data_lhs, data_rhs), {1}, device);
        CHECK(allclose(tinytensor::matmul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_large);
}

// NOLINTNEXTLINE
TEST_CASE("Vector Matrix product") {
    // disabled for bool
    auto test_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {12}, device);
        Tensor rhs = full(true, {12, 4}, device);
        CHECK_THROWS_AS(std::ignore = matmul(lhs, rhs), TTException);
    };

    runner_boolean(test_bool);

    auto test_vector_matrix = []<typename T>(Device device) {
        constexpr int N = 256 + 123;
        constexpr int M = 512 + 123;
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int i = 0; i < N; ++i) {
            data_lhs.push_back(static_cast<T>(i % 100));
            for (int j = 0; j < M; ++j) {
                data_rhs.push_back(static_cast<T>((i + j) % 100));
            }
        }
        std::vector<T> data_result(M, 0);
        matmul(data_lhs.data(), data_rhs.data(), data_result.data(), 1, N, M);

        Tensor lhs(data_lhs, {N}, device);
        Tensor rhs(data_rhs, {N, M}, device);
        Tensor expected(data_result, {M}, device);
        CHECK(allclose(tinytensor::matmul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_vector_matrix);
}

// NOLINTNEXTLINE
TEST_CASE("Matrix Vector product") {
    // disabled for bool
    auto test_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {4, 12}, device);
        Tensor rhs = full(true, {12}, device);
        CHECK_THROWS_AS(std::ignore = matmul(lhs, rhs), TTException);
    };
    runner_boolean(test_bool);

    auto test_matrix_vector = []<typename T>(Device device) {
        constexpr int N = 256 + 123;
        constexpr int M = 512 + 123;
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                data_lhs.push_back(static_cast<T>((i + j) % 100));
            }
            data_rhs.push_back(static_cast<T>(i % 100));
        }
        std::vector<T> data_result(N, 0);
        matmul(data_lhs.data(), data_rhs.data(), data_result.data(), N, M, 1);

        Tensor lhs(data_lhs, {N, M}, device);
        Tensor rhs(data_rhs, {M}, device);
        Tensor expected(data_result, {N}, device);
        CHECK(allclose(tinytensor::matmul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_matrix_vector);
}

// NOLINTNEXTLINE
TEST_CASE("Matrix Matrix product") {
    // disabled for bool
    auto test_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {12, 4}, device);
        Tensor rhs = full(true, {4, 12}, device);
        CHECK_THROWS_AS(std::ignore = matmul(lhs, rhs), TTException);
    };
    runner_boolean(test_bool);

    auto test_matrix_matrix = []<typename T>(Device device) {
        constexpr int N = 256 + 123;
        constexpr int M = 1024 + 123;
        constexpr int K = 512 + 123;
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                data_lhs.push_back(static_cast<T>((i + j) % 100));
            }
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                data_rhs.push_back(static_cast<T>((i + j) % 100));
            }
        }
        std::vector<T> data_result(N * K, 0);
        matmul(data_lhs.data(), data_rhs.data(), data_result.data(), N, M, K);

        Tensor lhs(data_lhs, {N, M}, device);
        Tensor rhs(data_rhs, {M, K}, device);
        Tensor expected(data_result, {N, K}, device);
        CHECK(allclose(tinytensor::matmul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_matrix_matrix);
}

// NOLINTNEXTLINE
TEST_CASE("Batched Matrix Matrix product") {
    // disabled for bool
    auto test_bool = []<typename T>(Device device) {
        Tensor lhs = full(true, {2, 12, 4}, device);
        Tensor rhs = full(true, {2, 4, 12}, device);
        CHECK_THROWS_AS(std::ignore = matmul(lhs, rhs), TTException);
    };
    runner_boolean(test_bool);

    auto test_batched_matrix_matrix = []<typename T>(Device device) {
        constexpr int B = 4;
        constexpr int N = 256 + 123;
        constexpr int M = 1024 + 123;
        constexpr int K = 512 + 123;
        std::vector<T> data_lhs;
        std::vector<T> data_rhs;
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    data_lhs.push_back(static_cast<T>((i * b + j + b) % 100));
                }
            }
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < K; ++j) {
                    data_rhs.push_back(static_cast<T>((i + j + 2 * b) % 100));
                }
            }
        }
        std::vector<T> data_result(B * N * K, 0);
        for (int b = 0; b < B; ++b) {
            matmul(
                data_lhs.data() + (b * N * M),
                data_rhs.data() + (b * M * K),
                data_result.data() + (b * N * K),
                N,
                M,
                K
            );
        }

        Tensor lhs(data_lhs, {B, N, M}, device);
        Tensor rhs(data_rhs, {B, M, K}, device);
        Tensor expected(data_result, {B, N, K}, device);
        CHECK(allclose(tinytensor::matmul(lhs, rhs), expected));
    };
    runner_all_except_bool(test_batched_matrix_matrix);
}
