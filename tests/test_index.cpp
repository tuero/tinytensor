// test_index.cpp
// Test indexing related methods

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/index.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cmath>
#include <tuple>
#include <vector>

using namespace tinytensor;

namespace {
constexpr bool isPrime(int number) {
    if (number < 2) return false;
    if (number == 2) return true;
    if (number % 2 == 0) return false;
    for (int i = 3; (i * i) <= number; i += 2) {
        if (number % i == 0) return false;
    }
    return true;
}
}    // namespace

// NOLINTNEXTLINE
TEST_CASE("index") {
    auto test_value = []<typename T>(Device device) {
        using U = to_ctype_t<kDefaultInt>;
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _values;
        std::vector<T> _expected;
        std::vector<U> _indices;
        std::vector<bool> _mask;

        _values.reserve(N);
        _mask.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _values.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            _mask.push_back(flag);
            if (flag) {
                _expected.push_back(static_cast<T>(v));
                _indices.push_back(static_cast<U>(i));
            }
        }

        Tensor values = Tensor(_values, {N}, device);
        Tensor expected = Tensor(_expected, {static_cast<int>(_expected.size())}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        Tensor mask = Tensor(_mask, {N}, device);

        CHECK(allclose(index(values, mask), expected));
        CHECK(allclose(index(values, indices), expected));
    };
    runner_all_except_bool(test_value);
}

// NOLINTNEXTLINE
TEST_CASE("index throws dtype") {
    auto test_value = []<typename T>(Device device) {
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _values;
        std::vector<float> _indices;

        _values.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _values.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            if (flag) {
                _indices.push_back(static_cast<float>(i));
            }
        }

        Tensor values = Tensor(_values, {N}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        CHECK_THROWS_AS(std::ignore = index(values, indices), TTException);
    };
    runner_all_except_bool(test_value);
}

// NOLINTNEXTLINE
TEST_CASE("index put") {
    auto test_value_value = []<typename T>(Device device) {
        using U = to_ctype_t<kDefaultInt>;
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _values;
        std::vector<T> _expected;
        std::vector<U> _indices;
        std::vector<bool> _mask;
        T put_value = 0;

        _values.reserve(N);
        _mask.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _values.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            _mask.push_back(flag);
            _expected.push_back(flag ? put_value : static_cast<T>(v));
            if (flag) {
                _indices.push_back(static_cast<U>(i));
            }
        }

        Tensor values = Tensor(_values, {N}, device);
        Tensor expected = Tensor(_expected, {static_cast<int>(_expected.size())}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        Tensor mask = Tensor(_mask, {N}, device);

        CHECK(allclose(index_put(values, indices, put_value), expected));
    };
    runner_all_except_bool(test_value_value);

    auto test_value_mask = []<typename T>(Device device) {
        using U = to_ctype_t<kDefaultInt>;
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _values;
        std::vector<T> _expected;
        std::vector<U> _indices;
        std::vector<bool> _mask;
        T put_value = 0;

        _values.reserve(N);
        _mask.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _values.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            _mask.push_back(flag);
            _expected.push_back(flag ? put_value : static_cast<T>(v));
            if (flag) {
                _indices.push_back(static_cast<U>(i));
            }
        }

        Tensor values = Tensor(_values, {N}, device);
        Tensor expected = Tensor(_expected, {static_cast<int>(_expected.size())}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        Tensor mask = Tensor(_mask, {N}, device);

        CHECK(allclose(index_put(values, mask, put_value), expected));
    };
    runner_all_except_bool(test_value_mask);

    auto test_tensor_value = []<typename T>(Device device) {
        using U = to_ctype_t<kDefaultInt>;
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _input;
        std::vector<T> _values;
        std::vector<T> _expected;
        std::vector<U> _indices;
        std::vector<bool> _mask;

        _input.reserve(N);
        _mask.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _input.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            _mask.push_back(flag);
            T _v = static_cast<T>(100 - v);
            _expected.push_back(flag ? _v : static_cast<T>(v));
            if (flag) {
                _values.push_back(_v);
                _indices.push_back(static_cast<U>(i));
            }
        }

        Tensor input = Tensor(_input, {N}, device);
        Tensor values = Tensor(_values, {static_cast<int>(_values.size())}, device);
        Tensor expected = Tensor(_expected, {static_cast<int>(_expected.size())}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        Tensor mask = Tensor(_mask, {N}, device);

        CHECK(allclose(index_put(input, indices, values), expected));
    };
    runner_all_except_bool(test_tensor_value);

    auto test_tensor_mask = []<typename T>(Device device) {
        using U = to_ctype_t<kDefaultInt>;
        const int N = 2 * 512 * 512 + 17;
        std::vector<T> _input;
        std::vector<T> _values;
        std::vector<T> _expected;
        std::vector<U> _indices;
        std::vector<bool> _mask;

        _input.reserve(N);
        _mask.reserve(N);
        for (int i = 0; i < N; ++i) {
            int v = i % 100;
            _input.push_back(static_cast<T>(v));
            bool flag = isPrime(v);
            _mask.push_back(flag);
            T _v = static_cast<T>(100 - v);
            _expected.push_back(flag ? _v : static_cast<T>(v));
            if (flag) {
                _values.push_back(_v);
                _indices.push_back(static_cast<U>(i));
            }
        }

        Tensor input = Tensor(_input, {N}, device);
        Tensor values = Tensor(_values, {static_cast<int>(_values.size())}, device);
        Tensor expected = Tensor(_expected, {static_cast<int>(_expected.size())}, device);
        Tensor indices = Tensor(_indices, {static_cast<int>(_indices.size())}, device);
        Tensor mask = Tensor(_mask, {N}, device);

        CHECK(allclose(index_put(input, mask, values), expected));
    };
    runner_all_except_bool(test_tensor_mask);
}

// NOLINTNEXTLINE
TEST_CASE("index_select") {
    auto test1 = []<typename T>(Device device) {
        int dim = 0;
        std::vector<T> _expected = {0, 12, 20, 20, 2, 14, 3, 15, 8, 20, 20, 20, 10, 22, 11, 23};

        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = a.permute({1, 2, 0});
        b[{indexing::Slice(), 1}] = 20;
        std::vector<int> indices = {0, 2};
        Tensor expected = Tensor(_expected, device).reshape({2, 4, 2});

        CHECK(allclose(index_select(b, Tensor(indices, device), dim), expected));
        CHECK(allclose(index_select(b, indices, dim), expected));
    };
    runner_single_type<float>(test1);

    auto test2 = []<typename T>(Device device) {
        int dim = 1;
        std::vector<T> _expected = {0, 12, 2, 14, 4, 16, 6, 18, 8, 20, 10, 22};

        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = a.permute({1, 2, 0});
        b[{indexing::Slice(), 1}] = 20;
        std::vector<int> indices = {0, 2};
        Tensor expected = Tensor(_expected, device).reshape({3, 2, 2});

        CHECK(allclose(index_select(b, Tensor(indices, device), dim), expected));
        CHECK(allclose(index_select(b, indices, dim), expected));
    };
    runner_single_type<float>(test2);

    auto test3 = []<typename T>(Device device) {
        int dim = 2;
        std::vector<T> _expected = {0, 12, 0, 20, 20, 20, 2, 14, 2, 3,  15, 3,  4,  16, 4,  20, 20, 20,
                                    6, 18, 6, 7,  19, 7,  8, 20, 8, 20, 20, 20, 10, 22, 10, 11, 23, 11};

        Tensor a = arange({2, 3, 4}, to_scalar<T>::type, device);
        Tensor b = a.permute({1, 2, 0});
        b[{indexing::Slice(), 1}] = 20;
        std::vector<int> indices = {0, 1, 0};
        Tensor expected = Tensor(_expected, device).reshape({3, 4, 3});

        CHECK(allclose(index_select(b, Tensor(indices, device), dim), expected));
        CHECK(allclose(index_select(b, indices, dim), expected));
    };
    runner_single_type<float>(test3);
}

// NOLINTNEXTLINE
TEST_CASE("repeat_interleave") {
    auto test1 = []<typename T>(Device device) {
        int dim = 0;
        int num_repeats = 2;
        std::vector<T> _expected = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        Tensor expected = Tensor(_expected, device).reshape({6, 2});
        CHECK(allclose(repeat_interleave(a, num_repeats, dim), expected));
    };
    runner_all_except_bool(test1);

    auto test2 = []<typename T>(Device device) {
        int dim = 1;
        int num_repeats = 2;
        std::vector<T> _expected = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        Tensor expected = Tensor(_expected, device).reshape({3, 4});
        CHECK(allclose(repeat_interleave(a, num_repeats, dim), expected));
    };
    runner_all_except_bool(test2);

    auto test_error_dim = []<typename T>(Device device) {
        int dim = 2;
        int num_repeats = 2;
        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = repeat_interleave(a, num_repeats, dim), TTException);
    };
    runner_all_except_bool(test_error_dim);

    auto test_error_repeat = []<typename T>(Device device) {
        int dim = 0;
        int num_repeats = -2;
        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        CHECK_THROWS_AS(std::ignore = repeat_interleave(a, num_repeats, dim), TTException);
    };
    runner_all_except_bool(test_error_repeat);
}

// NOLINTNEXTLINE
TEST_CASE("repeat") {
    auto test1 = []<typename T>(Device device) {
        std::vector<T> _expected = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5};
        std::vector<int> repeats = {2, 2};

        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        Tensor expected = Tensor(_expected, device).reshape({6, 4});

        CHECK(allclose(repeat(a, repeats), expected));
        CHECK(allclose(repeat(a, Tensor(repeats, device)), expected));
    };
    runner_all_except_bool(test1);

    auto test2 = []<typename T>(Device device) {
        std::vector<T> _expected = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
        std::vector<int> repeats = {2, 1};

        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        Tensor expected = Tensor(_expected, device).reshape({6, 2});

        CHECK(allclose(repeat(a, repeats), expected));
        CHECK(allclose(repeat(a, Tensor(repeats, device)), expected));
    };
    runner_all_except_bool(test2);

    auto test_error_type = []<typename T>(Device device) {
        std::vector<T> _expected = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
        std::vector<float> repeats = {2, 1};

        Tensor a = arange({3, 2}, to_scalar<T>::type, device);
        Tensor expected = Tensor(_expected, device).reshape({6, 2});

        CHECK_THROWS_AS(std::ignore = repeat(a, Tensor(repeats, device)), TTException);
    };
    runner_all_except_bool(test_error_type);
}
