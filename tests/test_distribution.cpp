// test_distribution.cpp
// Test the distribution functions

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/random.h>
#include <tt/scalar.h>
#include <tt/shape.h>
#include <tt/tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

using namespace tinytensor;
using MTSEED = std::mt19937::result_type;

namespace {

template <typename T>
auto ecdf_distance(const std::vector<T> &l1, const std::vector<T> &l2) -> double {
    auto sorted_l1 = l1;
    auto sorted_l2 = l2;
    std::sort(std::begin(sorted_l1), std::end(sorted_l1));
    std::sort(std::begin(sorted_l2), std::end(sorted_l2));
    double D = 0;
    double Dmin = 0;
    double Dmax = 0;
    const std::size_t n1 = sorted_l1.size();
    const std::size_t n2 = sorted_l2.size();
    auto it1 = std::begin(sorted_l1);
    auto it2 = std::begin(sorted_l2);
    while (it1 != std::end(sorted_l1) && it2 != std::end(sorted_l2)) {
        if (*it1 == *it2) {
            auto v = static_cast<double>(*it1);
            while (static_cast<double>(*it1) == v && it1 != std::end(sorted_l1)) {
                D += static_cast<double>(n2);
                ++it1;
            }
            while (static_cast<double>(*it2) == v && it2 != std::end(sorted_l2)) {
                D -= static_cast<double>(n1);
                ++it2;
            }
            Dmax = std::max(Dmax, D);
            Dmin = std::min(Dmin, D);
        } else if (*it1 < *it2) {
            D += static_cast<double>(n2);
            ++it1;
            Dmax = std::max(Dmax, D);
        } else {
            D -= static_cast<double>(n1);
            ++it2;
            Dmin = std::min(Dmin, D);
        }
    }
    Dmin = (Dmin > 0) ? Dmin : -Dmin;
    Dmax = (Dmax > 0) ? Dmax : -Dmax;
    return std::max(static_cast<double>(Dmin), static_cast<double>(Dmax)) / static_cast<double>(n1 * n2);
}

// constexpr double CONFIDENCE_LEVEL = 1.358;
constexpr double CONFIDENCE_LEVEL = 1.628;

template <typename T>
auto ks_test(const std::vector<T> &l1, const std::vector<T> &l2, double c_a = CONFIDENCE_LEVEL) -> bool {
    double D = ecdf_distance(l1, l2);
    auto n1 = static_cast<double>(l1.size());
    auto n2 = static_cast<double>(l2.size());
    const auto v = c_a * std::sqrt((n1 + n2) / (n1 * n2));
    // std::cout << D << " >? " << v << std::endl;
    return !(D > v);
}

}    // namespace

// NOLINTNEXTLINE
TEST_CASE("Distribution uniform real") {
    auto test_uniform_init = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double low = -3;
        double high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = uniform_real(low, high, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_real_distribution<R> dis(static_cast<R>(low), static_cast<R>(high));
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = dis(gen_mt);
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_uniform_init);
    runner_signed_integral(test_uniform_init);

    auto test_uniform_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;

        double low = -3;
        double high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor _low = full(low, {100, 100}, options);
        Tensor _high = full(high, {100, 100}, options);
        Tensor array = uniform_real(_low, _high, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_real_distribution<R> dis(static_cast<R>(low), static_cast<R>(high));
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = dis(gen_mt);
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_uniform_init_array);
    runner_signed_integral(test_uniform_init_array);

    auto test_uniform_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double low = -3;
        double high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = uniform_real(low, high, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_real_distribution<R> dis(-static_cast<R>(high), -static_cast<R>(low));
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = dis(gen_mt);
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_uniform_init_ood);
    runner_signed_integral(test_uniform_init_ood);

    auto test_uniform_inplace = []<typename T>(Device device) {
        double low = -3;
        double high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.uniform_real_(low, high, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_real_distribution<T> dis(static_cast<T>(low), static_cast<T>(high));
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = dis(gen_mt);
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_uniform_inplace);

    auto test_uniform_inplace_exception = []<typename T>(Device device) {
        double low = -3;
        double high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.uniform_real_(low, high, gen), TTException);
    };
    runner_signed_integral(test_uniform_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution uniform int") {
    auto test_uniform_init = []<typename T>(Device device) {
        int64_t low = -3;
        int64_t high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = uniform_int(low, high, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_int_distribution<int64_t> dis(low, high - 1);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_uniform_init);
    runner_signed_integral(test_uniform_init);

    auto test_uniform_init_array = []<typename T>(Device device) {
        int64_t low = -3;
        int64_t high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor _low = full(low, {100, 100}, options);
        Tensor _high = full(high, {100, 100}, options);
        Tensor array = uniform_int(_low, _high, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_int_distribution<int64_t> dis(low, high - 1);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_uniform_init_array);
    runner_signed_integral(test_uniform_init_array);

    auto test_uniform_init_ood = []<typename T>(Device device) {
        int64_t low = -3;
        int64_t high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = uniform_int(low, high, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_int_distribution<int64_t> dis(-high, -low + 1);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_uniform_init_ood);
    runner_signed_integral(test_uniform_init_ood);

    auto test_uniform_inplace = []<typename T>(Device device) {
        int64_t low = -3;
        int64_t high = 5;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.uniform_int_(low, high, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::uniform_int_distribution<int64_t> dis(low, high - 1);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_uniform_inplace);
    runner_signed_integral(test_uniform_inplace);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution bernoulli") {
    auto test_bernoulli_init = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = bernoulli(p, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::bernoulli_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_bernoulli_init);
    runner_integral(test_bernoulli_init);

    auto test_bernoulli_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _p = full(p, {100, 100}, options);
        Tensor array = bernoulli(_p, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::bernoulli_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_bernoulli_init_array);
    runner_integral(test_bernoulli_init_array);

    auto test_bernoulli_init_ood = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = bernoulli(p, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::bernoulli_distribution dis(1.0 - p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_bernoulli_init_ood);
    runner_integral(test_bernoulli_init_ood);

    auto test_bernoulli_inplace = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.bernoulli_(p, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::bernoulli_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_bernoulli_inplace);

    auto test_bernoulli_inplace_exception = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.bernoulli_(p, gen), TTException);
    };
    runner_integral(test_bernoulli_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution binomial") {
    auto test_binomial_init = []<typename T>(Device device) {
        double p = 0.3;
        int num_draws = 10;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = binomial(p, num_draws, {10, 10}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::binomial_distribution dis(num_draws, p);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_binomial_init);
    runner_integral(test_binomial_init);

    auto test_binomial_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double p = 0.3;
        int num_draws = 10;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _p = full(p, {10, 10}, options);
        Tensor _num_draws = full(num_draws, {10, 10}, options);
        Tensor array = binomial(_p, _num_draws, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::binomial_distribution dis(num_draws, p);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_binomial_init_array);
    runner_integral(test_binomial_init_array);

    auto test_binomial_init_ood = []<typename T>(Device device) {
        double p = 0.3;
        int num_draws = 10;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = binomial(p, num_draws, {10, 10}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::binomial_distribution dis(num_draws, 1.0 - p);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_binomial_init_ood);
    runner_integral(test_binomial_init_ood);

    auto test_binomial_inplace = []<typename T>(Device device) {
        double p = 0.3;
        int num_draws = 10;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({10, 10}, options);
        array.binomial_(p, num_draws, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::binomial_distribution dis(num_draws, p);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_binomial_inplace);

    auto test_binomial_inplace_exception = []<typename T>(Device device) {
        double p = 0.3;
        int num_draws = 10;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.binomial_(p, num_draws, gen), TTException);
    };
    runner_integral(test_binomial_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution geometric") {
    auto test_geometric_init = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = geometric(p, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::geometric_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt) + 1);
        }

        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_geometric_init);
    runner_integral(test_geometric_init);

    auto test_geometric_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _p = full(p, {100, 100}, options);
        Tensor array = geometric(_p, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::geometric_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt) + 1);
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_geometric_init_array);
    runner_integral(test_geometric_init_array);

    auto test_geometric_init_ood = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = geometric(p, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::geometric_distribution dis(1.0 - p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt) + 1);
        }
        CHECK_FALSE(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_geometric_init_ood);
    runner_integral(test_geometric_init_ood);

    auto test_geometric_inplace = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.geometric_(p, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::geometric_distribution dis(p);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt) + 1);
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_geometric_inplace);

    auto test_geometric_inplace_exception = []<typename T>(Device device) {
        double p = 0.3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.geometric_(p, gen), TTException);
    };
    runner_integral(test_geometric_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution poisson") {
    auto test_poisson_init = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = poisson(lambda, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::poisson_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_poisson_init);
    runner_integral(test_poisson_init);

    auto test_poisson_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _lambda = full(lambda, {100, 100}, options);
        Tensor array = poisson(_lambda, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::poisson_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_poisson_init_array);
    runner_integral(test_poisson_init_array);

    auto test_poisson_init_ood = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = poisson(lambda, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::poisson_distribution dis(lambda + 4);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_poisson_init_ood);
    runner_integral(test_poisson_init_ood);

    auto test_poisson_inplace = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.poisson_(lambda, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::poisson_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_poisson_inplace);

    auto test_poisson_inplace_exception = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.poisson_(lambda, gen), TTException);
    };
    runner_integral(test_poisson_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution exponential") {
    auto test_exponential_init = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = exponential(lambda, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::exponential_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_exponential_init);
    runner_integral(test_exponential_init);

    auto test_exponential_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _lambda = full(lambda, {100, 100}, options);
        Tensor array = exponential(_lambda, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::exponential_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_exponential_init_array);
    runner_integral(test_exponential_init_array);

    auto test_exponential_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = exponential(lambda, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::exponential_distribution dis(lambda + 6);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_exponential_init_ood);
    runner_integral(test_exponential_init_ood);

    auto test_exponential_inplace = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        array.exponential_(lambda, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::exponential_distribution dis(lambda);
        std::vector<T> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_exponential_inplace);

    auto test_exponential_inplace_exception = []<typename T>(Device device) {
        double lambda = 4;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.exponential_(lambda, gen), TTException);
    };
    runner_integral(test_exponential_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution normal") {
    auto test_normal_init = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor array = normal(mu, std, {10, 10}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::normal_distribution dis(mu, std);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_normal_init);
    runner_integral(test_normal_init);

    auto test_normal_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _mu = full(mu, {10, 10}, options);
        Tensor _std = full(std, {10, 10}, options);
        Tensor array = normal(_mu, _std, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::normal_distribution dis(mu, std);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_normal_init_array);
    runner_integral(test_normal_init_array);

    auto test_normal_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = normal(mu, std, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::normal_distribution dis(0.0, 1.0);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_normal_init_ood);
    runner_integral(test_normal_init_ood);

    auto test_normal_inplace = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({10, 10}, options);
        array.normal_(mu, std, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::normal_distribution dis(mu, std);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_normal_inplace);

    auto test_normal_inplace_exception = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.normal_(mu, std, gen), TTException);
    };
    runner_integral(test_normal_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution cauchy") {
    auto test_cauchy_init = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor array = cauchy(mu, std, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::cauchy_distribution dis(mu, std);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_cauchy_init);
    runner_integral(test_cauchy_init);

    auto test_cauchy_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _mu = full(mu, {100, 100}, options);
        Tensor _std = full(std, {100, 100}, options);
        Tensor array = cauchy(_mu, _std, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::cauchy_distribution dis(mu, std);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_cauchy_init_array);
    runner_integral(test_cauchy_init_array);

    auto test_cauchy_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = cauchy(mu, std, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::cauchy_distribution dis(0.0, 1.0);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_cauchy_init_ood);
    runner_integral(test_cauchy_init_ood);

    auto test_cauchy_inplace = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({10, 10}, options);
        array.cauchy_(mu, std, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::cauchy_distribution dis(mu, std);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_cauchy_inplace);

    auto test_cauchy_inplace_exception = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.cauchy_(mu, std, gen), TTException);
    };
    runner_integral(test_cauchy_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution lognormal") {
    auto test_lognormal_init = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor array = lognormal(mu, std, {10, 10}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::lognormal_distribution dis(mu, std);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_lognormal_init);
    runner_integral(test_lognormal_init);

    auto test_lognormal_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _mu = full(mu, {10, 10}, options);
        Tensor _std = full(std, {10, 10}, options);
        Tensor array = lognormal(_mu, _std, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::lognormal_distribution dis(mu, std);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_lognormal_init_array);
    runner_integral(test_lognormal_init_array);

    auto test_lognormal_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = lognormal(mu, std, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::lognormal_distribution dis(0.0, 1.0);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_lognormal_init_ood);
    runner_integral(test_lognormal_init_ood);

    auto test_lognormal_inplace = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({10, 10}, options);
        array.lognormal_(mu, std, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::lognormal_distribution dis(mu, std);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_lognormal_inplace);

    auto test_lognormal_inplace_exception = []<typename T>(Device device) {
        double mu = -2;
        double std = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.lognormal_(mu, std, gen), TTException);
    };
    runner_integral(test_lognormal_inplace_exception);
}

// NOLINTNEXTLINE
TEST_CASE("Distribution weibull") {
    auto test_weibull_init = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 2;
        double k = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor array = weibull(lambda, k, {10, 10}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::weibull_distribution dis(k, lambda);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }

        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_weibull_init);
    runner_integral(test_weibull_init);

    auto test_weibull_init_array = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 2;
        double k = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<R>::type).device(device);
        Tensor _lambda = full(lambda, {10, 10}, options);
        Tensor _k = full(k, {10, 10}, options);
        Tensor array = weibull(_lambda, _k, false, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::weibull_distribution dis(k, lambda);
        std::vector<R> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_weibull_init_array);
    runner_integral(test_weibull_init_array);

    auto test_weibull_init_ood = []<typename T>(Device device) {
        using R = std::conditional_t<IsScalarFloatType<T>, T, to_ctype_t<kDefaultFloat>>;
        double lambda = 2;
        double k = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = weibull(lambda, k, {100, 100}, options, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::weibull_distribution dis(1.0, 1.0);
        std::vector<R> expected_data(100 * 100, 0);
        for (auto &x : expected_data) {
            x = static_cast<R>(dis(gen_mt));
        }
        CHECK_FALSE(ks_test(array.to_vec<R>(), expected_data));
    };
    runner_floating_point(test_weibull_init_ood);
    runner_integral(test_weibull_init_ood);

    auto test_weibull_inplace = []<typename T>(Device device) {
        double lambda = 2;
        double k = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({10, 10}, options);
        array.weibull_(lambda, k, gen);

        std::mt19937 gen_mt(static_cast<MTSEED>(seed));
        std::weibull_distribution dis(k, lambda);
        std::vector<T> expected_data(10 * 10, 0);
        for (auto &x : expected_data) {
            x = static_cast<T>(dis(gen_mt));
        }
        CHECK(ks_test(array.to_vec<T>(), expected_data));
    };
    runner_floating_point(test_weibull_inplace);

    auto test_weibull_inplace_exception = []<typename T>(Device device) {
        double lambda = 2;
        double k = 3;
        uint64_t seed = 0;
        Generator gen(seed);
        const auto options = TensorOptions().dtype(to_scalar<T>::type).device(device);
        Tensor array = zeros({100, 100}, options);
        CHECK_THROWS_AS(array.weibull_(lambda, k, gen), TTException);
    };
    runner_integral(test_weibull_inplace_exception);
}
