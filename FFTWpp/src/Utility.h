/**
 * @file Utility.h
 * @brief Provides miscellaneous utility functions for testing and data
 * handling.
 *
 * This file contains helper functions for common tasks such as calculating
 * the required sizes of input and output arrays, filling ranges with random
 * data for testing, and verifying the results of transforms.
 */
#ifndef FFTWPP_UTILITY_GUARD_H
#define FFTWPP_UTILITY_GUARD_H

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"

namespace FFTWpp {

/**
 * @brief Calculates the required input and output array sizes for a transform.
 * @details For complex-to-complex transforms, input and output sizes are
 * identical. For real-to-complex or complex-to-real, one size is for the full
 * real data and the other is for the half-complex data (where the last
 * dimension is size N/2 + 1).
 * @tparam InType The value type of the input data.
 * @tparam OutType The value type of the output data.
 * @tparam Dimensions A parameter pack of integral types for the dimensions.
 * @param dimensions The size of the transform along each dimension.
 * @return A `std::pair` where `first` is the required input array size and
 * `second` is the required output array size.
 */
template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType, typename... Dimensions>
requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
[[nodiscard]] constexpr auto DataSize(Dimensions... dimensions) {
  const auto dims =
      std::array<int, sizeof...(Dimensions)>{static_cast<int>(dimensions)...};
  const auto full =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  if constexpr (std::same_as<InType, OutType>) {
    return std::pair(full, full);
  } else {
    // One side is halfcomplex: its last dimension is n / 2 + 1.
    const auto halfcomplex =
        std::accumulate(dims.begin(), std::prev(dims.end()), 1,
                        std::multiplies<>()) *
        (dims.back() / 2 + 1);
    if constexpr (NumericConcepts::Real<InType> &&
                  NumericConcepts::Complex<OutType>) {
      return std::pair(full, halfcomplex);
    } else {
      return std::pair(halfcomplex, full);
    }
  }
}

/**
 * @brief Fills a range with random values from a standard normal distribution,
 * drawn from a caller-supplied generator.
 * @details Values are generated with a mean of 0.0 and a standard deviation
 * of 1.0. If the range contains complex numbers, both the real and imaginary
 * parts are filled with independent random values.
 * @tparam Range The type of the range to modify. Must be a writable range of
 * real or complex numbers.
 * @tparam Generator A uniform random bit generator.
 * @param range The range to fill with random values.
 * @param generator The source of randomness, advanced by this call.
 */
template <NumericConcepts::RealOrComplexWritableRange Range, typename Generator>
requires std::uniform_random_bit_generator<std::remove_reference_t<Generator>>
void RandomiseValues(Range&& range, Generator&& generator) {
  using Scalar = std::ranges::range_value_t<Range>;
  using Real = NumericConcepts::RemoveComplex<Scalar>;
  auto distribution = std::normal_distribution<Real>{0, 1};
  std::ranges::generate(range, [&]() {
    if constexpr (NumericConcepts::Real<Scalar>) {
      return distribution(generator);
    } else {
      return Scalar{distribution(generator), distribution(generator)};
    }
  });
}

/**
 * @brief Fills a range with random values from a standard normal distribution.
 * @details Equivalent to the generator-taking overload, using a thread-local
 * engine seeded once from `std::random_device`. The engine is thread-local, so
 * filling ranges concurrently is safe but the resulting sequence depends on
 * which thread ran.
 * @tparam Range The type of the range to modify. Must be a writable range of
 * real or complex numbers.
 * @param range The range to fill with random values.
 */
template <NumericConcepts::RealOrComplexWritableRange Range>
void RandomiseValues(Range&& range) {
  static thread_local auto generator = std::mt19937_64{std::random_device{}()};
  RandomiseValues(range, generator);
}

/**
 * @brief Fills a range with reproducible random values from a standard normal
 * distribution.
 * @details Two calls with the same seed and the same range type produce the
 * same values, which is what makes a failing test reproducible.
 * @tparam Range The type of the range to modify. Must be a writable range of
 * real or complex numbers.
 * @param range The range to fill with random values.
 * @param seed The seed for the generator.
 */
template <NumericConcepts::RealOrComplexWritableRange Range>
void RandomiseValues(Range&& range, std::uint64_t seed) {
  auto generator = std::mt19937_64{seed};
  RandomiseValues(range, generator);
}

/**
 * @brief Checks if two ranges are approximately equal after scaling one by a
 * norm.
 * @details This is useful for verifying the correctness of a
 * forward-and-backward transform. It computes `abs(in - copy * norm)` for each
 * element and checks if the result is within a tolerance defined by the machine
 * epsilon of the data type.
 * @tparam Range A readable range of real or complex numbers.
 * @tparam OtherRange A readable range of real or complex numbers.
 * @tparam Scalar A numeric type for the normalization factor.
 * @param in The first range (e.g., the original data).
 * @param copy The second range (e.g., the result of the inverse transform).
 * @param norm The normalization factor to apply to the second range.
 * @param tolerance The absolute tolerance. Defaults to 1000 machine epsilons
 * of `in`'s underlying real type, which is loose enough to absorb the rounding
 * of a forward and backward transform at any of the three precisions.
 * @return `true` if all corresponding values are approximately equal, `false`
 * otherwise.
 */
template <NumericConcepts::RealOrComplexRange Range,
          NumericConcepts::RealOrComplexRange OtherRange, typename Scalar>
requires requires() {
  requires std::convertible_to<Scalar, std::ranges::range_value_t<Range>>;
}
[[nodiscard]] auto CheckValues(
    Range&& in, OtherRange&& copy, Scalar norm,
    NumericConcepts::RemoveComplex<std::ranges::range_value_t<Range>>
        tolerance = 1000 * std::numeric_limits<NumericConcepts::RemoveComplex<
                               std::ranges::range_value_t<Range>>>::epsilon()) {
  return std::ranges::all_of(
      std::ranges::views::zip_transform(
          [norm](auto x, auto y) { return std::abs(x - y * norm); },
          std::ranges::views::all(in), std::ranges::views::all(copy)),
      [tolerance](auto x) { return x < tolerance; });
}

}  // namespace FFTWpp

#endif  // FFTWPP_UTILITY_GUARD_H
