#ifndef FFTWPP_UTILITY_GUARD_H
#define FFTWPP_UTILITY_GUARD_H

#include <algorithm>
#include <complex>
#include <random>
#include <ranges>

#include "Core.h"

namespace FFTWpp {

// Returns the size of in and out arrays for transforms with
// given dimensions.
template <IsScalar InType, IsScalar OutType, typename... Dimensions>
requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
auto DataSize(Dimensions... dimensions) {
  auto dims = std::vector{{dimensions...}};
  auto size0 = std::ranges::fold_left_first(std::ranges::views::all(dims),
                                            std::multiplies<>())
                   .value();
  if constexpr (std::same_as<InType, OutType>) {
    return std::pair(size0, size0);
  } else {
    auto rank = dims.size();
    auto last = dims.back();
    auto size1 =
        std::ranges::fold_left_first(
            std::ranges::views::all(dims) | std::ranges::views::take(rank - 1),
            std::multiplies<>())
            .value_or(1) *
        (last / 2 + 1);
    if constexpr (IsReal<InType> && IsComplex<OutType>) {
      return std::pair(size0, size1);
    } else {
      return std::pair(size1, size0);
    }
  }
}

// Sets values within a range using a standard normal distribution.
template <std::ranges::range Range>
requires requires() {
  requires std::ranges::output_range<Range, std::ranges::range_value_t<Range>>;
  requires IsScalar<std::ranges::range_value_t<Range>>;
}
void RandomiseValues(Range& range) {
  using Scalar = std::ranges::range_value_t<Range>;
  using Real = RemoveComplex<Scalar>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Real> d{0., 1.};
  std::transform(range.begin(), range.end(), range.begin(), [&](auto) {
    if constexpr (IsReal<Scalar>) {
      return d(gen);
    } else {
      return Scalar{d(gen), d(gen)};
    }
  });
}

// Check the values of two ranges agree once the second is scaled by
// the given norm.
template <std::ranges::range Range,
          typename Scalar = std::ranges::range_value_t<Range>>
auto CheckValues(Range&& in, Range&& copy, Scalar norm) {
  using Real = FFTWpp::RemoveComplex<Scalar>;
  return std::ranges::all_of(
      std::ranges::views::zip_transform(
          [norm](auto x, auto y) { return std::abs(x - y * norm); },
          std::ranges::views::all(in), std::ranges::views::all(copy)),
      [](auto x) { return x < 1000 * std::numeric_limits<Real>::epsilon(); });
}

}  // namespace FFTWpp

#endif  // FFTWPP_UTILITY_GUARD_H