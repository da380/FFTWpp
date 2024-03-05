#ifndef FFTWPP_UTILITY_GUARD_H
#define FFTWPP_UTILITY_GUARD_H

#include <algorithm>
#include <complex>
#include <random>
#include <ranges>

#include "Core.h"

namespace FFTWpp {

template <IsScalar InType, IsScalar OutType, typename... Dimensions>
requires(sizeof...(Dimensions) > 0) and
        (std::convertible_to<Dimensions, int> && ...)
auto DataSize(Dimensions... dimensions) {
  auto dims = std::vector<int>{{dimensions...}};
  auto size0 = std::ranges::fold_left_first(std::ranges::views::all(dims),
                                            std::multiplies<>())
                   .value();
  if constexpr (std::same_as<InType, OutType>) {
    return std::pair(size0, size0);
  } else {
    auto rank = dims.size();
    auto last = dims[rank - 1];

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

template <IsScalar InType, IsScalar OutType, typename... Dimensions>
requires(sizeof...(Dimensions) > 0) and
        (std::convertible_to<Dimensions, int> && ...)
auto DataDimensions(Dimensions... dimensions) {
  auto dimensions0 = std::vector<int>{{dimensions...}};
  if constexpr (std::same_as<InType, OutType>) {
    return std::pair(dimensions0, dimensions0);
  } else {
    auto dimensions1 = dimensions0;
    dimensions1.back() = dimensions1.back() / 2 + 1;
    if constexpr (IsReal<InType> && IsComplex<OutType>) {
      return std::pair(dimensions0, dimensions1);
    } else {
      return std::pair(dimensions1, dimensions0);
    }
  }
}

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

}  // namespace FFTWpp

#endif  // FFTWPP_UTILITY_GUARD_H