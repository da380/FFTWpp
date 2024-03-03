#ifndef FFTWPP_CONCEPTS_GUARD_H
#define FFTWPP_CONCEPTS_GUARD_H

#include <complex>
#include <concepts>
#include <iterator>
#include <ranges>

#include "Core.h"

namespace FFTWpp {

// Concepts for iterators
template <typename I>
concept RealIterator = requires() {
  requires std::random_access_iterator<I>;
  requires IsReal<std::iter_value_t<I>>;
};

template <typename I>
concept ComplexIterator = requires() {
  requires std::random_access_iterator<I>;
  requires IsComplex<std::iter_value_t<I>>;
  requires IsReal<RemoveComplex<std::iter_value_t<I>>>;
};

template <typename I>
concept ScalarIterator = RealIterator<I> or ComplexIterator<I>;

template <ScalarIterator I>
using IteratorPrecision = RemoveComplex<std::iter_value_t<I>>;

template <typename I>
concept IntegralIterator = requires() {
  requires std::random_access_iterator<I>;
  requires std::integral<std::iter_value_t<I>>;
};

// Concepts for iterator pairs.

template <typename I, typename O>
concept IteratorPair = requires() {
  requires ScalarIterator<I>;
  requires ScalarIterator<O>;
  requires std::same_as<IteratorPrecision<I>, IteratorPrecision<O>>;
};

template <typename I, typename O>
concept R2RIteratorPair = requires() {
  requires RealIterator<I>;
  requires RealIterator<O>;
  requires std::same_as<IteratorPrecision<I>, IteratorPrecision<O>>;
};

template <typename I, typename O>
concept C2CIteratorPair = requires() {
  requires ComplexIterator<I>;
  requires ComplexIterator<O>;
  requires std::same_as<IteratorPrecision<I>, IteratorPrecision<O>>;
};

template <typename I, typename O>
concept C2RIteratorPair = requires() {
  requires ComplexIterator<I>;
  requires RealIterator<O>;
  requires std::same_as<IteratorPrecision<I>, IteratorPrecision<O>>;
};

template <typename I, typename O>
concept R2CIteratorPair = requires() {
  requires RealIterator<I>;
  requires ComplexIterator<O>;
  requires std::same_as<IteratorPrecision<I>, IteratorPrecision<O>>;
};

// Concepts for ranges
template <typename R>
concept IntegralRange = requires() {
  requires std::ranges::random_access_range<R>;
  requires std::integral<std::ranges::range_value_t<R>>;
};

template <typename R>
concept RealRange = requires() {
  requires std::ranges::random_access_range<R>;
  requires IsReal<std::ranges::range_value_t<R>>;
};

template <typename R>
concept ComplexRange = requires() {
  requires std::ranges::random_access_range<R>;
  requires IsComplex<std::ranges::range_value_t<R>>;
  requires IsReal<RemoveComplex<std::ranges::range_value_t<R>>>;
};

template <typename R>
concept ScalarRange = RealRange<R> or ComplexRange<R>;

template <ScalarRange R>
using RangePrecision = RemoveComplex<std::ranges::range_value_t<R>>;

}  // namespace FFTWpp

#endif  // FFTWPP_CONCEPTS_GUARD_H
