#ifndef FFTWPP_CONCEPTS_GUARD_H
#define FFTWPP_CONCEPTS_GUARD_H

#include <complex>
#include <concepts>
#include <iterator>
#include <ranges>

#include "Concepts/All"

namespace FFTWpp {

// Concepts for iterator pairs.
template <typename I, typename O>
concept R2RIteratorPair = requires() {
  requires Concepts::RealIterator<I>;
  requires Concepts::RealIterator<O>;
  requires std::same_as<Concepts::IteratorPrecision<I>,
                        Concepts::IteratorPrecision<O>>;
};

template <typename I, typename O>
concept C2CIteratorPair = requires() {
  requires Concepts::ComplexIterator<I>;
  requires Concepts::ComplexIterator<O>;
  requires std::same_as<Concepts::IteratorPrecision<I>,
                        Concepts::IteratorPrecision<O>>;
};

template <typename I, typename O>
concept C2RIteratorPair = requires() {
  requires Concepts::ComplexIterator<I>;
  requires Concepts::RealIterator<O>;
  requires std::same_as<Concepts::IteratorPrecision<I>,
                        Concepts::IteratorPrecision<O>>;
};

template <typename I, typename O>
concept R2CIteratorPair = requires() {
  requires Concepts::RealIterator<I>;
  requires Concepts::ComplexIterator<O>;
  requires std::same_as<Concepts::IteratorPrecision<I>,
                        Concepts::IteratorPrecision<O>>;
};

}  // namespace FFTWpp

#endif  // FFTWPP_CONCEPTS_GUARD_H
