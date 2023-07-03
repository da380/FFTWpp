#ifndef FFTWConcepts_GUARD_H
#define FFTWConcepts_GUARD_H

#include <complex>
#include <concepts>
#include <iterator>
#include <ranges>

namespace FFTW {

// Concepts for floating point types.
template <typename T>
concept IsReal = std::floating_point<T>;

template <typename Float>
concept IsSingle = std::same_as<Float, float>;

template <typename Float>
concept IsDouble = std::same_as<Float, double>;

template <typename Float>
concept IsLongDouble = std::same_as<Float, long double>;

// Concepts for complex numbers.
template <typename T>
struct IsComplexHelper : std::false_type {};

template <typename T>
struct IsComplexHelper<std::complex<T>> : std::true_type {};

template <typename T>
concept IsComplex = IsComplexHelper<T>::value;

template <typename T>
struct RemoveComplexHelper {
  using value_type = T;
};

template <typename T>
struct RemoveComplexHelper<std::complex<T>> {
  using value_type = T;
};

template <typename T>
using RemoveComplex = typename RemoveComplexHelper<T>::value_type;

template <typename T>
concept IsScalar = IsReal<T> or IsComplex<T>;

// Concepts for iterators.
template <typename I>
concept RandomAccessIterator = requires() {
  std::same_as<typename std::iterator_traits<I>::iterator_category,
               std::random_access_iterator_tag>;
};

template <RandomAccessIterator I>
using IteratorValue = typename std::iterator_traits<I>::value_type;

template <typename I>
concept RealIterator = requires() {
  requires RandomAccessIterator<I>;
  requires IsReal<IteratorValue<I>>;
};

template <typename I>
concept ComplexIterator = requires() {
  requires RandomAccessIterator<I>;
  requires IsComplex<IteratorValue<I>>;
  requires IsReal<RemoveComplex<IteratorValue<I>>>;
};

template <typename I>
concept ScalarIterator = RealIterator<I> or ComplexIterator<I>;

template <ScalarIterator I>
using IteratorPrecision = RemoveComplex<IteratorValue<I>>;

template <typename I>
concept IntegralIterator = requires() {
  requires RandomAccessIterator<I>;
  requires std::integral<IteratorValue<I>>;
};

// Concepts for iterator pairs.
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

}  // namespace FFTW

#endif  // FFTWConcepts_GUARD_H
