#ifndef FFTWPP_CONCEPTS_GUARD_H
#define FFTWPP_CONCEPTS_GUARD_H

#ifndef FFTWPP_MODULE_H
#error \
    "Please include FFTWpp.h instead of including headers inside the src directory directly."
#endif

#include <complex>
#include <concepts>
#include <iterator>
#include <ranges>

namespace FFTWpp {

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
concept Iterator = requires() {
  std::same_as<typename std::iterator_traits<I>::iterator_category,
               std::random_access_iterator_tag>;
};

template <Iterator I>
using IteratorValue = typename std::iterator_traits<I>::value_type;

template <typename I>
concept RealIterator = requires() {
  requires Iterator<I>;
  requires IsReal<IteratorValue<I>>;
};

template <typename I>
concept ComplexIterator = requires() {
  requires Iterator<I>;
  requires IsComplex<IteratorValue<I>>;
  requires IsReal<RemoveComplex<IteratorValue<I>>>;
};

template <typename I>
concept ScalarIterator = RealIterator<I> or ComplexIterator<I>;

template <ScalarIterator I>
using IteratorPrecision = RemoveComplex<IteratorValue<I>>;

template <typename I>
concept IntegralIterator = requires() {
  requires Iterator<I>;
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

}  // namespace FFTWpp

#endif  // FFTWPP_CONCEPTS_GUARD_H
