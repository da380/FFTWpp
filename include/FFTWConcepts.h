#ifndef FFTWConcepts_GUARD_H
#define FFTWConcepts_GUARD_H

#include <complex>
#include <concepts>
#include <iterator>

namespace FFTW {

// Some helper functions.
template <typename T>
concept HasValueType = requires() {
  typename T::value_type;
};

template <bool, typename T>
struct GetValueTypeHelper {
  using value_type = T::value_type;
};

template <typename T>
struct GetValueTypeHelper<false, T> {
  using value_type = T;
};

template <typename T>
using GetValueType = GetValueTypeHelper<HasValueType<T>, T>::value_type;

template <typename T>
using GetPrecision = GetValueType<GetValueType<T>>;

template <typename T>
concept HasPrecision = requires() {
  requires std::floating_point<GetPrecision<T>>;
};

template <typename S, typename T>
concept SamePrecision = requires() {
  requires HasPrecision<S>;
  requires HasPrecision<T>;
  requires std::same_as<GetPrecision<S>, GetPrecision<T>>;
};

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
concept IsComplex = requires() {
  requires IsComplexHelper<T>::value;
  requires std::floating_point<typename T::value_type>;
};

// Concepts for iterators.
template <typename I>
concept RandomAccessIterator = requires() {
  std::same_as<typename std::iterator_traits<I>::iterator_category,
               std::random_access_iterator_tag>;
};

template <typename I>
concept ScalarIterator = requires() {
  requires RandomAccessIterator<I>;
  requires std::floating_point<GetPrecision<I>>;
};

template <typename I>
concept RealIterator = requires() {
  requires ScalarIterator<I>;
  requires IsReal<typename std::iterator_traits<I>::value_type>;
};

template <typename I>
concept ComplexIterator = requires() {
  requires ScalarIterator<I>;
  requires IsComplex<typename std::iterator_traits<I>::value_type>;
};

// Concepts for iterator pairs.
template <typename I, typename O>
concept R2RIteratorPair = requires() {
  requires RealIterator<I>;
  requires RealIterator<O>;
  requires SamePrecision<I, O>;
};

template <typename I, typename O>
concept C2CIteratorPair = requires() {
  requires ComplexIterator<I>;
  requires ComplexIterator<O>;
  requires SamePrecision<I, O>;
};

template <typename I, typename O>
concept C2RIteratorPair = requires() {
  requires ComplexIterator<I>;
  requires RealIterator<O>;
  requires SamePrecision<I, O>;
};

template <typename I, typename O>
concept R2CIteratorPair = requires() {
  requires RealIterator<I>;
  requires ComplexIterator<O>;
  requires SamePrecision<I, O>;
};

}  // namespace FFTW

#endif  // FFTWConcepts_GUARD_H
