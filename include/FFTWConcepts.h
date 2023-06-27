#ifndef FFTWConcepts_GUARD_H
#define FFTWConcepts_GUARD_H

#include <concepts>
#include <iterator>

namespace FFTW {

// Concepts for floating point types.
template <typename Float>
concept IsSingle = std::same_as<Float, float>;

template <typename Float>
concept IsDouble = std::same_as<Float, double>;

template <typename Float>
concept IsQuadruple = std::same_as<Float, long double>;

// Concepts for complex numbers.
template <typename T>
struct ComplexHelper : std::false_type {};

template <typename T>
struct ComplexHelper<std::complex<T>> : std::true_type {};

template <typename T>
concept Complex = requires() {
  requires ComplexHelper<T>::value;
  requires std::floating_point<typename T::value_type>;
};

// Concepts for real iterators
template <typename Iter>
concept RealIterator = requires() {
  requires std::contiguous_iterator<Iter>;
  requires std::floating_point<typename Iter::value_type>;
};

template <typename Iter, typename Float>
concept RealIteratorWithPrecision = requires() {
  requires RealIterator<Iter>;
  requires std::floating_point<Float>;
  requires std::same_as<typename Iter::value_type, Float>;
};

// Concepts for complex iterators
template <typename Iter>
concept ComplexIterator = requires() {
  requires std::contiguous_iterator<Iter>;
  requires Complex<typename Iter::value_type>;
};

template <typename Iter, typename Float>
concept ComplexIteratorWithPrecision = requires() {
  requires ComplexIterator<Iter>;
  requires std::floating_point<Float>;
  requires std::same_as<typename Iter::value_type::value_type, Float>;
};

}  // namespace FFTW

#endif  // FFTWConcepts_GUARD_H
