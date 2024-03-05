#ifndef MAKE_DATA_GUARD_H
#define MAKE_DATA_GUARD_H

#include <FFTWpp/Core>
#include <algorithm>
#include <complex>
#include <random>
#include <ranges>

/*
template <typename Iter>
requires FFTWpp::ComplexIterator<Iter>
void MakeComplexData(Iter first, Iter last) {
  using Float = FFTWpp::IteratorPrecision<Iter>;
  using Complex = std::complex<Float>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Float> d{0., 1.};
  std::transform(first, last, first, [&gen, &d](Complex) {
    return Complex{d(gen), d(gen)};
  });
}

template <typename Iter>
requires FFTWpp::RealIterator<Iter>
void MakeRealData(Iter first, Iter last) {
  using Float = FFTWpp::IteratorPrecision<Iter>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<typename Iter::value_type> d{0., 1.};
  std::transform(first, last, first, [&gen, &d](Float) { return d(gen); });
}

*/

template <FFTWpp::IsScalar Scalar>
auto MakeData(int size) {
  using Real = FFTWpp::RemoveComplex<Scalar>;
  auto data = FFTWpp::vector<Scalar>();
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Real> d{0., 1.};
  if constexpr (FFTWpp::IsReal<Scalar>) {
    std::generate_n(data.begin(), data.end(), std::back_inserter(data),
                    [&]() { return d(gen); });
  } else {
    std::generate_n(data.begin(), data.end(), std::back_inserter(data), [&]() {
      return Scalar{d(gen), d(gen)};
    });
  }
  return data;
}

#endif  // MAKE_DATA_GUARD_H
