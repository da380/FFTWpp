#ifndef MAKE_DATA_GUARD_H
#define MAKE_DATA_GUARD_H

#include <algorithm>
#include <complex>
#include <iterator>
#include <random>
#include <FFTWpp/Core>


template <typename Iter>
requires FFTWpp::ComplexIterator<Iter>
void MakeComplexData(Iter first, Iter last) {
  using Float = FFTWpp::IteratorPrecision<Iter>;
  using Complex = std::complex<Float>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Float> d{0., 1.};
  std::transform(first,last,first, [&gen,&d](Complex) { return Complex{d(gen),d(gen)}; });
}



template <typename Iter>
requires FFTWpp::RealIterator<Iter>
void MakeRealData(Iter first, Iter last) {
  using Float = FFTWpp::IteratorPrecision<Iter>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<typename Iter::value_type> d{0., 1.};
  std::transform(first,last,first, [&gen,&d](Float) { return d(gen) ; });
}


#endif // MAKE_DATA_GUARD_H
