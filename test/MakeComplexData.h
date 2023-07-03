#ifndef MakeComplexData_GUARD_H
#define MakeComplexData_GUARD_H

#include <algorithm>
#include <complex>
#include <iterator>
#include <random>
#include "FFTWConcepts.h"


template <typename Iter>
requires FFTW::ComplexIterator<Iter>
void MakeComplexData(Iter first, Iter last) {
  using Float = FFTW::IteratorPrecision<Iter>;
  using Complex = std::complex<Float>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<Float> d{0., 1.};
  std::transform(first,last,first, [&gen,&d](Complex) { return Complex{d(gen),d(gen)}; });
}

#endif // MakeComplexData_GUARD_H
