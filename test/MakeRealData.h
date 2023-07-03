#ifndef MakeRealData_GUARD_H
#define MakeRealData_GUARD_H

#include <algorithm>
#include <random>
#include "FFTWConcepts.h"

template <typename Iter>
requires FFTW::RealIterator<Iter>
void MakeRealData(Iter first, Iter last) {
  using Float = FFTW::IteratorPrecision<Iter>;
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<typename Iter::value_type> d{0., 1.};
  std::transform(first,last,first, [&gen,&d](Float) { return d(gen) ; });
}


#endif // MakeRealData_GUARD_H
