#ifndef FFTWPP_DATA_GUARD_H
#define FFTWPP_DATA_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

#include "Concepts.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTWpp {

// Tag classes for different storage orders.
struct RowMajor {};
struct ColumnMajor {};

template <typename Storage>
concept StorageOption =
    std::same_as<Storage, RowMajor> or std::same_as<Storage, ColumnMajor>;

template <ScalarIterator I>
class DataView {
 public:
  using value_type = std::iter_value_t<I>;
  using iterator = I;

  // Constructor.
  template <IntegralIterator IntIt>
  DataView(I start, I finish, int rank, IntIt nStart, IntIt nFinish,
           int howmany, IntIt embedStart, IntIt embedFinish, int stride,
           int dist)
      : _start{start},
        _finish{finish},
        _rank{rank},
        _n{std::vector<int>(nStart, nFinish)},
        _howmany{howmany},
        _embed{std::vector<int>(embedStart, embedFinish)},
        _stride{stride},
        _dist{dist} {}

  // Return appropriate fftw3 pointer to the start of the data.
  auto data() requires ComplexIterator<I> { return ComplexCast(&_start[0]); }
  auto data() requires RealIterator<I> { return &_start[0]; }

  // Return iterators to the data
  auto begin() { return _start; }
  auto end() { return _finish; }

  // Return const iterators to storage arrays
  auto nBegin() const { return _n.cbegin(); }
  auto nEnd() const { return _n.cend(); }
  auto nRBegin() const { return _n.crbegin(); }
  auto nREnd() const { return _n.crend(); }
  auto embedBegin() const { return _embed.cbegin(); }
  auto embedEnd() const { return _embed.cend(); }

  // Functions returnig storage information in suitable form
  auto rank() const { return _rank; }
  auto n() { return &_n[0]; }
  auto howmany() const { return _howmany; }
  auto embed() { return &_embed[0]; }
  auto stride() const { return _stride; }
  auto dist() const { return _dist; }

  // Check whether another data reference is comparable.
  template <ScalarIterator J>
  bool Comparable(DataView<J> other) requires IteratorPair<I, J> {
    if (_rank != other.rank()) return false;
    if (_howmany != other.howmany()) return false;
    if constexpr (C2CIteratorPair<I, J> or R2RIteratorPair<I, J>) {
      // return std::equal(this->nBegin(), this->nEnd(), other.nBegin());
    }
    if constexpr (C2RIteratorPair<I, J>) {
      auto it1 = this->nRBegin();
      auto it2 = other.nRBegin();
      auto check = *it1++ == *it2++ / 2 + 1;
      return check && std::equal(it1, this->nREnd(), it2);
    }
    if constexpr (R2CIteratorPair<I, J>) {
      auto it1 = this->nRBegin();
      auto it2 = other.nRBegin();
      auto check = *it1++ / 2 + 1 == *it2++;
      return check && std::equal(it1, this->nREnd(), it2);
    }
    return true;
  }

  // Normalise the data as required after an inverse transformation.
  void normalise() {
    using Float = IteratorPrecision<I>;
    auto dim = std::reduce(_n.begin(), _n.end(), 1, std::multiplies<>());
    auto norm = static_cast<Float>(1) / static_cast<Float>(dim);
    for (int i = 0; i < _howmany; i++) {
      I start = std::next(_start, i * _dist);
      I finish = std::next(start, dim);
      std::transform(start, finish, start,
                     [&norm](auto x) { return x * norm; });
    }
  }

 private:
  // Stored iterators to the data.
  I _start;
  I _finish;

  // Parameters describing data storage.
  int _rank;
  std::vector<int> _n;
  int _howmany;
  std::vector<int> _embed;
  int _stride;
  int _dist;
};

template <ScalarIterator I>
auto MakeDataView1D(I start, I finish) {
  auto dim = std::distance(start, finish);
  assert(dim > 0);
  std::vector<int> n(1, dim);
  return DataView(start, finish, 1, n.begin(), n.end(), 1, n.begin(), n.end(),
                  1, 1);
}

template <std::ranges::random_access_range R>
auto MakeDataView1D(R&& in) {
  return MakeDataView1D(in.begin(), in.end());
}

template <ScalarIterator I, StorageOption Storage = ColumnMajor>
auto MakeDataView1DMany(I start, I finish, int howmany) {
  auto total = std::distance(start, finish);
  assert(total > 0);
  assert(howmany > 0);
  assert(total % howmany == 0);
  int dim = total / howmany;
  std::vector<int> n(1, dim);
  int stride = std::same_as<Storage, RowMajor> ? 1 : dim;
  int dist = std::same_as<Storage, RowMajor> ? dim : 1;
  return DataView(start, finish, 1, n.begin(), n.end(), howmany, n.begin(),
                  n.end(), stride, dist);
}

}  // namespace FFTWpp

#endif  // FFTWPP_DATA_GUARD_H
