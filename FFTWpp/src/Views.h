#ifndef FFTWPP_VIEWS_GUARD_H
#define FFTWPP_VIEWS_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "Concepts.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTWpp {

template <ScalarIterator I>
class DataView {
 public:
  using value_type = std::iter_value_t<I>;
  using iterator = I;

  // Constructor.
  template <IntegralIterator IntIt>
  DataView(I start, I finish, int rank, IntIt nStart, IntIt nFinish,
           int howMany, IntIt embedStart, IntIt embedFinish, int stride,
           int dist)
      : start{start},
        finish{finish},
        rank{rank},
        n{std::make_shared<std::vector<int>>(nStart, nFinish)},
        howMany{howMany},
        embed{std::make_shared<std::vector<int>>(embedStart, embedFinish)},
        stride{stride},
        dist{dist} {
    assert(CheckConsistency());
  }

  // Return appropriate fftw3 pointer to the start of the data.
  auto Data() requires ComplexIterator<I> { return ComplexCast(&start[0]); }
  auto Data() requires RealIterator<I> { return &start[0]; }

  // Return iterators to the data
  auto begin() { return start; }
  auto end() { return finish; }

  // Functions returnig storage information in suitable form
  auto Rank() const { return rank; }
  auto N() { return &*n->begin(); }
  auto HowMany() const { return howMany; }
  auto Embed() { return &*embed->begin(); }
  auto Stride() const { return stride; }
  auto Dist() const { return dist; }

  // Return views of the storage arrays
  auto NView() const { return std::views::all(*n); }
  auto EmbedView() const { return std::views::all(*n); }

  // Check whether another data view is comparable.
  template <ScalarIterator J>
  bool Comparable(DataView<J> other) requires IteratorPair<I, J> {
    if (rank != other.Rank()) return false;
    if (howMany != other.HowMany()) return false;
    if constexpr (C2CIteratorPair<I, J> or R2RIteratorPair<I, J>) {
      return std::ranges::equal(this->NView(), other.NView());
    }
    if constexpr (C2RIteratorPair<I, J>) {
      return std::ranges::equal(
                 this->NView() | std::views::reverse | std::views::take(1),
                 other.NView() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x == y / 2 + 1; }) &&
             std::ranges::equal(
                 this->NView() | std::views::reverse | std::views::drop(1),
                 other.NView() | std::views::reverse | std::views::drop(1));
    }
    if constexpr (R2CIteratorPair<I, J>) {
      return std::ranges::equal(
                 this->NView() | std::views::reverse | std::views::take(1),
                 other.NView() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x / 2 + 1 == y; }) &&
             std::ranges::equal(
                 this->NView() | std::views::reverse | std::views::drop(1),
                 other.NView() | std::views::reverse | std::views::drop(1));
    }
    return true;
  }

  // Normalise the data as required after an inverse transformation.
  void normalise() {
    auto dim = std::reduce(this->NView().begin(), this->NView().end(), 1,
                           std::multiplies<>());
    auto norm = static_cast<value_type>(1) / static_cast<value_type>(dim);
    for (int i = 0; i < howMany; i++) {
      I it1 = std::next(start, i * dist);
      I it2 = std::next(it1, dim);
      std::transform(it1, it2, it1, [&norm](auto x) { return x * norm; });
    }
  }

 private:
  // Stored iterators to the data.
  I start;
  I finish;

  // Parameters describing data storage.
  int rank;
  std::shared_ptr<std::vector<int>> n;
  int howMany;
  std::shared_ptr<std::vector<int>> embed;

  int stride;
  int dist;

  // Checks consistence of stored data
  bool CheckConsistency() { return true; }
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

}  // namespace FFTWpp

#endif  // FFTWPP_VIEWS_GUARD_H
