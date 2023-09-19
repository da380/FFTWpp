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

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//               Definition of the DataView class           //
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <ScalarIterator I>
class DataView {
 public:
  using value_type = std::iter_value_t<I>;
  using iterator = I;

  // Constructor
  DataView(I start, I finish, int rank, std::vector<int> n, int howMany,
           std::vector<int> embed, int stride, int dist)
      : start{start},
        finish{finish},
        rank{rank},
        n{std::make_shared<std::vector<int>>(n)},
        howMany{howMany},
        embed{std::make_shared<std::vector<int>>(embed)},
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

  // Return views of the storage arrays.
  auto NView() const { return std::views::all(*n); }
  auto EmbedView() const { return std::views::all(*n); }

  // Functions returnig storage information in suitable form.
  auto Rank() const { return rank; }
  auto N() { return &*n->begin(); }
  auto HowMany() const { return howMany; }
  auto Embed() { return &*embed->begin(); }
  auto Stride() const { return stride; }
  auto Dist() const { return dist; }

  // Check whether another data view has equal storage parameters.
  template <ScalarIterator J>
  bool EqualStorage(DataView<J> other) requires IteratorPair<I, J> {
    if (rank != other.Rank()) return false;
    if (howMany != other.HowMany()) return false;
    if (!std::ranges::equal(this->NView(), other.NView())) return false;
    if (!std::ranges::equal(this->EmbedView(), other.EmbedView())) return false;
    return true;
  }

  // Check whether another data view is suitable for transformation into.
  template <ScalarIterator J>
  bool Transformable(DataView<J> other) requires IteratorPair<I, J> {
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
  bool CheckConsistency() {
    // size of the data
    int dataSize = std::distance(start, finish);
    // total size of hte storage parameters
    int storageSize = howMany * std::reduce(embed->begin(), embed->end(), 1,
                                            std::multiplies<>());
    return dataSize == storageSize;
  }
};

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//           Wrappers for building DataViews in common cases           //
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// Wrappers to make DataView for 1D transformation
template <ScalarIterator I>
auto MakeDataView1D(I start, I finish) {
  auto dim = std::distance(start, finish);
  assert(dim > 0);
  std::vector<int> n(1, dim);
  return DataView(start, finish, 1, n, 1, n, 1, 1);
}

template <std::ranges::random_access_range R>
auto MakeDataView1D(R&& in) {
  return MakeDataView1D(std::begin(in), std::end(in));
}

// Wrappers to make DataView for many 1D transformations. Here it
// assumed that the ith datum within the kth transform is located
// at i + k*dim, with dim the dimension.
template <ScalarIterator I>
auto MakeDataView1DMany(I start, I finish, int howMany) {
  assert(howMany > 0);
  int dim = std::distance(start, finish);
  assert(dim % howMany == 0);
  dim /= howMany;
  int rank = 1;
  std::vector<int> n(1, dim);
  int stride = 1;
  int dist = dim;
  return DataView(start, finish, rank, n, howMany, n, stride, dist);
}

template <std::ranges::random_access_range R>
auto MakeDataView1DMany(R&& in, int howMany) {
  return MakeDataView1DMany(std::begin(in), std::end(in), howMany);
}

}  // namespace FFTWpp

#endif  // FFTWPP_VIEWS_GUARD_H
