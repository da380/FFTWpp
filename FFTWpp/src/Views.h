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
//             Definition of the DataLayout class           //
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

class DataLayout {
 public:
  // Default constructor.
  DataLayout() = default;

  // General constructor.
  template <IntegralRange IntRange>
  DataLayout(int rank, IntRange n, int howMany, IntRange embed, int stride,
             int dist)
      : _rank{rank},
        _n{std::make_shared<std::vector<int>>(std::begin(n), std::end(n))},
        _howMany{howMany},
        _embed{std::make_shared<std::vector<int>>(std::begin(embed),
                                                  std::end(embed))},
        _stride{stride},
        _dist{dist} {}

  // Copy constructor.
  DataLayout(DataLayout const&) = default;

  // Move constructor.
  DataLayout(DataLayout&& other)
      : _rank{std::move(other._rank)},
        _n{std::move(other._n)},
        _howMany{std::move(other._howMany)},
        _embed{std::move(other._embed)},
        _stride{std::move(other._stride)},
        _dist{std::move(other._dist)} {}

  // Copy assigment.
  DataLayout& operator=(DataLayout const&) = default;

  // Move assigment.
  DataLayout& operator=(DataLayout&& other) {
    _rank = std::move(other._rank);
    _n = std::move(other._n);
    _howMany = std::move(other._howMany);
    _embed = std::move(other._embed);
    _stride = std::move(other._stride);
    _dist = std::move(other._dist);
    return *this;
  }

  // Return views of the storage arrays.
  auto NView() const {
  return std::views::all(*_n);
}
  auto EmbedView() const { return std::views::all(*_n); }

  // Functions returnig storage information in suitable form.
  auto Rank() const { return _rank; }
  auto N() { return _n->data(); }
  auto HowMany() const { return _howMany; }
  auto Embed() { return _embed->data(); }
  auto Stride() const { return _stride; }
  auto Dist() const { return _dist; }

  // Return the total storage size.
  std::size_t StorageSize() const {
    return HowMany() * std::reduce(EmbedView().begin(), EmbedView().end(), 1,
                                   std::multiplies<>());
  }

  // Check whether another data view has equal storage parameters.
  bool EqualStorage(DataLayout& other) {
    if (_rank != other.Rank()) return false;
    if (_howMany != other.HowMany()) return false;
    if (!std::ranges::equal(this->NView(), other.NView())) return false;
    if (!std::ranges::equal(this->EmbedView(), other.EmbedView())) return false;
    return true;
  }

  // Generate fake date of a given type.
  template <typename value_type>
  auto FakeData() {
    return std::make_shared<vector<value_type>>(StorageSize());
  }

 private:
  // Parameters describing data storage.
  int _rank;
  std::shared_ptr<std::vector<int>> _n;
  int _howMany;
  std::shared_ptr<std::vector<int>> _embed;
  int _stride;
  int _dist;
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//              Definition of the DataView class            //
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <ScalarIterator I>
class DataView {
 public:
  using value_type = std::iter_value_t<I>;
  using iterator = I;

  DataView() = default;

  // Constructor given iterators and storage parameters.
  template <IntegralRange IntRange>
  DataView(I start, I finish, int rank, IntRange n, int howMany, IntRange embed,
           int stride, int dist)
      : _start{start},
        _finish{finish},
        _layout{DataLayout(rank, n, howMany, embed, stride, dist)} {
    assert(CheckConsistency());
  }

  // Constructor given iterators and DataLayout instance.
  DataView(I start, I finish, DataLayout layout)
      : _start{start}, _finish{finish}, _layout{layout} {
    assert(CheckConsistency());
  }

  // Copy constructor.
  DataView(DataView const&) = default;

  // Move constructor.
  DataView(DataView&& other)
      : _start{std::move(other._start)},
        _finish{std::move(other._finish)},
        _layout{std::move(other._layout)} {}

  // Copy assigment.
  DataView& operator=(DataView const&) = default;

  // Move assigment.
  DataView& operator=(DataView&& other) {
    _start = std::move(other._start);
    _finish = std::move(other._finish);
    _layout = std::move(other._layout);
    return *this;
  }

  // Return appropriate fftw3 pointer to the start of the data.
  auto Data() requires ComplexIterator<I> { return ComplexCast(&_start[0]); }
  auto Data() requires RealIterator<I> { return &_start[0]; }

  // Return iterators to the data
  auto begin() { return _start; }
  auto end() { return _finish; }

  // Return views of the storage arrays.
  auto NView() const { return _layout.NView(); }
  auto EmbedView() const { return _layout.EmbedView(); }

  // Functions returning storage information in suitable form.
  auto Rank() const { return _layout.Rank(); }
  auto N() { return _layout.N(); }
  auto HowMany() const { return _layout.HowMany(); }
  auto Embed() { return _layout.Embed(); }
  auto Stride() const { return _layout.Stride(); }
  auto Dist() const { return _layout.Dist(); }

  // Check whether another data view has equal storage parameters.
  template <ScalarIterator J>
  bool EqualStorage(DataView<J>& other) requires IteratorPair<I, J> {
    return _layout.EqualStorage(other._layout);
  }

  // Return the total size of the data.
  size_t StorageSize() { return _layout.StorageSize(); }

  // Check whether another data view is suitable for transformation into.
  template <ScalarIterator J>
  bool Transformable(DataView<J>& other) requires IteratorPair<I, J> {
    if (this->Rank() != other.Rank()) return false;
    if (this->HowMany() != other.HowMany()) return false;
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

  // Return reference to the data layout.
  auto& Layout() { return _layout; }

  // Return a shared pointer to data of the correct size.
  auto FakeData() { return _layout.FakeData<value_type>(); }

 private:
  // Stored iterators to the data.
  I _start;
  I _finish;

  // Store the data layout.
  DataLayout _layout;

  // Checks consistence of stored data
  bool CheckConsistency() {
    // size of the data
    int dataSize = std::distance(_start, _finish);
    // total size of hte storage parameters
    int storageSize =
        HowMany() * std::reduce(EmbedView().begin(), EmbedView().end(), 1,
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
