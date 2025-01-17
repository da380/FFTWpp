#ifndef FFTWPP_VIEWS_GUARD_H
#define FFTWPP_VIEWS_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <memory>
#include <ranges>
#include <vector>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "fftw3.h"

namespace FFTWpp {

namespace Ranges {

class Layout {
 public:
  Layout() = default;

  // Constructor for multi-dimensional transforms.
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
  Layout(Dimensions... dimensions)
      : Layout(sizeof...(Dimensions), std::vector{dimensions...}, 1,
               std::vector{dimensions...}, 1, 0) {}

  template <std::ranges::range R1, std::ranges::range R2>
  requires requires() {
    requires std::convertible_to<std::ranges::range_value_t<R1>, int>;
    requires std::convertible_to<std::ranges::range_value_t<R2>, int>;
  }
  Layout(int rank, R1&& n, int howMany, R2&& embed, int stride, int dist)
      : _rank{rank},
        _n{std::vector<int>(std::begin(n), std::end(n))},
        _howMany{howMany},
        _embed{std::vector<int>(std::begin(embed), std::end(embed))},
        _stride{stride},
        _dist{dist} {}

  // Access the layout information.
  auto Rank() const { return _rank; }
  auto N() const { return std::views::all(_n); }
  auto HowMany() const { return _howMany; }
  auto Embed() const { return std::views::all(_embed); }
  auto Stride() const { return _stride; }
  auto Dist() const { return _dist; }

  // Return pointers to the storage vectors.
  auto NPointer() { return _n.data(); }
  auto EmbedPointer() { return _embed.data(); }

  // Return the total storage size.
  auto size() const {
    return HowMany() *
           std::ranges::fold_left_first(Embed(), std::multiplies<>())
               .value_or(0);
  }

  bool operator==(const Layout&) const = default;

 private:
  int _rank;                // Rank of the transformations (i.e., 1D, 2D, etc).
  std::vector<int> _n;      // Vector of dimensions along each rank.
  int _howMany;             // Number of transforms to be performed.
  std::vector<int> _embed;  // Size along each rank.
  int _stride;              // Offset between elements of the data.
  int _dist;                // Offset between the start of each transformation.
};

template <NumericConcepts::RealOrComplexWritableView _View>
class View : public std::ranges::view_interface<View<_View>>, public Layout {
  using std::ranges::view_interface<View<_View>>::size;

 public:
  // Constructor given view and a layout.
  View(_View view, Layout layout) : Layout(layout), _view{view} {
    assert(CheckSize());
  }

  // Constructor given view which assumes a 1D transformation.
  View(_View view) : View(view, Layout(view.size())) {}

  // Constructor given view and multi-dimensional transform parameters.
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and
          (std::convertible_to<Dimensions, int> && ...)
  View(_View view, Dimensions... dimensions)
      : View(view, Layout(dimensions...)) {
    assert(CheckSize());
  }

  // Methods to inherit from view_interface.
  auto begin() { return _view.begin(); }
  auto end() { return _view.end(); }

  // Return appropriate fftw3 pointer to the start of the data.
  auto DataPointer() { return _view.data(); }

 private:
  // Store view to the data.
  _View _view;

  // Check the dimensions are consistent.
  auto CheckSize() const { return _view.size() == Layout::size(); }
};

// Deduction guide to allow range arguments.
template <std::ranges::range R, typename... Args>
View(R&&, Args...) -> View<std::ranges::views::all_t<R>>;

}  // namespace Ranges

}  // namespace FFTWpp

#endif  // FFTWPP_VIEWS_GUARD_H
