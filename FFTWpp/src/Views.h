/**
 * @file Views.h
 * @brief Provides C++20-style views and layout descriptors for FFTW data.
 *
 * This file defines two main classes: `Layout` and `View`. The `Layout` class
 * describes the N-dimensional shape and memory layout of data for an FFTW
 * transform, corresponding to the parameters of the advanced FFTW interface.
 * The `View` class combines a `Layout` with a C++20 range, providing a unified
 * object that can be used with the `FFTWpp::Ranges::Plan` class.
 */
#ifndef FFTWPP_VIEWS_GUARD_H
#define FFTWPP_VIEWS_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <functional>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "fftw3.h"

namespace FFTWpp {

namespace Ranges {

/**
 * @class Layout
 * @brief Describes the memory layout and dimensions of an FFTW transform.
 * @details This class holds all the parameters required by the advanced FFTW
 * planning routines, such as rank, dimensions, stride, and distance. It defines
 * the "shape" of the transform data.
 */
class Layout {
 public:
  /** @brief Default constructor. */
  Layout() = default;

  /**
   * @brief Constructor for simple, contiguous, multi-dimensional transforms.
   * @details Creates a layout for a single transform (`howMany`=1) with default
   * stride and distance.
   * @tparam Dimensions A parameter pack of integral types.
   * @param dimensions The size of the transform along each dimension.
   */
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and (std::integral<Dimensions> && ...)
  Layout(Dimensions... dimensions)
      : Layout(sizeof...(Dimensions), std::vector{dimensions...}, 1,
               std::vector{dimensions...}, 1, 0) {}

  /**
   * @brief General constructor for the advanced FFTW interface.
   * @tparam R1 A range type whose values are convertible to int.
   * @tparam R2 A range type whose values are convertible to int.
   * @param rank The number of dimensions of the transform.
   * @param n A range specifying the size of each dimension.
   * @param howMany The number of transforms to perform.
   * @param embed A range specifying the "embedded" size of the data arrays.
   * @param stride The distance between consecutive elements in a dimension.
   * @param dist The distance between the start of consecutive transform data
   * sets.
   * @throws std::invalid_argument if the arguments cannot describe a transform.
   * The checks are made unconditionally, not through `assert`, so that a
   * release build reports a bad layout instead of passing it to FFTW.
   */
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
        _dist{dist} {
    Validate();
  }

  // Access the layout information.
  /** @brief Gets the rank (number of dimensions) of the transform. */
  auto Rank() const { return _rank; }
  /** @brief Gets a view of the logical dimensions of the transform. */
  auto N() const { return std::views::all(_n); }
  /** @brief Gets the number of transforms to be computed. */
  auto HowMany() const { return _howMany; }
  /** @brief Gets a view of the embedded dimensions of the data array. */
  auto Embed() const { return std::views::all(_embed); }
  /** @brief Gets the stride between elements. */
  auto Stride() const { return _stride; }
  /** @brief Gets the distance between consecutive transforms. */
  auto Dist() const { return _dist; }

  // Return pointers to the storage vectors.
  /** @brief Gets a raw pointer to the logical dimensions vector (`n`). */
  auto NPointer() { return _n.data(); }
  /** @brief Gets a raw pointer to the embedded dimensions vector. */
  auto EmbedPointer() { return _embed.data(); }
  /** @brief Gets a read-only pointer to the logical dimensions vector (`n`). */
  auto NPointer() const { return _n.data(); }
  /** @brief Gets a read-only pointer to the embedded dimensions vector. */
  auto EmbedPointer() const { return _embed.data(); }

  /**
   * @brief Calculates the number of elements a single transform reads or
   * writes, ignoring `HowMany`.
   * @return The product of the embedded dimensions.
   */
  auto TransformSize() const {
    if (_embed.empty()) return 0;
    return std::accumulate(_embed.begin(), _embed.end(), 1,
                           std::multiplies<>());
  }

  /**
   * @brief Calculates the total storage size required for this layout.
   * @return The total number of elements in memory.
   */
  auto size() const { return HowMany() * TransformSize(); }

  /** @brief Defaulted equality operator. */
  bool operator==(const Layout&) const = default;

 private:
  /**
   * @brief Rejects argument combinations that cannot describe a transform.
   * @throws std::invalid_argument with a description of the first problem
   * found.
   */
  void Validate() const {
    const auto fail = [](const std::string& reason) {
      throw std::invalid_argument("FFTWpp::Ranges::Layout: " + reason);
    };
    if (_rank < 1) fail("the rank must be at least one");
    if (std::cmp_not_equal(_n.size(), _rank)) {
      fail("one dimension must be given for each rank");
    }
    if (std::cmp_not_equal(_embed.size(), _rank)) {
      fail("one embedded dimension must be given for each rank");
    }
    if (_howMany < 1) fail("at least one transform must be performed");
    if (_stride == 0) fail("the stride must be non-zero");
    if (std::ranges::any_of(_n, [](auto n) { return n < 1; })) {
      fail("every dimension must be positive");
    }
    if (!std::ranges::equal(_n, _embed, std::less_equal<>())) {
      fail(
          "every embedded dimension must be at least as large as the "
          "corresponding dimension");
    }
  }

  /// @brief Rank of the transformations (i.e., 1D, 2D, etc).
  int _rank = 0;
  /// @brief Vector of logical dimensions along each rank.
  std::vector<int> _n;
  /// @brief Number of transforms to be performed.
  int _howMany = 0;
  /// @brief Embedded size of the array along each rank.
  std::vector<int> _embed;
  /// @brief Offset between elements of the data.
  int _stride = 0;
  /// @brief Offset between the start of each transformation.
  int _dist = 0;
};

/**
 * @class View
 * @brief A C++20 ranges view combined with an FFTW data layout.
 * @details This class adapts a C++20 view (or range) to be used with the
 * `FFTWpp::Ranges::Plan`. It inherits from `Layout` to provide the necessary
 * dimensional information to FFTW's planning routines, and from
 * `std::ranges::view_interface` to provide standard range-based access.
 * @tparam _View The underlying C++20 view type. Must satisfy
 * `RealOrComplexWritableView`.
 */
template <NumericConcepts::RealOrComplexWritableView _View>
class View : public std::ranges::view_interface<View<_View>>, public Layout {
 public:
  /**
   * @brief Returns the number of elements in the underlying data view.
   * @details This resolves the ambiguity between `Layout::size`, which is the
   * storage the layout requires, and `view_interface::size`, which is the
   * storage actually present. The two are equal for every constructed `View`.
   */
  using std::ranges::view_interface<View<_View>>::size;

  /**
   * @brief Constructs a View from an existing view and a Layout object.
   * @param view The underlying data view.
   * @param layout The layout describing the transform shape.
   * @throws std::invalid_argument if the view does not hold exactly the number
   * of elements the layout requires.
   */
  View(_View view, Layout layout) : Layout(layout), _view{view} { CheckSize(); }

  /**
   * @brief Constructs a 1D View from an existing view.
   * @details The layout is inferred as a simple 1D transform of the view's
   * size.
   * @param view The underlying data view.
   */
  View(_View view) : View(view, Layout(view.size())) {}

  /**
   * @brief Constructs a multi-dimensional View from a view and dimensions.
   * @details Creates the `Layout` internally from the provided dimensions.
   * @tparam Dimensions A parameter pack of integral types for the dimensions.
   * @param view The underlying data view.
   * @param dimensions The size of the transform along each dimension.
   */
  template <typename... Dimensions>
  requires(sizeof...(Dimensions) > 0) and
          (std::convertible_to<Dimensions, int> && ...)
  View(_View view, Dimensions... dimensions)
      : View(view, Layout(dimensions...)) {}

  // Methods to inherit from view_interface.
  /** @brief Returns an iterator to the beginning of the view. */
  auto begin() { return _view.begin(); }
  /** @brief Returns an iterator to the end of the view. */
  auto end() { return _view.end(); }

  /**
   * @brief Returns a raw pointer to the beginning of the underlying data.
   * @details This is required for interfacing with the FFTW C API.
   * @return A pointer to the first element.
   */
  auto DataPointer() { return _view.data(); }

  /**
   * @brief Returns FFTW's alignment class for the start of the data.
   * @details Two views may be substituted for one another in a call to
   * `Plan::Execute(in, out)` only if they share an alignment class.
   * @return The opaque alignment class of the first element.
   * @see FFTWpp::AlignmentOf
   */
  auto Alignment() { return FFTWpp::AlignmentOf(_view.data()); }

  /** @brief Returns the `Layout` describing this view's shape. */
  const Layout& GetLayout() const { return *this; }

 private:
  /// @brief The stored C++20 view to the data.
  _View _view;

  /**
   * @brief Checks that the underlying view size matches the required storage
   * size of the Layout.
   * @throws std::invalid_argument if the sizes differ.
   */
  void CheckSize() const {
    if (std::cmp_not_equal(_view.size(), Layout::size())) {
      throw std::invalid_argument("FFTWpp::Ranges::View: the data holds " +
                                  std::to_string(_view.size()) +
                                  " elements but its layout requires " +
                                  std::to_string(Layout::size()));
    }
  }
};

/**
 * @brief Deduction guide to allow constructing a `View` from a range.
 * @details This allows a user to pass a container like `std::vector` directly
 * when constructing a `View`, automatically converting it to a view type.
 */
template <std::ranges::range R, typename... Args>
View(R&&, Args...) -> View<std::ranges::views::all_t<R>>;

}  // namespace Ranges

}  // namespace FFTWpp

#endif  // FFTWPP_VIEWS_GUARD_H
