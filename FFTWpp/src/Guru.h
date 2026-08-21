/**
 * @file Guru.h
 * @brief A range-based wrapper for FFTW's guru interface.
 *
 * The advanced interface, wrapped by `Ranges::Layout` and `Ranges::Plan`,
 * describes repeated transforms with a single `(howMany, dist)` pair. That
 * cannot express a transform along an interior axis of an array of rank three
 * or more, where the repetitions start at offsets that are not an arithmetic
 * progression. FFTW's guru interface can: every dimension carries its own
 * input and output stride, and the repetition loop is itself
 * multi-dimensional.
 *
 * This file provides `Ranges::GuruLayout`, which holds both lists of
 * dimensions and validates them, `Ranges::GuruPlan`, which owns the resulting
 * FFTW plan, and helpers that build a layout from a shape and a list of axes
 * so that no one has to derive strides by hand.
 */
#ifndef FFTWPP_GURU_GUARD_H
#define FFTWPP_GURU_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "fftw3.h"

namespace FFTWpp {

namespace Ranges {

/**
 * @brief `FFTWpp::Dim`, spelled in the namespace the rest of the guru
 * interface lives in.
 */
using Dim = FFTWpp::Dim;

/** @brief Which side of a transform a stride refers to. */
enum class Side { Input, Output };

/**
 * @brief Whether a side holds every element of a dimension or only the
 * halfcomplex half of it.
 * @details For a real-to-complex or complex-to-real transform, the complex
 * side holds `n / 2 + 1` elements along the last transform dimension rather
 * than `n`. Both the storage a side requires and whether its dimensions
 * overlap depend on which of the two it is.
 */
enum class Format { Full, Halfcomplex };

/**
 * @class GuruLayout
 * @brief The two lists of dimensions that describe a guru transform.
 * @details FFTW's guru interface takes two arrays of `fftw_iodim`: the
 * transform dimensions, and the dimensions of the loop over repeated
 * transforms. They have the same type and opposite meanings, so passing them
 * the wrong way round is a silent error that still plans. Holding both as
 * named members of one object removes that possibility.
 *
 * Most layouts are better built by `TransformAlong` and its siblings than
 * written out, since those compute the strides.
 *
 * @code
 * // A transform along axis 1 of a row-major (n0, n1, n2) array: the
 * // repetitions form a two-dimensional loop, over axes 0 and 2.
 * auto layout = Ranges::GuruLayout{
 *     {{.n = n1, .inStride = n2, .outStride = n2}},
 *     {{.n = n0, .inStride = n1 * n2, .outStride = n1 * n2},
 *      {.n = n2, .inStride = 1, .outStride = 1}}};
 * @endcode
 */
class GuruLayout {
 public:
  /** @brief Default constructor. Describes no transform. */
  GuruLayout() = default;

  /**
   * @brief Constructs a layout from its transform and repetition dimensions.
   * @param transform The dimensions being transformed. For a real-to-complex
   * or complex-to-real transform these are the extents of the *real* array.
   * @param batch The dimensions of the loop over repeated transforms. Empty
   * for a single transform.
   * @throws std::invalid_argument if a dimension cannot describe a transform:
   * an empty transform list, a non-positive extent, or a zero stride.
   */
  GuruLayout(std::vector<Dim> transform, std::vector<Dim> batch = {})
      : _transform{std::move(transform)}, _batch{std::move(batch)} {
    Validate();
  }

  /** @brief The dimensions being transformed. */
  const std::vector<Dim>& Transform() const { return _transform; }
  /** @brief The dimensions of the loop over repeated transforms. */
  const std::vector<Dim>& Batch() const { return _batch; }

  /** @brief The number of transform dimensions. */
  int Rank() const { return static_cast<int>(_transform.size()); }
  /** @brief The number of repetition dimensions. */
  int BatchRank() const { return static_cast<int>(_batch.size()); }

  /** @brief The number of transforms the layout describes. */
  std::ptrdiff_t HowMany() const {
    return std::accumulate(
        _batch.begin(), _batch.end(), std::ptrdiff_t{1},
        [](auto product, const auto& dim) { return product * dim.n; });
  }

  /**
   * @brief The logical size of one transform, used for normalisation.
   * @return The product of the transform extents.
   */
  std::ptrdiff_t TransformSize() const {
    return std::accumulate(
        _transform.begin(), _transform.end(), std::ptrdiff_t{1},
        [](auto product, const auto& dim) { return product * dim.n; });
  }

  /**
   * @brief The smallest number of elements a side's array must hold.
   * @details One past the largest offset the layout reaches, which is what a
   * buffer must provide for the transform to stay inside it. Strides may be
   * negative, in which case FFTW walks the axis backwards from the far end;
   * the extent is computed from their magnitudes, so it still bounds the
   * storage.
   * @param side Which array's strides to follow.
   * @param format Whether this side holds the full last transform dimension
   * or the halfcomplex `n / 2 + 1` of it.
   * @return The required number of elements.
   */
  std::ptrdiff_t Extent(Side side, Format format = Format::Full) const {
    auto extent = std::ptrdiff_t{1};
    for (std::size_t i = 0; i < _transform.size(); ++i) {
      auto n = _transform[i].n;
      if (format == Format::Halfcomplex && i + 1 == _transform.size()) {
        n = n / 2 + 1;
      }
      extent += (n - 1) * Magnitude(StrideOf(_transform[i], side));
    }
    for (const auto& dim : _batch) {
      extent += (dim.n - 1) * Magnitude(StrideOf(dim, side));
    }
    return extent;
  }

  /**
   * @brief Rejects a side whose dimensions would visit the same address twice.
   * @details FFTW requires the map from indices to offsets to be injective,
   * and does not check it; a layout that violates it silently computes the
   * wrong answer. The check here is the standard nesting condition: sorted by
   * stride magnitude, each dimension's stride must be at least the span of
   * every dimension inside it.
   *
   * Transform and repetition dimensions are checked together, since FFTW
   * iterates over both.
   * @param side Which array's strides to follow.
   * @param format Whether this side holds the full last transform dimension
   * or the halfcomplex half of it.
   * @throws std::invalid_argument naming the dimension that overlaps.
   */
  void ValidateNonOverlapping(Side side, Format format = Format::Full) const {
    // (stride magnitude, extent) for every dimension the transform walks.
    auto walked = std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>();
    walked.reserve(_transform.size() + _batch.size());
    for (std::size_t i = 0; i < _transform.size(); ++i) {
      auto n = _transform[i].n;
      if (format == Format::Halfcomplex && i + 1 == _transform.size()) {
        n = n / 2 + 1;
      }
      walked.emplace_back(Magnitude(StrideOf(_transform[i], side)), n);
    }
    for (const auto& dim : _batch) {
      walked.emplace_back(Magnitude(StrideOf(dim, side)), dim.n);
    }
    std::ranges::sort(walked);

    auto span = std::ptrdiff_t{1};
    for (const auto& [stride, n] : walked) {
      if (n > 1 && stride < span) {
        throw std::invalid_argument(
            "FFTWpp::Ranges::GuruLayout: the " +
            std::string(side == Side::Input ? "input" : "output") +
            " dimensions overlap -- a dimension of extent " +
            std::to_string(n) + " has stride " + std::to_string(stride) +
            ", but the dimensions inside it already span " +
            std::to_string(span) +
            " elements, so two indices would address the same element");
      }
      span += (n - 1) * stride;
    }
  }

  /** @brief Defaulted equality operator. */
  bool operator==(const GuruLayout&) const = default;

 private:
  std::vector<Dim> _transform;
  std::vector<Dim> _batch;

  static std::ptrdiff_t StrideOf(const Dim& dim, Side side) {
    return side == Side::Input ? dim.inStride : dim.outStride;
  }

  static std::ptrdiff_t Magnitude(std::ptrdiff_t stride) {
    return stride < 0 ? -stride : stride;
  }

  void Validate() const {
    const auto fail = [](const std::string& reason) {
      throw std::invalid_argument("FFTWpp::Ranges::GuruLayout: " + reason);
    };
    if (_transform.empty()) {
      fail("at least one transform dimension is required");
    }
    const auto check = [&fail](const std::vector<Dim>& dims,
                               const char* which) {
      for (const auto& dim : dims) {
        if (dim.n < 1) {
          fail(std::string("every ") + which +
               " extent must be positive, but one is " + std::to_string(dim.n));
        }
        if (dim.inStride == 0 || dim.outStride == 0) {
          fail(std::string("every ") + which +
               " stride must be non-zero; use an extent of one for a "
               "dimension that is not traversed");
        }
      }
    };
    check(_transform, "transform dimension");
    check(_batch, "repetition dimension");
  }
};

//----------------------------------------------------------//
//                    Building a layout                     //
//----------------------------------------------------------//

/**
 * @brief The strides of a contiguous row-major array of the given shape.
 * @details The last axis is contiguous, and each earlier stride is the product
 * of the extents that follow it. This is the layout of a C array, of a
 * `std::mdspan` with the default layout, and of the data `Ranges::Layout`
 * describes when no padding is given.
 * @param shape The extent along each axis.
 * @return One stride per axis.
 * @throws std::invalid_argument if the shape is empty or has a non-positive
 * extent.
 */
[[nodiscard]] inline std::vector<std::ptrdiff_t> RowMajorStrides(
    const std::vector<std::ptrdiff_t>& shape) {
  if (shape.empty()) {
    throw std::invalid_argument(
        "FFTWpp::Ranges::RowMajorStrides: the shape must have at least one "
        "axis");
  }
  if (std::ranges::any_of(shape, [](auto n) { return n < 1; })) {
    throw std::invalid_argument(
        "FFTWpp::Ranges::RowMajorStrides: every extent must be positive");
  }
  auto strides = std::vector<std::ptrdiff_t>(shape.size(), 1);
  for (auto axis = shape.size() - 1; axis > 0; --axis) {
    strides[axis - 1] = strides[axis] * shape[axis];
  }
  return strides;
}

}  // namespace Ranges

// These helpers live in FFTWpp::Internal rather than a nested
// Ranges::Internal: a second namespace of the same name inside Ranges would
// shadow this one, so an unqualified Internal:: elsewhere in Ranges would
// silently resolve to whichever was visible.
namespace Internal {

using Ranges::GuruLayout;

/** @brief Rejects an axis list that does not select axes of `shape`. */
inline std::vector<int> CheckedAxes(const std::vector<std::ptrdiff_t>& shape,
                                    const std::vector<int>& axes) {
  const auto fail = [](const std::string& reason) {
    throw std::invalid_argument("FFTWpp::Ranges::TransformAlong: " + reason);
  };
  if (axes.empty()) fail("at least one axis must be transformed");
  if (axes.size() > shape.size()) {
    fail("more axes were given than the shape has");
  }
  auto sorted = axes;
  std::ranges::sort(sorted);
  if (std::ranges::adjacent_find(sorted) != sorted.end()) {
    fail("an axis was given twice");
  }
  for (auto axis : sorted) {
    if (axis < 0 || std::cmp_greater_equal(axis, shape.size())) {
      fail("axis " + std::to_string(axis) + " is outside a shape of rank " +
           std::to_string(shape.size()));
    }
  }
  return sorted;
}

/** @brief Assembles a guru layout from per-axis strides on each side. */
inline GuruLayout Assemble(const std::vector<std::ptrdiff_t>& extents,
                           const std::vector<std::ptrdiff_t>& inStrides,
                           const std::vector<std::ptrdiff_t>& outStrides,
                           const std::vector<int>& axes) {
  auto transform = std::vector<Dim>();
  auto batch = std::vector<Dim>();
  for (std::size_t axis = 0; axis < extents.size(); ++axis) {
    const auto dim = Dim{.n = extents[axis],
                         .inStride = inStrides[axis],
                         .outStride = outStrides[axis]};
    if (std::ranges::find(axes, static_cast<int>(axis)) != axes.end()) {
      transform.push_back(dim);
    } else {
      batch.push_back(dim);
    }
  }
  return GuruLayout(std::move(transform), std::move(batch));
}

}  // namespace Internal

namespace Ranges {

/**
 * @brief A guru layout for transforming a contiguous row-major array along
 * the given axes.
 * @details Both sides have the same shape, so this suits complex-to-complex
 * and real-to-real transforms, in place or out of place. The axes not listed
 * become the repetition loop, which is what lets an interior axis be
 * transformed in a single plan.
 *
 * @code
 * // One plan that transforms axis 1 of every (i0, i2) line of a
 * // (n0, n1, n2) array. The advanced interface cannot express this.
 * auto layout = Ranges::TransformAlong({n0, n1, n2}, {1});
 * @endcode
 *
 * @param shape The extent along each axis.
 * @param axes The axes to transform. Order does not matter; they are used in
 * increasing order, which is the order the data is laid out in.
 * @return The corresponding layout.
 * @throws std::invalid_argument if the shape or the axis list is malformed.
 */
[[nodiscard]] inline GuruLayout TransformAlong(
    const std::vector<std::ptrdiff_t>& shape, const std::vector<int>& axes) {
  const auto sorted = FFTWpp::Internal::CheckedAxes(shape, axes);
  const auto strides = RowMajorStrides(shape);
  return FFTWpp::Internal::Assemble(shape, strides, strides, sorted);
}

/**
 * @brief The shape of the halfcomplex array produced by transforming `shape`
 * along `axes`.
 * @details The last of the transformed axes holds `n / 2 + 1` elements rather
 * than `n`; every other axis is unchanged.
 * @param shape The shape of the real array.
 * @param axes The axes to transform.
 * @return The shape of the corresponding halfcomplex array.
 * @throws std::invalid_argument if the shape or the axis list is malformed.
 */
[[nodiscard]] inline std::vector<std::ptrdiff_t> HalfcomplexShape(
    std::vector<std::ptrdiff_t> shape, const std::vector<int>& axes) {
  const auto sorted = FFTWpp::Internal::CheckedAxes(shape, axes);
  const auto last = static_cast<std::size_t>(sorted.back());
  shape[last] = shape[last] / 2 + 1;
  return shape;
}

/**
 * @brief A guru layout for a real-to-complex transform of a contiguous
 * row-major array along the given axes.
 * @details The real array has `shape`; the complex array has
 * `HalfcomplexShape(shape, axes)`, and both are taken to be contiguous and
 * row-major. Extents are the real ones, as FFTW's guru interface requires.
 * @param shape The shape of the *real* array.
 * @param axes The axes to transform.
 * @return The corresponding layout, with real strides on the input side.
 * @throws std::invalid_argument if the shape or the axis list is malformed.
 */
[[nodiscard]] inline GuruLayout RealToComplexTransformAlong(
    const std::vector<std::ptrdiff_t>& shape, const std::vector<int>& axes) {
  const auto sorted = FFTWpp::Internal::CheckedAxes(shape, axes);
  return FFTWpp::Internal::Assemble(
      shape, RowMajorStrides(shape),
      RowMajorStrides(HalfcomplexShape(shape, sorted)), sorted);
}

/**
 * @brief A guru layout for a complex-to-real transform of a contiguous
 * row-major array along the given axes.
 * @details The mirror of `RealToComplexTransformAlong`: the halfcomplex
 * strides are on the input side and the real strides on the output side.
 * @param shape The shape of the *real* array.
 * @param axes The axes to transform.
 * @return The corresponding layout, with halfcomplex strides on the input
 * side.
 * @throws std::invalid_argument if the shape or the axis list is malformed.
 */
[[nodiscard]] inline GuruLayout ComplexToRealTransformAlong(
    const std::vector<std::ptrdiff_t>& shape, const std::vector<int>& axes) {
  const auto sorted = FFTWpp::Internal::CheckedAxes(shape, axes);
  return FFTWpp::Internal::Assemble(
      shape, RowMajorStrides(HalfcomplexShape(shape, sorted)),
      RowMajorStrides(shape), sorted);
}

//----------------------------------------------------------//
//                        GuruPlan                          //
//----------------------------------------------------------//

/**
 * @class GuruPlan
 * @brief An RAII wrapper for a plan built through FFTW's guru interface.
 * @details The counterpart of `Ranges::Plan` for layouts the advanced
 * interface cannot express. It takes ranges directly rather than `View`s,
 * because a guru descriptor couples both sides in one object and so cannot
 * hang off a one-sided `Layout`.
 *
 * Everything else matches `Ranges::Plan`: the transform type follows from the
 * value types, the plan is destroyed with the object, copying builds an
 * equivalent new plan, moving transfers the handle, and new-array execution
 * can be checked.
 *
 * @code
 * // Transform axis 1 of a row-major (n0, n1, n2) array, in one plan.
 * auto layout = Ranges::TransformAlong({n0, n1, n2}, {1});
 * auto plan = Ranges::GuruPlan(in, out, layout, Measure, Forward);
 * plan.Execute();
 * @endcode
 *
 * @tparam InView The type of the input view.
 * @tparam OutView The type of the output view.
 * @requires Both must be views, not containers, and their precision must
 * match. Requiring a view is what makes the deduction guide below the only
 * viable one: were a container deducible, passing one by value would build
 * the plan on a *copy* of the caller's data, and every transform would write
 * somewhere the caller cannot see.
 */
template <NumericConcepts::RealOrComplexWritableView InView,
          NumericConcepts::RealOrComplexWritableView OutView>
requires NumericConcepts::SameRangePrecision<InView, OutView>
class GuruPlan {
  using InType = std::ranges::range_value_t<InView>;
  using OutType = std::ranges::range_value_t<OutView>;
  using Real = NumericConcepts::RemoveComplex<InType>;

  static constexpr auto InFormat =
      NumericConcepts::Complex<InType> && NumericConcepts::Real<OutType>
          ? Format::Halfcomplex
          : Format::Full;
  static constexpr auto OutFormat =
      NumericConcepts::Real<InType> && NumericConcepts::Complex<OutType>
          ? Format::Halfcomplex
          : Format::Full;

 public:
  /** @brief Default construction is deleted; a plan needs data and a layout. */
  GuruPlan() = delete;

  /**
   * @brief Constructor for complex-to-complex transforms.
   * @param in The input data range.
   * @param out The output data range.
   * @param layout The guru layout describing the transform.
   * @param flag The planner flag.
   * @param direction `Forward` or `Backward`.
   */
  GuruPlan(InView in, OutView out, GuruLayout layout, Flag flag,
           Direction direction)
  requires NumericConcepts::Complex<InType> and
               NumericConcepts::Complex<OutType>
      : _in{std::move(in)},
        _out{std::move(out)},
        _layout{std::move(layout)},
        _flag{flag},
        _direction{direction} {
    ValidateInputs();
    MakePlan(_flag);
  }

  /**
   * @brief Constructor for real-to-complex and complex-to-real transforms.
   * @param in The input data range.
   * @param out The output data range.
   * @param layout The guru layout, whose extents are those of the real array.
   * @param flag The planner flag.
   */
  GuruPlan(InView in, OutView out, GuruLayout layout, Flag flag)
  requires(NumericConcepts::Complex<InType> and
           NumericConcepts::Real<OutType>) or
              (NumericConcepts::Real<InType> and
               NumericConcepts::Complex<OutType>)
      : _in{std::move(in)},
        _out{std::move(out)},
        _layout{std::move(layout)},
        _flag{flag} {
    ValidateInputs();
    MakePlan(_flag);
  }

  /**
   * @brief Constructor for real-to-real transforms.
   * @details As with `Ranges::Plan`, a kind list shorter than the transform
   * rank is extended with its last entry.
   * @param in The input data range.
   * @param out The output data range.
   * @param layout The guru layout describing the transform.
   * @param flag The planner flag.
   * @param kinds The transform kind for each transform dimension.
   */
  GuruPlan(InView in, OutView out, GuruLayout layout, Flag flag,
           std::vector<RealKind> kinds)
  requires NumericConcepts::Real<InType> and NumericConcepts::Real<OutType>
      : _in{std::move(in)},
        _out{std::move(out)},
        _layout{std::move(layout)},
        _flag{flag},
        _kinds{std::move(kinds)} {
    CompleteKinds();
    ValidateInputs();
    MakePlan(_flag);
  }

  /**
   * @brief Constructor for real-to-real transforms taking kinds directly.
   * @tparam RealKinds A parameter pack of `FFTWpp::RealKind`.
   * @param in The input data range.
   * @param out The output data range.
   * @param layout The guru layout describing the transform.
   * @param flag The planner flag.
   * @param kinds The transform kind for each transform dimension.
   */
  template <typename... RealKinds>
  requires(sizeof...(RealKinds) > 0) and
          (std::same_as<RealKinds, RealKind> && ...)
  GuruPlan(InView in, OutView out, GuruLayout layout, Flag flag,
           RealKinds... kinds)
      : GuruPlan(std::move(in), std::move(out), std::move(layout), flag,
                 std::vector<RealKind>{kinds...}) {}

  /**
   * @brief Copy constructor. Builds an equivalent, independent plan.
   * @details As with `Ranges::Plan`, a plan created with anything other than
   * `Estimate` is rebuilt with `WisdomOnly`, so that the copy reuses the
   * original's measurements rather than overwriting the data to repeat them.
   */
  GuruPlan(const GuruPlan& other)
      : _in{other._in},
        _out{other._out},
        _layout{other._layout},
        _flag{other._flag},
        _direction{other._direction},
        _kinds{other._kinds} {
    MakePlan(_flag == Estimate ? Estimate : WisdomOnly);
  }

  /** @brief Move constructor. Transfers the FFTW handle. */
  GuruPlan(GuruPlan&& other)
      : _in{std::move(other._in)},
        _out{std::move(other._out)},
        _layout{std::move(other._layout)},
        _flag{other._flag},
        _direction{std::move(other._direction)},
        _kinds{std::move(other._kinds)},
        _plan{std::move(other._plan)},
        _inAlignment{other._inAlignment},
        _outAlignment{other._outAlignment} {
    other.Pointer() = nullptr;
  }

  /** @brief Copy assignment. */
  auto& operator=(const GuruPlan& other) {
    if (this == &other) return *this;
    GuruPlan replacement(other);
    Swap(replacement);
    return *this;
  }

  /** @brief Move assignment. */
  auto& operator=(GuruPlan&& other) {
    if (this == &other) return *this;
    GuruPlan replacement(std::move(other));
    Swap(replacement);
    return *this;
  }

  /** @brief Destructor. Destroys the underlying FFTW plan. */
  ~GuruPlan() { Destroy(); }

  /** @brief The raw FFTW plan handle. */
  auto Pointer() const {
    if constexpr (NumericConcepts::Float<Real>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  /** @brief A reference to the raw FFTW plan handle. */
  auto& Pointer() {
    if constexpr (NumericConcepts::Float<Real>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  /** @brief Whether the underlying plan is null. */
  auto IsNull() const { return Pointer() == nullptr; }

  /** @brief The layout the plan was built from. */
  const GuruLayout& Layout() const { return _layout; }

  /** @brief The planner flag the plan was created with. */
  [[nodiscard]] auto PlannerFlag() const { return _flag; }

  /**
   * @brief The normalisation factor for an inverse transform.
   * @details `1 / N`, where `N` is the logical size of one transform: the
   * product of the transform extents, which for a real-to-complex or
   * complex-to-real transform are the real-space ones. For a real-to-real
   * transform each dimension contributes the logical size of its own kind.
   * @return The normalisation factor, cast to the output value type.
   */
  auto Normalisation() const {
    auto dim = std::ptrdiff_t{1};
    if constexpr (NumericConcepts::Real<InType> &&
                  NumericConcepts::Real<OutType>) {
      const auto& kinds = std::get<std::vector<RealKind>>(_kinds);
      for (std::size_t i = 0; i < _layout.Transform().size(); ++i) {
        dim *= kinds[i].LogicalDimension(
            static_cast<int>(_layout.Transform()[i].n));
      }
    } else {
      dim = _layout.TransformSize();
    }
    return static_cast<OutType>(1) / static_cast<OutType>(dim);
  }

  /** @brief Executes the plan on the ranges given at construction. */
  void Execute() { FFTWpp::Execute(Pointer()); }

  /**
   * @brief Executes the plan on new buffers, unchecked.
   * @details Valid only for buffers matching the planning buffers in size and
   * in FFTW alignment class; see `Ranges::Plan::Execute`.
   */
  template <NumericConcepts::RealOrComplexWritableRange NewInView,
            NumericConcepts::RealOrComplexWritableRange NewOutView>
  requires NumericConcepts::SameRangeValueType<InView, NewInView> &&
           NumericConcepts::SameRangeValueType<OutView, NewOutView>
  void Execute(NewInView&& in, NewOutView&& out) {
    assert(CanExecuteOn(in, out) &&
           "new-array execution requires matching sizes and alignment");
    FFTWpp::Execute(Pointer(), std::ranges::data(in), std::ranges::data(out));
  }

  /** @brief Whether the plan may be executed on the given buffers. */
  template <NumericConcepts::RealOrComplexWritableRange NewInView,
            NumericConcepts::RealOrComplexWritableRange NewOutView>
  requires NumericConcepts::SameRangeValueType<InView, NewInView> &&
           NumericConcepts::SameRangeValueType<OutView, NewOutView>
  [[nodiscard]] bool CanExecuteOn(NewInView&& in, NewOutView&& out) const {
    return std::cmp_greater_equal(std::ranges::size(in),
                                  _layout.Extent(Side::Input, InFormat)) &&
           std::cmp_greater_equal(std::ranges::size(out),
                                  _layout.Extent(Side::Output, OutFormat)) &&
           (IgnoresAlignment() ||
            (FFTWpp::AlignmentOf(std::ranges::data(in)) == _inAlignment &&
             FFTWpp::AlignmentOf(std::ranges::data(out)) == _outAlignment));
  }

  /**
   * @brief Whether this plan was created with `Unaligned`, and so makes no
   * assumption about the alignment of the arrays it runs on.
   * @details Such a plan may be executed on any correctly sized buffer, at the
   * cost of the SIMD kernels an aligned plan could have used.
   */
  [[nodiscard]] bool IgnoresAlignment() const {
    return (static_cast<unsigned>(_flag) & FFTW_UNALIGNED) != 0;
  }

  /**
   * @brief Executes the plan on new buffers, having checked that it is valid.
   * @throws std::invalid_argument if the buffers are too small or differ in
   * alignment class from the planning buffers.
   */
  template <NumericConcepts::RealOrComplexWritableRange NewInView,
            NumericConcepts::RealOrComplexWritableRange NewOutView>
  requires NumericConcepts::SameRangeValueType<InView, NewInView> &&
           NumericConcepts::SameRangeValueType<OutView, NewOutView>
  void ExecuteChecked(NewInView&& in, NewOutView&& out) {
    if (!CanExecuteOn(in, out)) {
      throw std::invalid_argument(
          "FFTWpp::Ranges::GuruPlan::ExecuteChecked: the given buffers do not "
          "match the size and alignment class of the buffers this plan was "
          "created for, so new-array execution would be undefined");
    }
    FFTWpp::Execute(Pointer(), std::ranges::data(in), std::ranges::data(out));
  }

  /** @brief FFTW's alignment class for the planning input buffer. */
  [[nodiscard]] auto InputAlignment() const { return _inAlignment; }
  /** @brief FFTW's alignment class for the planning output buffer. */
  [[nodiscard]] auto OutputAlignment() const { return _outAlignment; }

 private:
  InView _in;
  OutView _out;
  GuruLayout _layout;
  Flag _flag;
  std::variant<std::monostate, Direction> _direction;
  std::variant<std::monostate, std::vector<RealKind>> _kinds;
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> _plan;
  int _inAlignment = 0;
  int _outAlignment = 0;

  void CompleteKinds()
  requires(NumericConcepts::Real<InType> && NumericConcepts::Real<OutType>)
  {
    auto& kinds = std::get<std::vector<RealKind>>(_kinds);
    if (kinds.empty() ||
        std::cmp_greater(kinds.size(), _layout.Transform().size())) {
      throw std::invalid_argument(
          "FFTWpp::Ranges::GuruPlan: an R2R plan requires between one and "
          "rank transform kinds");
    }
    kinds.resize(_layout.Transform().size(), kinds.back());
  }

  void ValidateInputs() const {
    _layout.ValidateNonOverlapping(Side::Input, InFormat);
    _layout.ValidateNonOverlapping(Side::Output, OutFormat);

    const auto check = [](std::size_t given, std::ptrdiff_t needed,
                          const char* which) {
      if (std::cmp_less(given, needed)) {
        throw std::invalid_argument(
            std::string("FFTWpp::Ranges::GuruPlan: the ") + which +
            " range holds " + std::to_string(given) +
            " elements but the layout reaches " + std::to_string(needed));
      }
    };
    check(std::ranges::size(_in), _layout.Extent(Side::Input, InFormat),
          "input");
    check(std::ranges::size(_out), _layout.Extent(Side::Output, OutFormat),
          "output");
  }

  void MakePlan(Flag flag) {
    const auto rank = _layout.Rank();
    const auto batchRank = _layout.BatchRank();
    const auto* dims = _layout.Transform().data();
    const auto* batch = _layout.Batch().data();
    auto* in = std::ranges::data(_in);
    auto* out = std::ranges::data(_out);

    if constexpr (NumericConcepts::Complex<InType> &&
                  NumericConcepts::Complex<OutType>) {
      _plan = FFTWpp::Plan(rank, dims, batchRank, batch, in, out,
                           std::get<Direction>(_direction), flag);
    } else if constexpr (NumericConcepts::Real<InType> &&
                         NumericConcepts::Real<OutType>) {
      auto kinds = std::vector<fftw_r2r_kind>();
      kinds.reserve(_layout.Transform().size());
      for (auto kind : std::get<std::vector<RealKind>>(_kinds)) {
        kinds.push_back(static_cast<fftw_r2r_kind>(kind));
      }
      _plan = FFTWpp::Plan(rank, dims, batchRank, batch, in, out, kinds.data(),
                           flag);
    } else {
      _plan = FFTWpp::Plan(rank, dims, batchRank, batch, in, out, flag);
    }

    if (IsNull()) {
      throw std::runtime_error(
          flag == WisdomOnly
              ? "FFTW failed to create a guru plan: no wisdom is available "
                "for it"
              : "FFTW failed to create a guru plan");
    }
    _inAlignment = FFTWpp::AlignmentOf(in);
    _outAlignment = FFTWpp::AlignmentOf(out);
    FFTWpp::Internal::LivePlanCounter().fetch_add(1, std::memory_order_relaxed);
  }

  void Destroy() noexcept {
    if (IsNull()) return;
    FFTWpp::Destroy(Pointer());
    Pointer() = nullptr;
    FFTWpp::Internal::LivePlanCounter().fetch_sub(1, std::memory_order_relaxed);
  }

  void Swap(GuruPlan& other) {
    using std::swap;
    swap(_in, other._in);
    swap(_out, other._out);
    swap(_layout, other._layout);
    swap(_flag, other._flag);
    swap(_direction, other._direction);
    swap(_kinds, other._kinds);
    swap(_plan, other._plan);
    swap(_inAlignment, other._inAlignment);
    swap(_outAlignment, other._outAlignment);
  }
};

/**
 * @brief Deduction guide, so that containers may be passed directly.
 */
template <std::ranges::range R1, std::ranges::range R2, typename... Args>
GuruPlan(R1&&, R2&&, GuruLayout, Args...)
    -> GuruPlan<std::ranges::views::all_t<R1>, std::ranges::views::all_t<R2>>;

}  // namespace Ranges

/**
 * @brief The storage each side of a guru transform requires.
 * @details The counterpart of `DataSize(dimensions...)` for the guru
 * interface. The halfcomplex side is accounted for automatically from the
 * value types.
 * @tparam InType The value type of the input data.
 * @tparam OutType The value type of the output data.
 * @param layout The layout describing the transform.
 * @return A pair whose `first` is the required input size and `second` the
 * required output size.
 */
template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType>
[[nodiscard]] inline auto DataSize(const Ranges::GuruLayout& layout) {
  using Ranges::Format;
  using Ranges::Side;
  constexpr auto inFormat =
      NumericConcepts::Complex<InType> && NumericConcepts::Real<OutType>
          ? Format::Halfcomplex
          : Format::Full;
  constexpr auto outFormat =
      NumericConcepts::Real<InType> && NumericConcepts::Complex<OutType>
          ? Format::Halfcomplex
          : Format::Full;
  return std::pair(layout.Extent(Side::Input, inFormat),
                   layout.Extent(Side::Output, outFormat));
}

}  // namespace FFTWpp

#endif  // FFTWPP_GURU_GUARD_H
