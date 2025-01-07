#ifndef FFTWPP_PLAN_GUARD_H
#define FFTWPP_PLAN_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <initializer_list>
#include <ranges>
#include <variant>

#include "Core.h"
#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "Views.h"
#include "fftw3.h"

namespace FFTWpp {

namespace Ranges {

template <NumericConcepts::RealOrComplexWritableRange InView,
          NumericConcepts::RealOrComplexWritableRange OutView>
requires NumericConcepts::SameRangePrecision<InView, OutView>

class Plan {
  using InType = std::ranges::range_value_t<InView>;
  using OutType = std::ranges::range_value_t<OutView>;
  using Real = NumericConcepts::RemoveComplex<InType>;

 public:
  // Remove default constructor;
  Plan() = delete;

  // Constructor for C2C.
  Plan(View<InView> in, View<OutView> out, Flag flag, Direction direction)
  requires NumericConcepts::Complex<InType> and
               NumericConcepts::Complex<OutType>
      : _in{in}, _out{out}, _flag{flag}, _direction{direction} {
    assert(CheckInputs());
    MakePlan(_flag);
  }

  // Constructor for R2C or C2R.
  Plan(View<InView> in, View<OutView> out, Flag flag)
  requires(NumericConcepts::Complex<InType> and
           NumericConcepts::Real<OutType>) or
              (NumericConcepts::Real<InType> and
               NumericConcepts::Complex<OutType>)
      : _in{in}, _out{out}, _flag{flag} {
    assert(CheckInputs());
    MakePlan(_flag);
  }

  // Constructor for R2R.
  template <typename... RealKinds>
  requires(sizeof...(RealKinds) > 0) and
              (std::same_as<RealKinds, RealKind> && ...)
  Plan(View<InView> in, View<OutView> out, Flag flag, RealKinds... kinds)
      : _in{in},
        _out{out},
        _flag{flag},
        _kinds{std::vector<RealKind>{kinds...}} {
    assert(Kinds().size() <= _in.Rank());
    if (Kinds().size() < _in.Rank()) {
      auto kinds = std::get<std::vector<RealKind>>(_kinds);
      while (kinds.size() < _in.Rank()) {
        kinds.push_back(kinds.back());
      }
      _kinds = kinds;
    }
    assert(CheckInputs());
    MakePlan(_flag);
  }

  // Copy constructor.
  Plan(const Plan& other)
      : _in{other._in},
        _out{other._out},
        _flag{other._flag},
        _direction{other._direction},
        _kinds{other._kinds} {
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
  }

  // Move constructor.
  Plan(Plan&& other)
      : _in{std::move(other._in)},
        _out{std::move(other._out)},
        _flag{std::move(other._flag)},
        _direction{std::move(other._direction)},
        _kinds{std::move(other._kinds)} {
    other.Destroy();
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
  }

  // Copy assignment.
  auto& operator=(const Plan& other) {
    _in = other._in;
    _out = other._out;
    _flag = other._flag;
    _direction = other._direction;
    _kinds = other._kinds;
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
    return *this;
  }

  // Move assignment.
  auto& operator=(Plan&& other) {
    other.Destroy();
    _in = std::move(other._in);
    _out = std::move(other._out);
    _flag = std::move(other._flag);
    _direction = std::move(other._direction);
    _kinds = std::move(other._kinds);
    auto flag = _flag == Estimate ? Estimate : WisdomOnly;
    MakePlan(flag);
    return *this;
  }

  // Destructor.
  ~Plan() { Destroy(); }

  // return pointer to the fftw3 plan.
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

  // Returns true is plan is not set up.
  auto IsNull() { return Pointer() == nullptr; }

  // Normalisation factor for inverse transformations.
  auto Normalisation() const {
    int dim;
    if constexpr (NumericConcepts::Complex<InType> ||
                  NumericConcepts::Complex<OutType>) {
      dim = std::ranges::fold_left_first(_out.N(), std::multiplies<>()).value();
    } else {
      dim = std::ranges::fold_left_first(
                std::ranges::views::zip_transform(
                    [](auto n, auto kind) { return kind.LogicalDimension(n); },
                    _out.N(), Kinds()),
                std::multiplies<>())
                .value();
    }
    return static_cast<OutType>(1) / static_cast<OutType>(dim);
  }

  // Execute the plan.
  void Execute() { FFTWpp::Execute(Pointer()); }

  // Execute using new data.
  template <NumericConcepts::RealOrComplexWritableRange NewInView,
            NumericConcepts::RealOrComplexWritableRange NewOutView>
  requires NumericConcepts::SameRangeValueType<InView, NewInView> &&
           NumericConcepts::SameRangeValueType<OutView, NewOutView>
  void Execute(NewInView in, NewOutView out) {
    FFTWpp::Execute(Pointer(), in.data(), out.data());
  }

 private:
  View<InView> _in;
  View<OutView> _out;
  Flag _flag;
  std::variant<std::monostate, Direction> _direction;
  std::variant<std::monostate, std::vector<RealKind>> _kinds;
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> _plan;

  auto CheckInputs() const {
    if (_in.Rank() != _out.Rank()) return false;
    if (_in.HowMany() != _out.HowMany()) return false;
    if constexpr (std::same_as<InType, OutType>) {
      return std::ranges::equal(_in.N(), _out.N());
    } else if constexpr (NumericConcepts::Complex<InType> &&
                         NumericConcepts::Real<OutType>) {
      return std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::take(1),
                 _out.N() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x == y / 2 + 1; }) &&
             std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::drop(1),
                 _out.N() | std::views::reverse | std::views::drop(1));
    } else if constexpr (NumericConcepts::Real<InType> &&
                         NumericConcepts::Complex<OutType>) {
      return std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::take(1),
                 _out.N() | std::views::reverse | std::views::take(1),
                 [](auto x, auto y) { return x / 2 + 1 == y; }) &&
             std::ranges::equal(
                 _in.N() | std::views::reverse | std::views::drop(1),
                 _out.N() | std::views::reverse | std::views::drop(1));
    }
  }

  void MakePlan(Flag flag) {
    if constexpr (NumericConcepts::Complex<InType> &&
                  NumericConcepts::Complex<OutType>) {
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(),
                           std::get<Direction>(_direction), flag);
    } else if constexpr ((NumericConcepts::Complex<InType> &&
                          NumericConcepts::Real<OutType>)) {
      _plan = FFTWpp::Plan(_out.Rank(), _out.NPointer(), _out.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), flag);
    } else if constexpr ((NumericConcepts::Real<InType> &&
                          NumericConcepts::Complex<OutType>)) {
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), flag);
    } else if constexpr (NumericConcepts::Real<InType> &&
                         NumericConcepts::Real<OutType>) {
      auto kinds = std::vector<fftw_r2r_kind>();
      std::transform(
          Kinds().begin(), Kinds().end(), std::back_inserter(kinds),
          [](auto kind) { return static_cast<fftw_r2r_kind>(kind); });
      _plan = FFTWpp::Plan(_in.Rank(), _in.NPointer(), _in.HowMany(),
                           _in.DataPointer(), _in.EmbedPointer(), _in.Stride(),
                           _in.Dist(), _out.DataPointer(), _out.EmbedPointer(),
                           _out.Stride(), _out.Dist(), kinds.data(), flag);
    }
    assert(!IsNull());
  }

  auto Kinds() const
  requires(NumericConcepts::Real<InType> && NumericConcepts::Real<OutType>)
  {
    return std::ranges::views::all(std::get<std::vector<RealKind>>(_kinds));
  }

  // Destroy the stored plan.
  void Destroy() {
    if (IsNull()) return;
    FFTWpp::Destroy(Pointer());
    Pointer() = nullptr;
  }
};

}  // namespace Ranges

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
