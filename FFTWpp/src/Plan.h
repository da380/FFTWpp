#ifndef FFTWPP_PLAN_GUARD_H
#define FFTWPP_PLAN_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <numeric>
#include <variant>

#include "Concepts.h"
#include "Flags.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTWpp {

template <ScalarIterator InputIt, ScalarIterator OutputIt,
          typename PlanFlagExpression>
requires IteratorPair<InputIt, OutputIt>
class Plan {
 public:
  using Float = IteratorPrecision<InputIt>;
  using InputValueType = std::iter_value_t<InputIt>;
  using OutputValueType = std::iter_value_t<OutputIt>;
  using InRef = DataReference<InputIt>;
  using OutRef = DataReference<OutputIt>;

  // Constructors
  Plan() = delete;

  // Complex to complex constructor.
  Plan(InRef& in, OutRef& out, PlanFlagExpression flag,
       Direction direction) requires C2CIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{direction} {
    assert(in.Comparable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft(in.rank(), in.n(), in.howmany(), in.data(),
                                 in.embed(), in.stride(), in.dist(), out.data(),
                                 out.embed(), out.stride(), out.dist(),
                                 direction(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft(in.rank(), in.n(), in.howmany(), in.data(),
                                in.embed(), in.stride(), in.dist(), out.data(),
                                out.embed(), out.stride(), out.dist(),
                                direction(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft(in.rank(), in.n(), in.howmany(), in.data(),
                                 in.embed(), in.stride(), in.dist(), out.data(),
                                 out.embed(), out.stride(), out.dist(),
                                 direction(), flag());
    }
  }

  // Complex to real constructor.
  Plan(InRef& in, OutRef& out, PlanFlagExpression flag,
       Direction direction = Backward) requires
      C2RIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{Backward} {
    assert(in.Comparable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft_c2r(
          in.rank(), out.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft_c2r(
          in.rank(), out.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft_c2r(
          in.rank(), out.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }
  }

  // Real to complex constructor.
  Plan(InRef& in, OutRef& out, PlanFlagExpression flag,
       Direction direction = Forward) requires
      R2CIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{Forward} {
    assert(in.Comparable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft_r2c(
          in.rank(), in.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft_r2c(
          in.rank(), in.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft_r2c(
          in.rank(), in.n(), in.howmany(), in.data(), in.embed(), in.stride(),
          in.dist(), out.data(), out.embed(), out.stride(), out.dist(), flag());
    }
  }

  // Real to real constructor.
  Plan(InRef& in, OutRef& out, PlanFlagExpression flag,
       Direction direction) requires R2RIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{direction} {
    assert(in.Comparable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_r2r(in.rank(), in.n(), in.howmany(), in.data(),
                                 in.embed(), in.stride(), in.dist(), out.data(),
                                 out.embed(), out.stride(), out.dist(),
                                 direction.template operator()<true>(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_r2r(in.rank(), in.n(), in.howmany(), in.data(),
                                in.embed(), in.stride(), in.dist(), out.data(),
                                out.embed(), out.stride(), out.dist(),
                                direction.template operator()<true>(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_r2r(in.rank(), in.n(), in.howmany(), in.data(),
                                 in.embed(), in.stride(), in.dist(), out.data(),
                                 out.embed(), out.stride(), out.dist(),
                                 direction.template operator()<true>(), flag());
    }
  }

  // return plan as an appropriate fftw3 pointer.
  auto operator()() const {
    if constexpr (IsSingle<Float>) {
      return std::get<fftwf_plan>(plan);
    }
    if constexpr (IsDouble<Float>) {
      return std::get<fftw_plan>(plan);
    }
    if constexpr (IsLongDouble<Float>) {
      return std::get<fftwl_plan>(plan);
    }
  }

  // Execute the plan.
  void execute() {
    if constexpr (IsSingle<Float>) {
      fftwf_execute(this->operator()());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute(this->operator()());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute(this->operator()());
    }
  }

  // Execute the plan given new complex-complex data.
  void execute(InRef& newIn,
               OutRef& newOut) requires C2CIteratorPair<InputIt, OutputIt> {
    assert(in.Comparable(newIn));
    assert(out.Comparable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft(this->operator()(), newIn.data(), newOut.data());
    }
  }

  // Execute the plan given new complex-real data.
  void execute(InRef& newIn,
               OutRef& newOut) requires C2RIteratorPair<InputIt, OutputIt> {
    assert(in.Comparable(newIn));
    assert(out.Comparable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_c2r(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_c2r(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_c2r(this->operator()(), newIn.data(), newOut.data());
    }
  }

  // Execute the plan given new real-complex data.
  void execute(InRef& newIn,
               OutRef& newOut) requires R2CIteratorPair<InputIt, OutputIt> {
    assert(in.Comparable(newIn));
    assert(out.Comparable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_r2c(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_r2c(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_r2c(this->operator()(), newIn.data(), newOut.data());
    }
  }

  // Execute the plan given new real-real data.
  void execute(InRef& newIn,
               OutRef& newOut) requires R2RIteratorPair<InputIt, OutputIt> {
    assert(in.Comparable(newIn));
    assert(out.Comparable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_r2r(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_r2r(this->operator()(), newIn.data(), newOut.data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_r2r(this->operator()(), newIn.data(), newOut.data());
    }
  }

  void normalise() { out.normalise; }

 private:
  // Store data references.
  InRef in;
  OutRef out;

  // Store transform options
  Direction direction;
  PlanFlagExpression flag;

  // Store the plan as a std::variant.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> plan;
};

// Returns a plan for a 1D transformation given data as ranges.
template <typename InputRange, typename OutputRange,
          typename PlanFlagExpression>
auto MakePlan1D(InputRange& in, OutputRange& out, PlanFlagExpression flag,
                Direction direction = Forward) {
  auto inRef = MakeDataReference1D(in.begin(), in.end());
  auto outRef = MakeDataReference1D(out.begin(), out.end());
  return Plan(inRef, outRef, flag, direction);
}

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
