#ifndef FFTWPlan_GUARD_H
#define FFTWPlan_GUARD_H

#include <cassert>
#include <complex>
#include <variant>

#include "FFTWConcepts.h"
#include "FFTWFlags.h"
#include "FFTWMemory.h"
#include "fftw3.h"

namespace FFTW {

template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt>
class Plan {
 public:
  // Constructor for 1D complex to complex transformation
  Plan(InputIt, InputIt, OutputIt, DirectionFlag,
       PlanFlag) requires C2CIteratorPair<InputIt, OutputIt>;

  // Constructor for real to complex transformation
  Plan(InputIt, InputIt, OutputIt,
       PlanFlag) requires R2CIteratorPair<InputIt, OutputIt>;

  // Constructor for complex to real transformation
  Plan(InputIt, InputIt, OutputIt,
       PlanFlag) requires C2RIteratorPair<InputIt, OutputIt>;

  // Execute the plan.
  void execute() {
    if constexpr (IsSingle<Float>) {
      fftwf_execute(ConvertPlan());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute(ConvertPlan());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute(ConvertPlan());
    }
  }

  // Execute the plan given new complex data.
  void execute(InputIt in_first,
               OutputIt out_first) requires C2CIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft(ConvertPlan(), ComplexCast(&*in_first),
                        ComplexCast(&*out_first));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft(ConvertPlan(), ComplexCast(&*in_first),
                       ComplexCast(&*out_first));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft(ConvertPlan(), ComplexCast(&*in_first),
                        ComplexCast(&*out_first));
    }
  }

  // Execute the plan given new real to complex data.
  void execute(InputIt in_first,
               OutputIt out_first) requires R2CIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_r2c(ConvertPlan(), &*in_first,
                            ComplexCast(&*out_first));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_r2c(ConvertPlan(), &*in_first, ComplexCast(&*out_first));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_r2c(ConvertPlan(), &*in_first,
                            ComplexCast(&*out_first));
    }
  }

  // Execute the plan given new complex to real data.
  void execute(InputIt in_first,
               OutputIt out_first) requires C2RIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in_first),
                            &*out_first);
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in_first), &*out_first);
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in_first),
                            &*out_first);
    }
  }

  // Normalise the result of an inverse transformation.
  void normalise(OutputIt first, OutputIt last, OutputIt dest) {
    auto norm = static_cast<Float>(1) / static_cast<Float>(n);
    std::transform(first, last, dest,
                   [norm](OutputValueType x) { return x * norm; });
  }

  // Overload when the new values are written in place.
  void normalise(OutputIt first, OutputIt last) {
    normalise(first, last, first);
  }

  // Destructor.
  ~Plan() {
    if constexpr (IsSingle<Float>) {
      fftwf_destroy_plan(ConvertPlan());
    }
    if constexpr (IsDouble<Float>) {
      fftw_destroy_plan(ConvertPlan());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_destroy_plan(ConvertPlan());
    }
  }

 private:
  // Store some type aliases
  using Float = GetPrecision<InputIt>;
  using InputValueType = GetValueType<InputIt>;
  using OutputValueType = GetValueType<OutputIt>;

  // Store the dimension.
  size_t n;

  // Store the plan as a std::variant.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> plan;

  // Get plan in fftw3 form.
  auto ConvertPlan() {
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
};

// Constructor for 1D complex to complex transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    InputIt in_first, InputIt in_last, OutputIt out_first,
    DirectionFlag direction, PlanFlag flag)
requires C2CIteratorPair<InputIt, OutputIt> {
  n = in_last - in_first;
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_1d(
        n, ComplexCast(&*in_first), ComplexCast(&*out_first),
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_1d(
        n, ComplexCast(&*in_first), ComplexCast(&*out_first),
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_1d(
        n, ComplexCast(&*in_first), ComplexCast(&*out_first),
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
}

// Constructor for 1D real to complex transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    InputIt in_first, InputIt in_last, OutputIt out_first, PlanFlag flag)
requires R2CIteratorPair<InputIt, OutputIt> {
  n = in_last - in_first;
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_r2c_1d(n, &*in_first, ComplexCast(&*out_first),
                                 ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_r2c_1d(n, &*in_first, ComplexCast(&*out_first),
                                ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_r2c_1d(n, &*in_first, ComplexCast(&*out_first),
                                 ConvertPlanFlag(flag));
  }
}

// Constructor for 1D complex to real transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    InputIt in_first, InputIt in_last, OutputIt out_first, PlanFlag flag)
requires C2RIteratorPair<InputIt, OutputIt> {
  size_t m = in_last - in_first;
  n = 2 * (m - 1);
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_c2r_1d(n, ComplexCast(&*in_first), &*out_first,
                                 ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_c2r_1d(n, ComplexCast(&*in_first), &*out_first,
                                ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_c2r_1d(n, ComplexCast(&*in_first), &*out_first,
                                 ConvertPlanFlag(flag));
  }
}

}  // namespace FFTW

#endif  // FFTWPlan1D_GUARD_H
