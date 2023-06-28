#ifndef FFTWPlan_GUARD_H
#define FFTWPlan_GUARD_H

#include <algorithm>
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
  Plan(size_t n, InputIt, OutputIt, DirectionFlag,
       PlanFlag) requires C2CIteratorPair<InputIt, OutputIt>;

  // Constructor for 1D real to complex transformation
  Plan(size_t n, InputIt, OutputIt,
       PlanFlag) requires R2CIteratorPair<InputIt, OutputIt>;

  // Constructor for 1D complex to real transformation
  Plan(size_t n, InputIt, OutputIt,
       PlanFlag) requires C2RIteratorPair<InputIt, OutputIt>;

  // Constructor for 1D real to real transformations
  Plan(size_t n, InputIt, OutputIt, DirectionFlag,
       PlanFlag) requires R2RIteratorPair<InputIt, OutputIt>;

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
  void execute(InputIt in,
               OutputIt out) requires C2CIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft(ConvertPlan(), ComplexCast(&*in), ComplexCast(&*out));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft(ConvertPlan(), ComplexCast(&*in), ComplexCast(&*out));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft(ConvertPlan(), ComplexCast(&*in), ComplexCast(&*out));
    }
  }

  // Execute the plan given new real to complex data.
  void execute(InputIt in,
               OutputIt out) requires R2CIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_r2c(ConvertPlan(), &*in, ComplexCast(&*out));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_r2c(ConvertPlan(), &*in, ComplexCast(&*out));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_r2c(ConvertPlan(), &*in, ComplexCast(&*out));
    }
  }

  // Execute the plan given new complex to real data.
  void execute(InputIt in,
               OutputIt out) requires C2RIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in), &*out);
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in), &*out);
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_c2r(ConvertPlan(), ComplexCast(&*in), &*out);
    }
  }

  // Execute the plan given new real data.
  void execute(InputIt in,
               OutputIt out) requires R2RIteratorPair<InputIt, OutputIt> {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_r2r(ConvertPlan(), &*in, &*out);
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_r2r(ConvertPlan(), &*in, &*out);
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_r2r(ConvertPlan(), &*in, &*out);
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
    size_t n, InputIt in, OutputIt out, DirectionFlag direction, PlanFlag flag)
requires C2CIteratorPair<InputIt, OutputIt> : n{n} {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_1d(n, ComplexCast(&*in), ComplexCast(&*out),
                             ConvertDirectionFlag(direction),
                             ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_1d(n, ComplexCast(&*in), ComplexCast(&*out),
                            ConvertDirectionFlag(direction),
                            ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_1d(n, ComplexCast(&*in), ComplexCast(&*out),
                             ConvertDirectionFlag(direction),
                             ConvertPlanFlag(flag));
  }
}

// Constructor for 1D real to complex transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    size_t n, InputIt in, OutputIt out, PlanFlag flag)
requires R2CIteratorPair<InputIt, OutputIt> : n{n} {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_r2c_1d(n, &*in, ComplexCast(&*out),
                                 ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_r2c_1d(n, &*in, ComplexCast(&*out),
                                ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_r2c_1d(n, &*in, ComplexCast(&*out),
                                 ConvertPlanFlag(flag));
  }
}

// Constructor for 1D complex to real transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    size_t n, InputIt in, OutputIt out, PlanFlag flag)
requires C2RIteratorPair<InputIt, OutputIt> : n{n} {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_dft_c2r_1d(n, ComplexCast(&*in), &*out,
                                 ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_dft_c2r_1d(n, ComplexCast(&*in), &*out,
                                ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_dft_c2r_1d(n, ComplexCast(&*in), &*out,
                                 ConvertPlanFlag(flag));
  }
}

// Constructor for 1D real to real transformation
template <ScalarIterator InputIt, ScalarIterator OutputIt>
requires SamePrecision<InputIt, OutputIt> Plan<InputIt, OutputIt>::Plan(
    size_t n, InputIt in, OutputIt out, DirectionFlag direction, PlanFlag flag)
requires R2RIteratorPair<InputIt, OutputIt> : n{n} {
  if constexpr (IsSingle<Float>) {
    plan =
        fftwf_plan_r2r_1d(n, &*in, &*out, ConvertDirectionFlag<true>(direction),
                          ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan =
        fftw_plan_r2r_1d(n, &*in, &*out, ConvertDirectionFlag<true>(direction),
                         ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan =
        fftwl_plan_r2r_1d(n, &*in, &*out, ConvertDirectionFlag<true>(direction),
                          ConvertPlanFlag(flag));
  }
}

}  // namespace FFTW

#endif  // FFTWPlan1D_GUARD_H
