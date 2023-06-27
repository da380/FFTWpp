#ifndef FFTWPlan_GUARD_H
#define FFTWPlan_GUARD_H

#include <cassert>
#include <variant>

#include "FFTWConcepts.h"
#include "fftw3.h"

namespace FFTW {

template <std::floating_point Float>
class Plan {
 public:
  Plan();

  // Constructor for 1D complex to complex.
  template <typename InputIt, typename OutputIt>
  Plan(InputIt, InputIt, OutputIt, DirectionFlag,
       PlanFlag = PlanFlag::Estimate) requires
      ComplexIteratorWithPrecision<InputIt, Float> and
      ComplexIteratorWithPrecision<OutputIt, Float>;

  // Constructor for 1D real to complex.
  template <typename InputIt, typename OutputIt>
  Plan(InputIt, InputIt, OutputIt, PlanFlag = PlanFlag::Estimate) requires
      RealIteratorWithPrecision<InputIt, Float> and
      ComplexIteratorWithPrecision<OutputIt, Float>;

  // Constructor for 1D complex to real.
  template <typename InputIt, typename OutputIt>
  Plan(InputIt, InputIt, OutputIt, PlanFlag = PlanFlag::Estimate) requires
      ComplexIteratorWithPrecision<InputIt, Float> and
      RealIteratorWithPrecision<OutputIt, Float>;

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
  template <typename InputIt, typename OutputIt>
  void execute(InputIt in_first, OutputIt out_first)
    requires
    ComplexIteratorWithPrecision<InputIt, Float> and
    ComplexIteratorWithPrecision<OutputIt, Float>
  {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft(ConvertPlan(),ComplexCast(&*in_first),ComplexCast(&*out_first));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft(ConvertPlan(),ComplexCast(&*in_first),ComplexCast(&*out_first));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft(ConvertPlan(),ComplexCast(&*in_first),ComplexCast(&*out_first));
    }
  }



  // Execute the plan given new real to complex data.
  template <typename InputIt, typename OutputIt>
  void execute(InputIt in_first, OutputIt out_first)
    requires
    RealIteratorWithPrecision<InputIt, Float> and
    ComplexIteratorWithPrecision<OutputIt, Float>
  {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_r2c(ConvertPlan(),&*in_first,ComplexCast(&*out_first));
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_r2c(ConvertPlan(),&*in_first,ComplexCast(&*out_first));
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_r2c(ConvertPlan(),&*in_first,ComplexCast(&*out_first));
    }
  }


  // Execute the plan given new complex to real data.
  template <typename InputIt, typename OutputIt>
  void execute(InputIt in_first, OutputIt out_first)
    requires
    ComplexIteratorWithPrecision<InputIt, Float> and
    RealIteratorWithPrecision<OutputIt, Float>
  {
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_c2r(ConvertPlan(),ComplexCast(&*in_first),&*out_first);
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_c2r(ConvertPlan(),ComplexCast(&*in_first),&*out_first);
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_c2r(ConvertPlan(),ComplexCast(&*in_first),&*out_first);
    }
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

  // Reinterpret cast std::complex* to fftw_complex*.
  auto ComplexCast(std::complex<Float>* z) {
    if constexpr (IsSingle<Float>) {
      return reinterpret_cast<fftwf_complex*>(z);
    }
    if constexpr (IsDouble<Float>) {
      return reinterpret_cast<fftw_complex*>(z);
    }
    if constexpr (IsLongDouble<Float>) {
      return reinterpret_cast<fftwl_complex*>(z);
    }
  }
};

// Constructor for 1D complex data.
template <std::floating_point Float>
template <typename InputIt, typename OutputIt>
Plan<Float>::Plan(InputIt in_first, InputIt in_last, OutputIt out_first,
                  DirectionFlag direction, PlanFlag flag) requires
    ComplexIteratorWithPrecision<InputIt, Float> and
    ComplexIteratorWithPrecision<OutputIt, Float> {
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

// Constructor for 1D real to complex transformation.
template <std::floating_point Float>
template <typename InputIt, typename OutputIt>
Plan<Float>::Plan(InputIt in_first, InputIt in_last, OutputIt out_first,
                  PlanFlag flag) requires
    RealIteratorWithPrecision<InputIt, Float> and
    ComplexIteratorWithPrecision<OutputIt, Float> {
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

// Constructor for 1D complex to real transformation.
template <std::floating_point Float>
template <typename InputIt, typename OutputIt>
Plan<Float>::Plan(InputIt in_first, InputIt in_last, OutputIt out_first,
                  PlanFlag flag) requires
    ComplexIteratorWithPrecision<InputIt, Float> and
    RealIteratorWithPrecision<OutputIt, Float> {
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
