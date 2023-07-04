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
class Plan {
 public:
  // General complex to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt n, int howmany, InputIt in, IntIt inembed, int istride,
       int idist, OutputIt out, IntIt onembed, int ostride, int odist,
       DirectionFlag direction,
       PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt> {
    MakePlan(rank, n, howmany, in, inembed, istride, idist, out, onembed,
             ostride, odist, direction, flag);
  }

  // General real to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt n, int howmany, InputIt in, IntIt inembed, int istride,
       int idist, OutputIt out, IntIt onembed, int ostride, int odist,
       PlanFlag flag) requires R2CIteratorPair<InputIt, OutputIt> or
      C2RIteratorPair<InputIt, OutputIt> {
    MakePlan(rank, n, howmany, in, inembed, istride, idist, out, onembed,
             ostride, odist, flag);
  }

  // Constructor for 1D complex to complex transformation
  Plan(int dimension, InputIt in, OutputIt out, DirectionFlag direction,
       PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt>
      : dimension{dimension} {
    auto n = std::vector<int>(1, dimension);
    auto it = n.begin();
    MakePlan(1, it, 1, in, it, 1, 1, out, it, 1, 1, direction, flag);
  }

  // Constructor for 1D real to complex transformation
  Plan(int dimension, InputIt in, OutputIt out, PlanFlag flag) requires
      R2CIteratorPair<InputIt, OutputIt> or C2RIteratorPair<InputIt, OutputIt>
      : dimension{dimension} {
    auto n = std::vector<int>(1, dimension);
    auto it = n.begin();
    MakePlan(1, it, 1, in, it, 1, 1, out, it, 1, 1, flag);
  }
  

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
    auto norm = static_cast<Float>(1) / static_cast<Float>(dimension);
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
  using Float = IteratorPrecision<InputIt>;
  using InputValueType = IteratorValue<InputIt>;
  using OutputValueType = IteratorValue<OutputIt>;

  int dimension;

  // Store the plan as a std::variant.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> plan;

  // Make a general complex to complex plan
  template <IntegralIterator IntIt>
  void MakePlan(int, IntIt, int, InputIt, IntIt, int, int, OutputIt, IntIt, int,
                int, DirectionFlag,
                PlanFlag) requires C2CIteratorPair<InputIt, OutputIt>;

  // Make a general real to complex plan
  template <IntegralIterator IntIt>
  void MakePlan(int, IntIt, int, InputIt, IntIt, int, int, OutputIt, IntIt, int,
                int, PlanFlag) requires R2CIteratorPair<InputIt, OutputIt>;

  // Make a general complex to real plan
  template <IntegralIterator IntIt>
  void MakePlan(int, IntIt, int, InputIt, IntIt, int, int, OutputIt, IntIt, int,
                int, PlanFlag) requires C2RIteratorPair<InputIt, OutputIt>;

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

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt n, int howmany, InputIt in, IntIt inembed, int istride,
    int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    DirectionFlag direction,
    PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt> {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft(rank, &*n, howmany, ComplexCast(&*in), &*inembed,
                               istride, idist, ComplexCast(&*out), &*onembed,
                               ostride, odist, ConvertDirectionFlag(direction),
                               ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft(rank, &*n, howmany, ComplexCast(&*in), &*inembed,
                              istride, idist, ComplexCast(&*out), &*onembed,
                              ostride, odist, ConvertDirectionFlag(direction),
                              ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft(rank, &*n, howmany, ComplexCast(&*in), &*inembed,
                               istride, idist, ComplexCast(&*out), &*onembed,
                               ostride, odist, ConvertDirectionFlag(direction),
                               ConvertPlanFlag(flag));
  }
}

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt n, int howmany, InputIt in, IntIt inembed, int istride,
    int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag) requires R2CIteratorPair<InputIt, OutputIt> {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft_r2c(rank, &*n, howmany, &*in, &*inembed, istride,
                                   idist, ComplexCast(&*out), &*onembed,
                                   ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft_r2c(rank, &*n, howmany, &*in, &*inembed, istride,
                                  idist, ComplexCast(&*out), &*onembed, ostride,
                                  odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft_r2c(rank, &*n, howmany, &*in, &*inembed, istride,
                                   idist, ComplexCast(&*out), &*onembed,
                                   ostride, odist, ConvertPlanFlag(flag));
  }
}

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt n, int howmany, InputIt in, IntIt inembed, int istride,
    int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag) requires C2RIteratorPair<InputIt, OutputIt> {
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft_c2r(rank, &*n, howmany, ComplexCast(&*in),
                                   &*inembed, istride, idist, &*out, &*onembed,
                                   ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft_c2r(rank, &*n, howmany, ComplexCast(&*in),
                                  &*inembed, istride, idist, &*out, &*onembed,
                                  ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft_c2r(rank, &*n, howmany, ComplexCast(&*in),
                                   &*inembed, istride, idist, &*out, &*onembed,
                                   ostride, odist, ConvertPlanFlag(flag));
  }
}

}  // namespace FFTW

#endif  // FFTWPlan_GUARD_H
