#ifndef FFTWPlan_GUARD_H
#define FFTWPlan_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <variant>

#include "Concepts.h"
#include "Flags.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTW {

template <ScalarIterator InputIt, ScalarIterator OutputIt>
class Plan {
  // Store some type aliases
  using Float = IteratorPrecision<InputIt>;
  using InputValueType = IteratorValue<InputIt>;
  using OutputValueType = IteratorValue<OutputIt>;
  
 public:
  // General complex to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
       int istride, int idist, OutputIt out, IntIt onembed, int ostride,
       int odist, DirectionFlag direction,
       PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt> {
    MakePlan(rank, dimensions, howmany, in, inembed, istride, idist, out,
             onembed, ostride, odist, direction, flag);
  }

  // General real to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
       int istride, int idist, OutputIt out, IntIt onembed, int ostride,
       int odist, PlanFlag flag) requires R2CIteratorPair<InputIt, OutputIt> or
      C2RIteratorPair<InputIt, OutputIt> {
    MakePlan(rank, dimensions, howmany, in, inembed, istride, idist, out,
             onembed, ostride, odist, flag);
  }

  // Constructor for 1D complex to complex transformation
  Plan(int dimension, InputIt in, OutputIt out, DirectionFlag direction,
       PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt> {
    auto dimensions = std::vector<int>(1, dimension);
    auto it = dimensions.begin();
    MakePlan(1, it, 1, in, it, 1, 1, out, it, 1, 1, direction, flag);
  }

  // Constructor for 1D real to complex transformation
  Plan(int dimension, InputIt in, OutputIt out, PlanFlag flag) requires
      R2CIteratorPair<InputIt, OutputIt> or C2RIteratorPair<InputIt, OutputIt> {
    auto dimensions = std::vector<int>(1, dimension);
    auto it = dimensions.begin();
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
  void normalise(OutputIt first, OutputIt last, OutputIt dest) const {
    std::transform(first, last, dest,
                   [this](OutputValueType x) { return x * norm; });
  }

  // Overload when the new values are written in place.
  void normalise(OutputIt first, OutputIt last) {
    normalise(first, last, first);
  }

  Float GetNorm() const
  {
    return norm;
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
  // Normalising constant for the inverse transformation
  Float norm;

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

template <IntegralIterator I>
int GetDimension(I first, I last) {
  int dim = 1;
  for (; first != last; first++) dim *= *first;
  return dim;
}

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    DirectionFlag direction,
    PlanFlag flag) requires C2CIteratorPair<InputIt, OutputIt> {
  norm = static_cast<Float>(1) /
         static_cast<Float>(GetDimension(dimensions, dimensions + rank));
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, ComplexCast(&*out), &*onembed, ostride, odist,
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, ComplexCast(&*out), &*onembed, ostride, odist,
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, ComplexCast(&*out), &*onembed, ostride, odist,
        ConvertDirectionFlag(direction), ConvertPlanFlag(flag));
  }
}

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag) requires R2CIteratorPair<InputIt, OutputIt> {
  norm = static_cast<Float>(1) /
         static_cast<Float>(GetDimension(dimensions, dimensions + rank));
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft_r2c(
        rank, &*dimensions, howmany, &*in, &*inembed, istride, idist,
        ComplexCast(&*out), &*onembed, ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft_r2c(rank, &*dimensions, howmany, &*in, &*inembed,
                                  istride, idist, ComplexCast(&*out), &*onembed,
                                  ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft_r2c(
        rank, &*dimensions, howmany, &*in, &*inembed, istride, idist,
        ComplexCast(&*out), &*onembed, ostride, odist, ConvertPlanFlag(flag));
  }
}

template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
void Plan<InputIt, OutputIt>::MakePlan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag) requires C2RIteratorPair<InputIt, OutputIt> {
  norm = static_cast<Float>(1) /
         static_cast<Float>(GetDimension(dimensions, dimensions + rank));
  if constexpr (IsSingle<Float>) {
    plan = fftwf_plan_many_dft_c2r(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, &*out, &*onembed, ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsDouble<Float>) {
    plan = fftw_plan_many_dft_c2r(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, &*out, &*onembed, ostride, odist, ConvertPlanFlag(flag));
  }
  if constexpr (IsLongDouble<Float>) {
    plan = fftwl_plan_many_dft_c2r(
        rank, &*dimensions, howmany, ComplexCast(&*in), &*inembed, istride,
        idist, &*out, &*onembed, ostride, odist, ConvertPlanFlag(flag));
  }
}

}  // namespace FFTW

#endif  // FFTWPlan_GUARD_H
