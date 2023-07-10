#ifndef FFTWPP_PLAN_GUARD_H
#define FFTWPP_PLAN_GUARD_H

#ifndef FFTWPP_MODULE_H
#error \
    "Please include FFTWpp.h instead of including headers inside the src directory directly."
#endif

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <variant>

#include "Concepts.h"
#include "Flags.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTWpp {

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

  Float GetNorm() const { return norm; }

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
  return std::reduce(first,last,1,std::multiplies<>());
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

// Wrapper function to return a plan for 1D transforms
template <ScalarIterator InputIt, ScalarIterator OutputIt>
auto Plan1D(int dimension, InputIt in, OutputIt out, PlanFlag flag,
            DirectionFlag direction = DirectionFlag::Forward) {
  int rank = 1;
  int howmany = 1;
  auto dimensions = std::vector<int>(rank, dimension);
  int idist = 1;
  int istride = 1;
  int odist = 1;
  int ostride = 1;
  auto it = dimensions.begin();
  if constexpr (C2CIteratorPair<InputIt, OutputIt>) {
    return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
                ostride, direction, flag);
  }
  if constexpr (R2CIteratorPair<InputIt, OutputIt> ||
                C2RIteratorPair<InputIt, OutputIt>) {
    return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
                ostride, flag);
  }
}

// Wrapper function to return a plan for many 1D transforms. It is
// assumed that the data are laid out in contiguous memory such that
// the ith entry in the jth transform is located at i + j * dimension.
template <ScalarIterator InputIt, ScalarIterator OutputIt>
auto Plan1DMany(int dimension, int howmany, InputIt in, OutputIt out,
                PlanFlag flag,
                DirectionFlag direction = DirectionFlag::Forward) {
  int rank = 1;
  auto dimensions = std::vector<int>(rank, dimension);
  int idist = dimension;
  int istride = 1;
  int odist = dimension;
  int ostride = 1;
  auto it = dimensions.begin();
  if constexpr (C2CIteratorPair<InputIt, OutputIt>) {
    return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
                ostride, direction, flag);
  }
  if constexpr (R2CIteratorPair<InputIt, OutputIt> ||
                C2RIteratorPair<InputIt, OutputIt>) {
    return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
                ostride, flag);
  }
}

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
