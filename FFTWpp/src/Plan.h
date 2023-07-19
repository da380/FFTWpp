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

template <ScalarIterator InputIt, ScalarIterator OutputIt>
class Plan {
  // Store some type aliases
  using Float = IteratorPrecision<InputIt>;
  using InputValueType = IteratorValue<InputIt>;
  using OutputValueType = IteratorValue<OutputIt>;

 public:
  // Complex to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
       int istride, int idist, OutputIt out, IntIt onembed, int ostride,
       int odist, PlanFlag flag,
       DirectionFlag direction) requires C2CIteratorPair<InputIt, OutputIt>;

  // Real to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
       int istride, int idist, OutputIt out, IntIt onembed, int ostride,
       int odist, PlanFlag flag,
       DirectionFlag direction = DirectionFlag::Forward) requires
      R2CIteratorPair<InputIt, OutputIt>;

  // Real to complex constructor
  template <IntegralIterator IntIt>
  Plan(int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
       int istride, int idist, OutputIt out, IntIt onembed, int ostride,
       int odist, PlanFlag flag,
       DirectionFlag direction = DirectionFlag::Backward) requires
      C2RIteratorPair<InputIt, OutputIt>;

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

// Works out product of the dimensions for a (multi)dimensional transform.
// This value is needed when normalising the result of an inverse
// transformation.
template <IntegralIterator I>
int GetDimension(I first, I last) {
  return std::reduce(first, last, 1, std::multiplies<>());
}

// Constructor for complex-to-complex transforms.
template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
Plan<InputIt, OutputIt>::Plan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag,
    DirectionFlag direction) requires C2CIteratorPair<InputIt, OutputIt> {
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

// Constructor for real-to-complex transforms.  Note that the
// direction argument is not used; tt has a default value, and
// so can be ignored within calls.
template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
Plan<InputIt, OutputIt>::Plan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag, DirectionFlag) requires R2CIteratorPair<InputIt, OutputIt> {
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

// Constructor for complex-to-real transforms. Note that the
// direction argument is not used. It has a default value, and
// so can be ignored within calls.
template <ScalarIterator InputIt, ScalarIterator OutputIt>
template <IntegralIterator IntIt>
Plan<InputIt, OutputIt>::Plan(
    int rank, IntIt dimensions, int howmany, InputIt in, IntIt inembed,
    int istride, int idist, OutputIt out, IntIt onembed, int ostride, int odist,
    PlanFlag flag, DirectionFlag) requires C2RIteratorPair<InputIt, OutputIt> {
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
  return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
              ostride, flag, direction);
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
  return Plan(rank, it, howmany, in, it, idist, istride, out, it, odist,
              ostride, flag, direction);
}

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
