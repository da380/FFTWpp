#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#ifndef FFTWPP_MODULE_H
#error \
    "Please include FFTWpp.h instead of including headers inside the src directory directly."
#endif

#include "fftw3.h"

namespace FFTW {

// Enum class for transformation directions.
enum class DirectionFlag { Forward, Backward };

auto ConvertDirectionFlag(DirectionFlag direction) {
  switch (direction) {
    case DirectionFlag::Forward: {
      return FFTW_FORWARD;
    }
    case DirectionFlag::Backward: {
      return FFTW_BACKWARD;
      default:
        return FFTW_FORWARD;
    }
  }
}

// Enum class for plan flags.
enum class PlanFlag { Estimate, Measure, Patient, Exhaustive };

auto ConvertPlanFlag(PlanFlag flag) {
  switch (flag) {
    case PlanFlag::Estimate: {
      return FFTW_ESTIMATE;
    }
    case PlanFlag::Measure: {
      return FFTW_MEASURE;
    }
    case PlanFlag::Patient: {
      return FFTW_PATIENT;
    }
    case PlanFlag::Exhaustive: {
      return FFTW_EXHAUSTIVE;
    }
    default:
      return FFTW_ESTIMATE;
  }
}

}  // namespace FFTW

#endif  //  FFTWPP_FLAGS_GUARD_H
