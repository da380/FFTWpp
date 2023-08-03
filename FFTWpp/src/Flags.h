#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "fftw3.h"

namespace FFTWpp {

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

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H