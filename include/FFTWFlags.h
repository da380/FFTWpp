#ifndef FFTWFlags_GUARD_H
#define FFTWFlags_GUARD_H

#include "fftw3.h"

namespace FFTW {

// Enum class for transformation directions.
enum class DirectionFlag { Forward, Backward };

auto ConvertDirectionFlag(DirectionFlag direction) {
  if (direction == DirectionFlag::Forward) {
    return FFTW_FORWARD;
  } else {
    return FFTW_BACKWARD;
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

#endif  //  FFTWFlags_GUARD_H
