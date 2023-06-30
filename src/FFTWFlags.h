#ifndef FFTWFlags_GUARD_H
#define FFTWFlags_GUARD_H

#include "fftw3.h"

namespace FFTW {

// Enum class for transformation directions.
enum class DirectionFlag { Forward, Backward };

template <bool R2R = false>
auto ConvertDirectionFlag(DirectionFlag direction) {
  if constexpr (!R2R) {
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

  if constexpr (R2R) {
    switch (direction) {
      case DirectionFlag::Forward: {
        return FFTW_R2HC;
      }
      case DirectionFlag::Backward: {
        return FFTW_HC2R;
        default:
          return FFTW_R2HC;
      }
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

#endif  //  FFTWFlags_GUARD_H
