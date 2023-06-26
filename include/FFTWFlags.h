#ifndef FFTWFlags_GUARD_H
#define FFTWFlags_GUARD_H

namespace FFTW {

// Enum class for transformation directions.
enum class FFTWDirectionFlag { Forward, Backward };

auto ConvertDirectionFlag(FFTWDirectionFlag direction) {
  if (direction == FFTWDirectionFlag::Forward) {
    return FFTW_FORWARD;
  } else {
    return FFTW_BACKWARD;
  }
}

// Enum class for plan flags.
enum class FFTWPlanFlag { Estimate, Measure, Patient, Exhaustive };

auto ConvertPlanFlag(FFTWPlanFlag flag) {
  switch (flag) {
    case FFTWPlanFlag::Estimate: {
      return FFTW_ESTIMATE;
    }
    case FFTWPlanFlag::Measure: {
      return FFTW_MEASURE;
    }
    case FFTWPlanFlag::Patient: {
      return FFTW_PATIENT;
    }
    case FFTWPlanFlag::Exhaustive: {
      return FFTW_EXHAUSTIVE;
    }
    default:
      return FFTW_ESTIMATE;
  }
}

}  // namespace FFTW

#endif  //  FFTWFlags_GUARD_H
