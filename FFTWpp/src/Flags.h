#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

enum class DirectionOption { Forward, Backward };

class Direction {
 public:
  Direction() : option{DirectionOption::Forward} {}
  Direction(DirectionOption option) : option{option} {}

  template <ScalarIterator InputIt, ScalarIterator OutputIt>
  auto Convert() const {
    if constexpr (R2RIteratorPair<InputIt, OutputIt>) {
      switch (option) {
        case DirectionOption::Forward: {
          return FFTW_R2HC;
        }
        case DirectionOption::Backward: {
          return FFTW_HC2R;
          default:
            return FFTW_R2HC;
        }
      }
    } else {
      switch (option) {
        case DirectionOption::Forward: {
          return FFTW_FORWARD;
        }
        case DirectionOption::Backward: {
          return FFTW_BACKWARD;
          default:
            return FFTW_FORWARD;
        }
      }
    }
  }

 private:
  DirectionOption option;
};

// Define constant instances of the Direction class for convenience.
const auto Forward = Direction(DirectionOption::Forward);
const auto Backward = Direction(DirectionOption::Backward);

// Enum class listing the basic plan flags.
enum class PlanOption { Estimate, Measure, Patient, Exhaustive, WisdomOnly };

// Define the plan flag class.
class PlanFlag {
 public:
  PlanFlag() : option{PlanOption::Estimate} {}
  PlanFlag(PlanOption option) : option{option} {}

  auto Convert() const {
    switch (option) {
      case PlanOption::Estimate: {
        return FFTW_ESTIMATE;
      }
      case PlanOption::Measure: {
        return FFTW_MEASURE;
      }
      case PlanOption::Patient: {
        return FFTW_PATIENT;
      }
      case PlanOption::Exhaustive: {
        return FFTW_EXHAUSTIVE;
      }
      case PlanOption::WisdomOnly: {
        return FFTW_WISDOM_ONLY;
      }
      default:
        return FFTW_ESTIMATE;
    }
  }

 private:
  PlanOption option;
};

// Expression template for the bitwise or operation.
template <typename PF1, typename PF2>
class PlanFlagOr {
 public:
  PlanFlagOr(const PF1& pf1, const PF2& pf2) : pf1{pf1}, pf2{pf2} {}
  auto Convert() const { return pf1.Convert() | pf2.Convert(); }

 private:
  const PF1& pf1;
  const PF2& pf2;
};

// Implement the bitwise or returning an expression.
template <typename PF1, typename PF2>
PlanFlagOr<PF1, PF2> operator|(const PF1& pf1, const PF2& pf2) {
  return {pf1, pf2};
}

// Define constant instances of the basic plan flags for convienience.
const auto Estimate = PlanFlag(PlanOption::Estimate);
const auto Measure = PlanFlag(PlanOption::Measure);
const auto Patient = PlanFlag(PlanOption::Patient);
const auto Exhaustive = PlanFlag(PlanOption::Exhaustive);
const auto WisdomOnly = PlanFlag(PlanOption::WisdomOnly);

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H
