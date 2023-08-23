#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

enum class NormalisationOption { No, Yes };

const auto Normalise = NormalisationOption::Yes;

enum class DirectionOption { Forward, Backward };

class Direction {
 public:
  Direction() : option{DirectionOption::Forward} {}
  Direction(DirectionOption option) : option{option} {}

  bool operator==(const Direction& other) { return option == other.option; };

  template <bool R2R = false>
  auto operator()() const {
    if constexpr (R2R) {
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
enum class PlanOption {
  Estimate,
  Measure,
  Patient,
  Exhaustive,
  WisdomOnly,
  DestroyInput,
  PreserveInput,
  Unaligned
};

// Define the plan flag class.
class PlanFlag {
 public:
  PlanFlag() : option{PlanOption::Estimate} {}
  PlanFlag(PlanOption option) : option{option} {}

  auto operator()() const {
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
      case PlanOption::DestroyInput: {
        return FFTW_DESTROY_INPUT;
      }
      case PlanOption::PreserveInput: {
        return FFTW_PRESERVE_INPUT;
      }
      case PlanOption::Unaligned: {
        return FFTW_UNALIGNED;
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
  auto operator()() const { return pf1() | pf2(); }

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
const auto DestroyInput = PlanFlag(PlanOption::DestroyInput);
const auto PreserveInput = PlanFlag(PlanOption::PreserveInput);
const auto Unaligned = PlanFlag(PlanOption::Unaligned);

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H
