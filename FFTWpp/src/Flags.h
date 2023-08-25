#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

/////////////////////////////////
//      Normalisation flags    //
/////////////////////////////////

enum class NormalisationOption { Normalised, UnNormalised };
const auto Normalised = NormalisationOption::Normalised;
const auto UnNormalised = NormalisationOption::UnNormalised;

//////////////////////////////
//      Direction flags     //
//////////////////////////////

enum class DirectionOption { Forward, Backward };

class Direction {
 public:
  Direction() : option{DirectionOption::Forward} {}

  Direction(DirectionOption option) : option{option} {}

  bool operator==(const Direction& other) { return option == other.option; };

  auto operator()() const {
    switch (option) {
      case DirectionOption::Forward: {
        return FFTW_FORWARD;
      }
      case DirectionOption::Backward: {
        return FFTW_BACKWARD;
      }
      default:
        return FFTW_FORWARD;
    }
  }

 private:
  DirectionOption option;
};

const auto Forward = Direction(DirectionOption::Forward);
const auto Backward = Direction(DirectionOption::Backward);

//////////////////////////////////
//       Real-real flags        //
//////////////////////////////////

enum class R2ROption {
  NotR2R,
  HC,
  DHT,
  DCTI,
  DCTII,
  DCTIII,
  DCTIV,
  DSTI,
  DSTII,
  DSTIII,
  DSTIV
};

class R2R {
 public:
  R2R() : option{R2ROption::NotR2R} {}
  R2R(R2ROption option) : option{option} {}

  bool operator==(const R2R& other) { return option == other.option; };

  auto operator()(Direction direction) {
    switch (option) {
      case R2ROption::NotR2R: {
        return FFTW_R2HC;
      }

      case R2ROption::HC: {
        if (direction == Forward) {
          return FFTW_R2HC;
        } else {
          return FFTW_HC2R;
        }
      }

      case R2ROption::DHT: {
        if (direction == Forward) {
          return FFTW_DHT;
        } else {
          return FFTW_DHT;
        }
      }

      case R2ROption::DCTI: {
        if (direction == Forward) {
          return FFTW_REDFT00;
        } else {
          return FFTW_REDFT00;
        }
      }

      case R2ROption::DCTII: {
        if (direction == Forward) {
          return FFTW_REDFT10;
        } else {
          return FFTW_REDFT01;
        }
      }

      case R2ROption::DCTIII: {
        if (direction == Forward) {
          return FFTW_REDFT01;
        } else {
          return FFTW_REDFT10;
        }
      }

      case R2ROption::DCTIV: {
        if (direction == Forward) {
          return FFTW_REDFT11;
        } else {
          return FFTW_REDFT11;
        }
      }

      case R2ROption::DSTI: {
        if (direction == Forward) {
          return FFTW_RODFT00;
        } else {
          return FFTW_RODFT00;
        }
      }

      case R2ROption::DSTII: {
        if (direction == Forward) {
          return FFTW_RODFT10;
        } else {
          return FFTW_RODFT01;
        }
      }

      case R2ROption::DSTIII: {
        if (direction == Forward) {
          return FFTW_RODFT01;
        } else {
          return FFTW_RODFT10;
        }
      }

      case R2ROption::DSTIV: {
        if (direction == Forward) {
          return FFTW_RODFT11;
        } else {
          return FFTW_RODFT11;
        }
      }

      default:
        return FFTW_R2HC;
    }
  }

 private:
  R2ROption option;
};

const auto NotR2R = R2R(R2ROption::NotR2R);
const auto HC = R2R(R2ROption::HC);
const auto DHT = R2R(R2ROption::DHT);
const auto DCTI = R2R(R2ROption::DCTI);
const auto DCTII = R2R(R2ROption::DCTII);
const auto DCTIII = R2R(R2ROption::DCTIII);
const auto DCTIV = R2R(R2ROption::DCTIV);
const auto DSTI = R2R(R2ROption::DSTI);
const auto DSTII = R2R(R2ROption::DSTII);
const auto DSTIII = R2R(R2ROption::DSTIII);
const auto DSTIV = R2R(R2ROption::DSTIV);

///////////////////////////////////
//         Planning flags        //
///////////////////////////////////

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

class PlanFlag {
 public:
  using plan_flag_t = int;

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

template <typename PF1, typename PF2>
requires requires() {
  typename PF1::plan_flag_t;
  typename PF2::plan_flag_t;
}
class PlanFlagOr {
 public:
  using plan_flag_t = int;
  PlanFlagOr(const PF1& pf1, const PF2& pf2) : pf1{pf1}, pf2{pf2} {}
  auto operator()() const { return pf1() | pf2(); }

 private:
  const PF1& pf1;
  const PF2& pf2;
};

template <typename PF1, typename PF2>
requires requires() {
  typename PF1::plan_flag_t;
  typename PF2::plan_flag_t;
}
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
