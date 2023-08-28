#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

//////////////////////////////
//      Direction flags     //
//////////////////////////////

enum class DirectionOption { Forward, Backward };

class Direction {
 public:
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

enum class KindOption {
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

class Kind {
 public:
  Kind(KindOption option) : option{option} {}
  bool operator==(const Kind& other) { return option == other.option; };

  auto operator()(Direction direction) {
    switch (option) {
      case KindOption::HC: {
        if (direction == Forward) {
          return FFTW_R2HC;
        } else {
          return FFTW_HC2R;
        }
      }

      case KindOption::DHT: {
        if (direction == Forward) {
          return FFTW_DHT;
        } else {
          return FFTW_DHT;
        }
      }

      case KindOption::DCTI: {
        if (direction == Forward) {
          return FFTW_REDFT00;
        } else {
          return FFTW_REDFT00;
        }
      }

      case KindOption::DCTII: {
        if (direction == Forward) {
          return FFTW_REDFT10;
        } else {
          return FFTW_REDFT01;
        }
      }

      case KindOption::DCTIII: {
        if (direction == Forward) {
          return FFTW_REDFT01;
        } else {
          return FFTW_REDFT10;
        }
      }

      case KindOption::DCTIV: {
        if (direction == Forward) {
          return FFTW_REDFT11;
        } else {
          return FFTW_REDFT11;
        }
      }

      case KindOption::DSTI: {
        if (direction == Forward) {
          return FFTW_RODFT00;
        } else {
          return FFTW_RODFT00;
        }
      }

      case KindOption::DSTII: {
        if (direction == Forward) {
          return FFTW_RODFT10;
        } else {
          return FFTW_RODFT01;
        }
      }

      case KindOption::DSTIII: {
        if (direction == Forward) {
          return FFTW_RODFT01;
        } else {
          return FFTW_RODFT10;
        }
      }

      case KindOption::DSTIV: {
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

  auto LogicalSize(int n) {
    switch (option) {
      case KindOption::DCTI: {
        return 2 * (n - 1);
      }

      case KindOption::DCTII: {
        return 2 * n;
      }

      case KindOption::DCTIII: {
        return 2 * n;
      }

      case KindOption::DCTIV: {
        return 2 * n;
      }

      case KindOption::DSTI: {
        return 2 * (n + 1);
      }

      case KindOption::DSTII: {
        return 2 * n;
      }

      case KindOption::DSTIII: {
        return 2 * n;
      }

      case KindOption::DSTIV: {
        return 2 * n;
      }

      default:
        return n;
    }
  }

 private:
  KindOption option;
};

const auto HC = Kind(KindOption::HC);
const auto DHT = Kind(KindOption::DHT);
const auto DCTI = Kind(KindOption::DCTI);
const auto DCTII = Kind(KindOption::DCTII);
const auto DCTIII = Kind(KindOption::DCTIII);
const auto DCTIV = Kind(KindOption::DCTIV);
const auto DSTI = Kind(KindOption::DSTI);
const auto DSTII = Kind(KindOption::DSTII);
const auto DSTIII = Kind(KindOption::DSTIII);
const auto DSTIV = Kind(KindOption::DSTIV);

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
