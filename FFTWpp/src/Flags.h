#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include <iostream>

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

//////////////////////////////
//      Direction flags     //
//////////////////////////////

enum class DirectionOption { Forward, Backward };

class Direction {
 public:
  Direction(DirectionOption option) : _option{option} {}

  bool operator==(const Direction& other) { return _option == other._option; }
  bool operator!=(const Direction& other) { return !(*this == other); }

  auto operator()() const {
    switch (_option) {
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

  Direction Reverse() const {
    if (_option == DirectionOption::Forward) {
      return Direction(DirectionOption::Backward);
    }
    return Direction(DirectionOption::Forward);
  }

private : DirectionOption _option;
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
  Kind(KindOption option) : _option{option} {}
  bool operator==(const Kind& other) { return _option == other._option; }
  bool operator!=(const Kind& other) { return !(*this == other); }

  auto operator()(Direction direction) {
    switch (_option) {
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
    switch (_option) {
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
  KindOption _option;
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

class PlanFlag {
  using FlagType = decltype(FFTW_ESTIMATE);

 public:
  PlanFlag(FlagType value) : _value{value} {}
  FlagType operator()() const { return _value; }
  bool operator==(const PlanFlag& other) { return _value == other._value; }
  bool operator!=(const PlanFlag& other) { return !(*this == other); }

 private:
  FlagType _value;
};

PlanFlag operator|(const PlanFlag& pf1, const PlanFlag& pf2) {
  return PlanFlag(pf1() | pf2());
}

// Define constant instances of the basic plan flags for convienience.
const auto Estimate = PlanFlag(FFTW_ESTIMATE);
const auto Measure = PlanFlag(FFTW_MEASURE);
const auto Patient = PlanFlag(FFTW_PATIENT);
const auto Exhaustive = PlanFlag(FFTW_EXHAUSTIVE);
const auto WisdomOnly = PlanFlag(FFTW_WISDOM_ONLY);
const auto DestroyInput = PlanFlag(FFTW_DESTROY_INPUT);
const auto PreserveInput = PlanFlag(FFTW_PRESERVE_INPUT);
const auto Unaligned = PlanFlag(FFTW_UNALIGNED);

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H
