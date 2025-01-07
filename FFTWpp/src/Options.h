#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "fftw3.h"

namespace FFTWpp {

class Direction {
 public:
  constexpr Direction() = default;

  constexpr Direction(int direction) : _direction{direction} {
    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
  }

  constexpr operator int() const { return _direction; }

  bool operator==(const Direction&) const = default;

 private:
  int _direction;
};

constexpr auto Forward = Direction{FFTW_FORWARD};

constexpr auto Backward = Direction{FFTW_BACKWARD};

class Flag {
 public:
  constexpr Flag() = default;

  constexpr Flag(unsigned flag) : _flag{flag} {}

  constexpr operator unsigned() const { return _flag; }

  bool operator==(const Flag&) const = default;

 private:
  unsigned _flag;
};

constexpr auto operator|(Flag&& lhs, Flag&& rhs) {
  return Flag{static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)};
}

constexpr auto Estimate = Flag{FFTW_ESTIMATE};

constexpr auto Measure = Flag{FFTW_MEASURE};

constexpr auto Patient = Flag{FFTW_PATIENT};

constexpr auto Exhaustive = Flag{FFTW_EXHAUSTIVE};

constexpr auto WisdomOnly = Flag{FFTW_WISDOM_ONLY};

constexpr auto DestroyInput = Flag{FFTW_DESTROY_INPUT};

constexpr auto PreserveInput = Flag{FFTW_PRESERVE_INPUT};

constexpr auto Unaligned = Flag{FFTW_UNALIGNED};

class RealKind {
 public:
  constexpr RealKind() = default;

  constexpr RealKind(fftw_r2r_kind kind) : _kind{kind} {}

  constexpr operator fftw_r2r_kind() const { return _kind; }

  bool operator==(const RealKind&) const = default;

  constexpr auto Inverse() const {
    switch (_kind) {
      case FFTW_R2HC:
        return RealKind{FFTW_HC2R};
      case FFTW_HC2R:
        return RealKind{FFTW_R2HC};
      case FFTW_DHT:
        return RealKind{FFTW_DHT};
      case FFTW_REDFT00:
        return RealKind{FFTW_REDFT00};
      case FFTW_REDFT10:
        return RealKind{FFTW_REDFT01};
      case FFTW_REDFT01:
        return RealKind{FFTW_REDFT10};
      case FFTW_REDFT11:
        return RealKind{FFTW_REDFT11};
      case FFTW_RODFT00:
        return RealKind{FFTW_RODFT00};
      case FFTW_RODFT10:
        return RealKind{FFTW_RODFT01};
      case FFTW_RODFT01:
        return RealKind{FFTW_RODFT10};
      case FFTW_RODFT11:
        return RealKind{FFTW_RODFT11};
      default:
        return RealKind{FFTW_HC2R};
    }
  }

  constexpr auto LogicalDimension(int n) const {
    switch (_kind) {
      case FFTW_R2HC:
        return n;
      case FFTW_HC2R:
        return n;
      case FFTW_DHT:
        return n;
      case FFTW_REDFT00:
        return 2 * (n - 1);
      case FFTW_REDFT10:
        return 2 * n;
      case FFTW_REDFT01:
        return 2 * n;
      case FFTW_REDFT11:
        return 2 * n;
      case FFTW_RODFT00:
        return 2 * (n + 1);
      case FFTW_RODFT10:
        return 2 * n;
      case FFTW_RODFT01:
        return 2 * n;
      case FFTW_RODFT11:
        return 2 * n;
      default:
        return n;
    }
  }

 private:
  fftw_r2r_kind _kind;
};

constexpr auto R2HC = RealKind{FFTW_R2HC};

constexpr auto HC2R = RealKind{FFTW_HC2R};

constexpr auto DHT = RealKind{FFTW_DHT};

constexpr auto REDFT00 = RealKind{FFTW_REDFT00};

constexpr auto REDFT10 = RealKind{FFTW_REDFT10};

constexpr auto REDFT01 = RealKind{FFTW_REDFT01};

constexpr auto REDFT11 = RealKind{FFTW_REDFT11};

constexpr auto RODFT00 = RealKind{FFTW_RODFT00};

constexpr auto RODFT10 = RealKind{FFTW_RODFT10};

constexpr auto RODFT01 = RealKind{FFTW_RODFT01};

constexpr auto RODFT11 = RealKind{FFTW_RODFT11};

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H
