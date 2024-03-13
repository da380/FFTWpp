#ifndef FFTWPP_FLAGS_GUARD_H
#define FFTWPP_FLAGS_GUARD_H

#include "fftw3.h"

namespace FFTWpp {

/** @brief Stores directions for complex-to-complex
 transformations.

 This class provides a type-safe wrapper for working with the
 `int`  constants `FFTW_FORWARD` and `FFTW_BACKWARD`.
 */
class Direction {
 public:
  /**
   *
   * @param direction Must be equal to either `FFTW_FORWARD` or
   * `FFTW_BACKWARD`.
   *
   */
  constexpr Direction(int direction) : _direction{direction} {
    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
  }

  /**
   * @brief Cast Direction to int.
   *
   */
  constexpr operator int() const { return _direction; }

  /**
   * @brief Equality operator.
   */
  bool operator==(const Direction&) const = default;

 private:
  int _direction;
};

/** \var Forward
    \brief FFTWpp::Direction equivalent to `FFTW_FORWARD`.
*/
constexpr auto Forward = Direction{FFTW_FORWARD};

/** \var Backward
    \brief FFTWpp::Direction equivalent to `FFTW_BACKWARD`.
*/
constexpr auto Backward = Direction{FFTW_BACKWARD};

/** @brief Stores planning flags for transformations.
 *
 * This class provides a type-safe wrapper for working with the `unsigned`
 *  planner flags within `fftw3`.
 *
 */
class Flag {
 public:
  /**
   * @param flag `fftw3` planner flag or combination thereof using bitwise or.
   *
   */
  constexpr Flag(unsigned flag) : _flag{flag} {}

  /**
   * @brief Cast Flag to unsigned.
   *
   */
  constexpr operator unsigned() const { return _flag; }

  /**
   * @brief Equality operator.
   */
  bool operator==(const Flag&) const = default;

 private:
  unsigned _flag;
};

/**
 * @brief Overload of "bitwise or" used to combine FFTWpp::Flag instances
 following the `fftw3` conventions.
 *
 *
 * @return FFTWpp::Flag storing the combined options.
 *

 */
constexpr auto operator|(Flag&& lhs, Flag&& rhs) {
  return Flag{static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)};
}

/**
 * \var Estimate
 * \brief FFTWpp::Flag equivalent to `FFTW_ESTIMATE`.
 *
 * Specifies that, instead of actual measurements of different algorithms, a
 * simple heuristic is used to pick a (probably sub-optimal) plan quickly. With
 * this flag, the input/output arrays are not overwritten during planning.
 */
constexpr auto Estimate = Flag{FFTW_ESTIMATE};

/**
 * \var Measure
 * \brief FFTWpp::Flag equivalent to `FFTW_MEASURE`.
 *
 * Tells `fftw3` to find an optimized plan by actually computing several FFTs
 * and measuring their execution time. Depending on your machine, this can take
 * some time (often a few seconds).
 */
constexpr auto Measure = Flag{FFTW_MEASURE};

/**
 * \var Patient
 * \brief FFTWpp::Flag equivalent to `FFTW_PATIENT`.
 *
 * Is like FFTW_MEASURE, but considers a wider range of algorithms and often
 * produces a “more optimal” plan (especially for large transforms), but at the
 * expense of several times longer planning time (especially for large
 * transforms).
 */
constexpr auto Patient = Flag{FFTW_PATIENT};

/**
 * \var Exhaustive
 * \brief FFTWpp::Flag equivalent to `FFTW_EXHAUSTIVE`.
 *
 *   Is like FFTW_PATIENT, but considers an even wider range of algorithms,
 * including many that we think are unlikely to be fast, to produce the most
 * optimal plan but with a substantially increased planning time
 *
 */
constexpr auto Exhaustive = Flag{FFTW_EXHAUSTIVE};

/**
 * \var WisdomOnly
 * \brief FFTWpp::Flag equivalent to `FFTW_WISDOM_ONLY`.
 *
 *  A special planning mode in which the plan is only created if wisdom is
 * available for the given problem. This
 * can be combined with other flags, for example,
 *
 *     FFTWpp::WisdomOnly | FFTWpp::Patient
 *
 * creates a plan only if wisdom is available that was created in
 * FFTWpp::Patient or FFTWpp::Exhaustive mode. The FFTWpp::WisdomOnly flag is
 * intended for users who need to detect whether wisdom is available; for
 * example, if wisdom is not available one may wish to allocate new arrays for
 * planning so that user data is not overwritten.
 *
 */
constexpr auto WisdomOnly = Flag{FFTW_WISDOM_ONLY};

/**
 * \var DestroyInput
 * \brief FFTWpp::Flag equivalent to `FFTW_DESTROY_INPUT`.
 *
 * Specifies that an out-of-place transform is allowed to overwrite its input
 * array with arbitrary data; this can sometimes allow more efficient algorithms
 * to be employed.
 */
constexpr auto DestroyInput = Flag{FFTW_DESTROY_INPUT};

/**
 * \var PreserveInput
 * \brief FFTWpp::Flag equivalent to `FFTW_PRESERVE_INPUT`.
 *
 * Specifies that an out-of-place transform must not change its input array.
 * This is ordinarily the default, except for complex-to-complex and
 * half-complex-to-real transforms for which
 * FFTWpp::DestroyInput is the default. In the latter cases, passing
 * FFTW::PreserveInput will attempt to use algorithms that do not destroy the
 * input, at the expense of worse performance; for multi-dimensional
 * complex-to-real transforms, however, no input-preserving algorithms are
 * implemented and a null plan will be created if this is requested.
 */
constexpr auto PreserveInput = Flag{FFTW_PRESERVE_INPUT};

/**
 * \var Unaligned
 * \brief FFTWpp::Flag equivalent to `FFTW_UNALIGNED`.
 *
 * Specifies that the algorithm may not impose any unusual alignment
 * requirements on the input/output arrays (i.e. no SIMD may be used). This flag
 * is normally not necessary, since the planner automatically detects misaligned
 * arrays. The only use for this flag is if you want to use the new-array
 * execute interface to execute a given plan on a different array that may not
 * be aligned like the original. Using FFTWpp::allocator insures that the
 * correct alignement is applied.
 *
 */
constexpr auto Unaligned = Flag{FFTW_UNALIGNED};

/** @brief Stores  kind options for real to real transformation.
 *
 * This class provides a type-safe wrapper for working with the `fftw_r2r_kind`
 *  kinds within `fftw3`.
 *
 */
class RealKind {
 public:
  /**
   *
   * @param kind `fftw3` real kind option.
   *
   */
  constexpr RealKind(fftw_r2r_kind kind) : _kind{kind} {}

  /**
   * @brief Cast RealKind to fftw2_r2r_kind.
   *
   * @return fftw_r2r_kind
   */
  constexpr operator fftw_r2r_kind() const { return _kind; }

  /**
   * @brief Equality operator.
   *
   */
  bool operator==(const RealKind&) const = default;

  /**
   * @brief Returns a RealKind for the inverse transformation.
   *
   */
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

  /**
   * @brief Returns the logical dimension for a given data size.
   *
   * @param n data size.
   * @return `int` logical dimension.
   */
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

/**
 * @var R2HC
 *
 * @brief Real to half complex DFT.
 *
 */
constexpr auto R2HC = RealKind{FFTW_R2HC};

/**
 * @var HC2R
 *
 * @brief Half complex to real DFT.
 *
 */
constexpr auto HC2R = RealKind{FFTW_HC2R};

/**
 * @var DHT
 *
 * @brief Discrete Hartley transformation.
 *
 */
constexpr auto DHT = RealKind{FFTW_DHT};

/**
 * @var REDFT00
 * @brief Discrete cosine transformation I.
 *
 */
constexpr auto REDFT00 = RealKind{FFTW_REDFT00};

/**
 * @var REDFT10
 * @brief  Discrete cosine transformation II.
 */
constexpr auto REDFT10 = RealKind{FFTW_REDFT10};

/**
 * @var REDFT01
 * @brief Discrete cosine transformation III.
 *
 */
constexpr auto REDFT01 = RealKind{FFTW_REDFT01};

/**
 * @var REDFT11
 * @brief Discrete cosine transformation IV.
 *
 */
constexpr auto REDFT11 = RealKind{FFTW_REDFT11};

/**
 * @var RODFT00
 * @brief Discrete sine transformation I.
 *
 */
constexpr auto RODFT00 = RealKind{FFTW_RODFT00};

/**
 * @var RODTF10
 * @brief Discrete sine transformation II.
 *
 */
constexpr auto RODFT10 = RealKind{FFTW_RODFT10};

/**
 * @var RODFT01
 * @brief Discrete sine transformation III.
 *
 */
constexpr auto RODFT01 = RealKind{FFTW_RODFT01};

/**
 * @var RODFT11
 * @brief Discrete sine transformation IV.
 *
 */
constexpr auto RODFT11 = RealKind{FFTW_RODFT11};

}  // namespace FFTWpp

#endif  //  FFTWPP_FLAGS_GUARD_H
