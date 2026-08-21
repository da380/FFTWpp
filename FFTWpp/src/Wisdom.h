/**
 * @file Wisdom.h
 * @brief Provides utility functions for managing FFTW wisdom.
 *
 * Wisdom in FFTW is a way to save and reuse information about how to compute
 * transforms of a given size and type optimally. This can significantly speed
 * up the creation of new plans. This file provides functions to import, export,
 * forget, and pre-generate wisdom for various transform types.
 */
#ifndef FFTWPP_WISDOM_GUARD_H
#define FFTWPP_WISDOM_GUARD_H

#include <algorithm>
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "Plan.h"
#include "Views.h"
#include "fftw3.h"

namespace FFTWpp {

namespace Internal {

/** @brief The human-readable name of a precision, for diagnostics. */
template <NumericConcepts::Real Real>
constexpr const char* PrecisionName() {
  if constexpr (NumericConcepts::Float<Real>) return "single";
  if constexpr (NumericConcepts::Double<Real>) return "double";
  if constexpr (NumericConcepts::LongDouble<Real>) return "long double";
}

}  // namespace Internal

/**
 * @brief Exports accumulated wisdom of a given precision to a file.
 * @details FFTW maintains a separate wisdom store, and a separate file format,
 * for each precision. Wisdom written for one precision cannot be read back for
 * another, so an application using more than one precision needs one file per
 * precision.
 * @tparam Real The precision whose wisdom store is to be written.
 * @param filename The path to the file where wisdom will be saved.
 * @throws std::runtime_error if the file cannot be written.
 */
template <NumericConcepts::Real Real>
void ExportWisdom(const std::string& filename) {
  const auto lock = PlannerLock{};
  int io = 0;
  if constexpr (NumericConcepts::Float<Real>) {
    io = fftwf_export_wisdom_to_filename(filename.c_str());
  } else if constexpr (NumericConcepts::Double<Real>) {
    io = fftw_export_wisdom_to_filename(filename.c_str());
  } else {
    io = fftwl_export_wisdom_to_filename(filename.c_str());
  }
  if (io == 0) {
    throw std::runtime_error(std::string("failed to export ") +
                             Internal::PrecisionName<Real>() +
                             "-precision FFTW wisdom to " + filename);
  }
}

/**
 * @brief Imports wisdom of a given precision from a file.
 * @tparam Real The precision whose wisdom store is to be populated.
 * @param filename The path to the file from which to load wisdom.
 * @throws std::runtime_error if the file cannot be read, which includes the
 * case of a file written for a different precision.
 */
template <NumericConcepts::Real Real>
void ImportWisdom(const std::string& filename) {
  const auto lock = PlannerLock{};
  int io = 0;
  if constexpr (NumericConcepts::Float<Real>) {
    io = fftwf_import_wisdom_from_filename(filename.c_str());
  } else if constexpr (NumericConcepts::Double<Real>) {
    io = fftw_import_wisdom_from_filename(filename.c_str());
  } else {
    io = fftwl_import_wisdom_from_filename(filename.c_str());
  }
  if (io == 0) {
    throw std::runtime_error(std::string("failed to import ") +
                             Internal::PrecisionName<Real>() +
                             "-precision FFTW wisdom from " + filename);
  }
}

/**
 * @brief Exports accumulated double-precision wisdom to a file.
 * @details Equivalent to `ExportWisdom<double>(filename)`.
 * @param filename The path to the file where wisdom will be saved.
 * @throws std::runtime_error if the file cannot be written.
 */
inline void ExportWisdom(const std::string& filename) {
  ExportWisdom<double>(filename);
}

/**
 * @brief Imports double-precision wisdom from a file.
 * @details Equivalent to `ImportWisdom<double>(filename)`.
 * @param filename The path to the file from which to load wisdom.
 * @throws std::runtime_error if the file cannot be read.
 */
inline void ImportWisdom(const std::string& filename) {
  ImportWisdom<double>(filename);
}

/**
 * @brief Imports the system-wide wisdom file for a given precision.
 * @details Wraps `fftw*_import_system_wisdom`, which reads the wisdom file an
 * administrator may have generated for this machine (`/etc/fftw/wisdom` and
 * its per-precision siblings on a typical installation).
 * @tparam Real The precision whose wisdom store is to be populated.
 * @return `true` if system wisdom was found and read, `false` otherwise. A
 * missing system wisdom file is a normal condition, not an error, so this
 * reports rather than throws.
 */
template <NumericConcepts::Real Real>
[[nodiscard]] bool ImportSystemWisdom() {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_import_system_wisdom() != 0;
  } else if constexpr (NumericConcepts::Double<Real>) {
    return fftw_import_system_wisdom() != 0;
  } else {
    return fftwl_import_system_wisdom() != 0;
  }
}

/**
 * @brief Exports accumulated wisdom of a given precision as a string.
 * @details Useful when wisdom is to be stored somewhere other than a file: a
 * database, a configuration blob, or a message to another process.
 * @tparam Real The precision whose wisdom store is to be written.
 * @return The serialised wisdom.
 * @throws std::runtime_error if FFTW cannot serialise its wisdom.
 */
template <NumericConcepts::Real Real>
[[nodiscard]] std::string ExportWisdomToString() {
  const auto lock = PlannerLock{};
  char* raw = nullptr;
  if constexpr (NumericConcepts::Float<Real>) {
    raw = fftwf_export_wisdom_to_string();
  } else if constexpr (NumericConcepts::Double<Real>) {
    raw = fftw_export_wisdom_to_string();
  } else {
    raw = fftwl_export_wisdom_to_string();
  }
  if (raw == nullptr) {
    throw std::runtime_error(std::string("failed to export ") +
                             Internal::PrecisionName<Real>() +
                             "-precision FFTW wisdom to a string");
  }
  auto wisdom = std::string(raw);
  fftw_free(raw);
  return wisdom;
}

/**
 * @brief Imports wisdom of a given precision from a string.
 * @tparam Real The precision whose wisdom store is to be populated.
 * @param wisdom Serialised wisdom, as produced by `ExportWisdomToString`.
 * @throws std::runtime_error if the string cannot be parsed as wisdom of this
 * precision.
 */
template <NumericConcepts::Real Real>
void ImportWisdomFromString(const std::string& wisdom) {
  const auto lock = PlannerLock{};
  int io = 0;
  if constexpr (NumericConcepts::Float<Real>) {
    io = fftwf_import_wisdom_from_string(wisdom.c_str());
  } else if constexpr (NumericConcepts::Double<Real>) {
    io = fftw_import_wisdom_from_string(wisdom.c_str());
  } else {
    io = fftwl_import_wisdom_from_string(wisdom.c_str());
  }
  if (io == 0) {
    throw std::runtime_error(std::string("failed to import ") +
                             Internal::PrecisionName<Real>() +
                             "-precision FFTW wisdom from a string");
  }
}

/**
 * @brief Forgets accumulated wisdom for float, double, and long double.
 */
inline void ForgetWisdom() {
  const auto lock = PlannerLock{};
  fftwf_forget_wisdom();
  fftw_forget_wisdom();
  fftwl_forget_wisdom();
}

/**
 * @brief Forgets accumulated wisdom for a single precision.
 * @tparam Real The precision whose wisdom store is to be cleared.
 */
template <NumericConcepts::Real Real>
void ForgetWisdom() {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    fftwf_forget_wisdom();
  } else if constexpr (NumericConcepts::Double<Real>) {
    fftw_forget_wisdom();
  } else {
    fftwl_forget_wisdom();
  }
}

/**
 * @brief Generates wisdom for complex-to-complex, real-to-complex, or
 * complex-to-real transforms.
 * @details This function creates temporary data buffers and then creates (but
 * does not execute) forward and backward plans for the specified layouts. This
 * process populates the internal FFTW wisdom cache for the given transform
 * size, type, and flags. No wisdom is generated if the flag is `Estimate`.
 * @tparam InType The value type of the input data (e.g.,
 * `std::complex<double>`).
 * @tparam OutType The value type of the output data (e.g., `double`).
 * @param inLayout The layout (shape) of the input data.
 * @param outLayout The layout (shape) of the output data.
 * @param flag The planner flag (`Measure`, `Patient`, etc.) to use for wisdom
 * generation.
 * @requires The precision of InType and OutType must be the same.
 */
template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  if constexpr (NumericConcepts::Complex<InType> &&
                NumericConcepts::Complex<OutType>) {
    [[maybe_unused]] auto planForward =
        Ranges::Plan(inView, outView, flag, Forward);
    [[maybe_unused]] auto planBackward =
        Ranges::Plan(outView, inView, flag, Backward);
  }
  if constexpr ((NumericConcepts::Real<InType> &&
                 NumericConcepts::Complex<OutType>) ||
                (NumericConcepts::Complex<InType> &&
                 NumericConcepts::Real<OutType>)) {
    [[maybe_unused]] auto planForward = Ranges::Plan(inView, outView, flag);
    [[maybe_unused]] auto planBackward = Ranges::Plan(outView, inView, flag);
  }
}

/**
 * @brief Generates wisdom for real-to-real transforms.
 * @details This function populates the FFTW wisdom cache for R2R transforms of
 * a given layout and kind. It creates plans for both the forward and backward
 * transforms.
 * @tparam InType Real input value type.
 * @tparam OutType Real output value type.
 * @param inLayout The layout (shape) of the input data.
 * @param outLayout The layout (shape) of the output data.
 * @param kinds An initializer list of `RealKind` for the forward transform.
 * @param flag The planner flag (`Measure`, `Patient`, etc.) to use.
 * @requires InType and OutType must be real types with the same precision.
 */
template <NumericConcepts::Real InType, NumericConcepts::Real OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    std::initializer_list<RealKind> kinds, Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  auto kindsForward = std::vector<RealKind>(kinds);
  auto kindsBackward = std::vector<RealKind>();
  kindsBackward.reserve(kindsForward.size());
  std::ranges::transform(kindsForward, std::back_inserter(kindsBackward),
                         [](auto kind) { return kind.Inverse(); });
  [[maybe_unused]] auto planForward =
      Ranges::Plan(inView, outView, flag, kindsForward);
  [[maybe_unused]] auto planBackward =
      Ranges::Plan(outView, inView, flag, kindsBackward);
}

}  // namespace FFTWpp

#endif  // FFTWPP_WISDOM_GUARD_H
