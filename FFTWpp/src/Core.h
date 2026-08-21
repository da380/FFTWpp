/**
 * @file Core.h
 * @brief Core header for the FFTWpp library, providing C++ wrappers for FFTW3
 * functions.
 *
 * This file defines type-safe and precision-aware C++ wrappers for the core
 * functionality of the FFTW3 library. It includes:
 * - A custom STL allocator for memory alignment required by FFTW.
 * - Template-based plan creation functions for various transform types (DFT,
 * R2C, C2R, R2R) across 1D, 2D, 3D, and N-dimensional data.
 * - Overloaded functions that automatically select the correct FFTW precision
 * (float, double, long double) based on the data type.
 * - Wrapper functions for executing and destroying plans.
 */
#ifndef FFTWPP_CORE_GUARD_H
#define FFTWPP_CORE_GUARD_H

#include <atomic>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "NumericConcepts/Numeric.hpp"
#include "fftw3.h"

/**
 * @brief Main namespace for the FFTWpp C++ wrapper library.
 */
namespace FFTWpp {

//------------------------------------------------------//
//              Define some useful concepts             //
//------------------------------------------------------//

/**
 * @brief Concept to check if a type is a valid FFTW plan handle.
 * @tparam T The type to check.
 */
template <typename T>
concept IsPlan = std::same_as<T, fftwf_plan> or std::same_as<T, fftw_plan> or
                 std::same_as<T, fftwl_plan>;

/**
 * @brief Concept to ensure that the precision of a plan matches the precision
 * of the real data type.
 * @tparam PlanType The FFTW plan type (e.g., fftwf_plan).
 * @tparam Real The real data type (e.g., float, double).
 */
template <typename PlanType, typename Real>
concept CheckPrecision =
    (std::same_as<PlanType, fftwf_plan> and NumericConcepts::Float<Real>) or
    (std::same_as<PlanType, fftw_plan> and NumericConcepts::Double<Real>) or
    (std::same_as<PlanType, fftwl_plan> and NumericConcepts::LongDouble<Real>);

//--------------------------------------------------------------//
//                    Planner serialisation                     //
//--------------------------------------------------------------//

/**
 * @brief Returns the process-wide mutex that serialises calls into the FFTW
 * planner.
 * @details FFTW's planner is documented as not re-entrant: with the sole
 * exception of the execution routines, no two FFTW API calls may run
 * concurrently. FFTWpp therefore takes this mutex around every planner call it
 * makes -- plan creation, plan destruction, wisdom manipulation and
 * `CleanUp()` -- so that consumers do not need a lock of their own.
 *
 * The mutex is exposed so that a consumer mixing FFTWpp with direct calls to
 * the FFTW C API can serialise against the same lock. Prefer `PlannerLock`,
 * which is the RAII form.
 *
 * Execution is deliberately *not* serialised: `Execute` is thread-safe in FFTW
 * and is where the work happens.
 *
 * @return A reference to the single process-wide planner mutex.
 */
[[nodiscard]] inline std::mutex& PlannerMutex() {
  static std::mutex mutex;
  return mutex;
}

/**
 * @class PlannerLock
 * @brief An RAII lock on the process-wide FFTW planner mutex.
 * @details Construct one for the duration of any direct call into the FFTW C
 * planner made alongside FFTWpp. All of FFTWpp's own planner calls take this
 * lock already, so a consumer that only uses FFTWpp never needs to.
 *
 * @code
 * {
 *   auto lock = FFTWpp::PlannerLock{};
 *   auto raw = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
 * }
 * @endcode
 *
 * The lock is not recursive; do not construct one around a call to an FFTWpp
 * function that takes it internally.
 */
class PlannerLock {
 public:
  /** @brief Acquires the planner mutex. */
  PlannerLock() : _lock{PlannerMutex()} {}

  PlannerLock(const PlannerLock&) = delete;
  PlannerLock(PlannerLock&&) = delete;
  PlannerLock& operator=(const PlannerLock&) = delete;
  PlannerLock& operator=(PlannerLock&&) = delete;

  /** @brief Releases the planner mutex. */
  ~PlannerLock() = default;

 private:
  std::scoped_lock<std::mutex> _lock;
};

//--------------------------------------------------------------//
//                    Custom fftw3 allocator                    //
//--------------------------------------------------------------//

/**
 * @brief A custom STL allocator that uses `fftw_malloc` and `fftw_free`.
 * @details This ensures that memory allocated for containers like std::vector
 * is correctly aligned for SIMD instructions, as required by FFTW for optimal
 * performance. The allocator is stateless, so all instances compare equal and
 * allocations made through one may be freed through any other.
 * @tparam T The type of the elements to be allocated.
 */
template <typename T>
class Allocator {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  /** @brief Rebinds the allocator to another value type. */
  template <typename U>
  struct rebind {
    using other = Allocator<U>;
  };

  /** @brief Default constructor. */
  constexpr Allocator() noexcept = default;
  /** @brief Copy constructor from an allocator of a different type. */
  template <class U>
  constexpr Allocator(const Allocator<U>&) noexcept {}

  /**
   * @brief Allocates `n` elements of type `T`.
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated, FFTW-aligned memory.
   * @throws std::bad_alloc if `n` elements cannot be allocated, either because
   * the request overflows `std::size_t` or because `fftw_malloc` fails.
   */
  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    auto p = static_cast<T*>(fftw_malloc(sizeof(T) * n));
    if (p == nullptr && n > 0) throw std::bad_alloc();
    return p;
  }

  /**
   * @brief Deallocates memory previously allocated with `allocate`.
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated (unused).
   */
  void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
    fftw_free(p);
  }
};

/**
 * @brief Compares two allocators for equality.
 * @return Always returns true, as FFTW allocators are stateless.
 */
template <class T, class U>
constexpr bool operator==(const Allocator<T>&, const Allocator<U>&) noexcept {
  return true;
}

/**
 * @brief Compares two allocators for inequality.
 * @return Always returns false, as FFTW allocators are stateless.
 */
template <class T, class U>
constexpr bool operator!=(const Allocator<T>&, const Allocator<U>&) noexcept {
  return false;
}

/**
 * @brief Alias for `std::vector` using the custom FFTW-aligned allocator.
 * @tparam T The element type of the vector.
 */
template <typename T>
using vector = std::vector<T, Allocator<T>>;

namespace Internal {

/** @brief The count of live `Ranges::Plan` objects. @see LivePlanCount */
inline std::atomic<int>& LivePlanCounter() {
  static std::atomic<int> count{0};
  return count;
}

}  // namespace Internal

/**
 * @brief Returns the number of `Ranges::Plan` objects currently alive.
 * @details `CleanUp` leaves every live plan undefined, so this is the quantity
 * that must be zero before calling it.
 *
 * Only plans owned by a `Ranges::Plan` are counted. A raw handle obtained from
 * the `Plan` factory functions in this header, and owned by the caller, is
 * not: this is a necessary condition for `CleanUp` being safe, not a proof of
 * it.
 * @return The number of live `Ranges::Plan` objects.
 */
[[nodiscard]] inline int LivePlanCount() {
  return Internal::LivePlanCounter().load(std::memory_order_relaxed);
}

/**
 * @brief Discards FFTW's process-global planner state for all three
 * precisions.
 * @details **Most programs should not call this.** FFTW's persistent state --
 * accumulated wisdom and the list of algorithms available in this
 * configuration -- lives in FFTW's own globals and is reachable for the
 * lifetime of the process, so leaving it alone is not a leak and no leak
 * checker reports one. Calling this is worth it in three situations, and
 * otherwise costs more than it saves:
 *
 * 1. Under a leak checker configured to report *still reachable* blocks,
 *    where a silent report is wanted.
 * 2. In a plugin or extension module that may be unloaded from a long-lived
 *    host process, where the state really would be orphaned.
 * 3. To reset FFTW deliberately, for instance to force re-measurement.
 *
 * Two costs come with it. Accumulated wisdom is discarded, so any
 * `ExportWisdom` must happen first. And every live plan becomes undefined --
 * including plans owned by unrelated code in the same process, which is why a
 * library should be reluctant to call this on its users' behalf.
 *
 * This function does not destroy plans, and does not clean up the optional
 * FFTW threads interfaces; see `CleanUpThreads` for the latter.
 *
 * @throws std::logic_error if any `Ranges::Plan` is still alive, since
 * cleaning up would leave it undefined. The check is a best-effort diagnostic:
 * it cannot see raw handles the caller owns, and in a threaded program another
 * thread may create a plan immediately afterwards.
 * @see LivePlanCount, CleanUpThreads
 */
inline void CleanUp() {
  const auto live = LivePlanCount();
  if (live > 0) {
    throw std::logic_error(
        "FFTWpp::CleanUp: " + std::to_string(live) +
        " plan(s) are still alive, and cleaning up would leave them "
        "undefined. Destroy every Ranges::Plan first -- or simply do not "
        "call CleanUp, which most programs have no reason to.");
  }
  const auto lock = PlannerLock{};
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

/**
 * @brief Safely casts a pointer to `std::complex<Real>` to the corresponding
 * FFTW complex type.
 * @tparam Real The floating-point precision (`float`, `double`, `long double`).
 * @param z A pointer to the `std::complex` data.
 * @return A pointer to the data cast as the appropriate FFTW complex type.
 */
template <NumericConcepts::Real Real>
auto ComplexCast(std::complex<Real>* z) {
  if constexpr (NumericConcepts::Float<Real>) {
    return reinterpret_cast<fftwf_complex*>(z);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return reinterpret_cast<fftw_complex*>(z);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return reinterpret_cast<fftwl_complex*>(z);
  }
}

//----------------------------------------------------------//
//                     Alignment queries                    //
//----------------------------------------------------------//

/**
 * @brief Returns FFTW's alignment class for a real data pointer.
 * @details This wraps `fftw*_alignment_of`. FFTW guarantees that a plan may be
 * executed on new arrays -- via the `Execute(plan, in, out)` overloads -- only
 * when those arrays have the same alignment class as the arrays the plan was
 * created for. The value itself is opaque; only equality between two values of
 * the same precision is meaningful.
 * @tparam Real The floating-point precision of the data.
 * @param p A pointer to the data whose alignment class is wanted.
 * @return The opaque alignment class of `p`.
 */
template <NumericConcepts::Real Real>
[[nodiscard]] int AlignmentOf(Real* p) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_alignment_of(p);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_alignment_of(p);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_alignment_of(p);
  }
}

/**
 * @brief Returns FFTW's alignment class for a complex data pointer.
 * @details `std::complex<Real>` is required to have the same layout as an
 * array of two `Real`, so the alignment class of the complex pointer is that
 * of its first real component.
 * @tparam Real The floating-point precision of the data.
 * @param z A pointer to the data whose alignment class is wanted.
 * @return The opaque alignment class of `z`.
 * @see AlignmentOf(Real*)
 */
template <NumericConcepts::Real Real>
[[nodiscard]] int AlignmentOf(std::complex<Real>* z) {
  return AlignmentOf(reinterpret_cast<Real*>(z));
}

/**
 * @brief Reports whether two pointers are interchangeable for new-array
 * execution.
 * @details Two pointers may be substituted for one another in a call to
 * `Execute(plan, in, out)` only if they share an alignment class. This is the
 * check that neither FFTW nor the raw `Execute` overloads perform.
 * @tparam T The pointee type of the first pointer, real or complex.
 * @tparam U The pointee type of the second pointer, real or complex.
 * @param first The first pointer.
 * @param second The second pointer.
 * @return `true` if the two pointers have the same alignment class.
 */
template <typename T, typename U>
[[nodiscard]] bool SameAlignment(T* first, U* second) {
  return AlignmentOf(first) == AlignmentOf(second);
}

//----------------------------------------------------------//
//                     FFTW-internal threads                //
//----------------------------------------------------------//

/**
 * @brief Reports whether this build of FFTWpp was configured against FFTW's
 * threads libraries.
 * @details Set to `true` when the macro `FFTWPP_ENABLE_THREADS` is defined,
 * which the CMake option `FFTWPP_USE_FFTW_THREADS` does alongside linking
 * `libfftw3_threads` (or `libfftw3_omp`) for every precision.
 *
 * The threading wrappers below are declared unconditionally, so consumers can
 * compile against them regardless; calling one without the corresponding FFTW
 * threads library on the link line is a link error rather than a silent
 * failure. Branch on this constant to keep such a call out of a build that
 * cannot satisfy it.
 */
inline constexpr bool ThreadsEnabled =
#ifdef FFTWPP_ENABLE_THREADS
    true;
#else
    false;
#endif

/**
 * @brief Initialises FFTW's own threading support for every precision.
 * @details Wraps `fftwf_init_threads`, `fftw_init_threads` and
 * `fftwl_init_threads`. It must be called once, before any plan is created,
 * and requires the FFTW threads libraries on the link line. Use
 * `PlanWithNumberOfThreads` afterwards to choose how many threads subsequently
 * created plans may use, and `CleanUpThreads` at exit.
 *
 * FFTWpp does not call this for you: FFTW-internal threading is opt-in,
 * because a consumer that already parallelises over many independent
 * transforms wants each plan single-threaded. It is intended for the opposite
 * case, of one transform large enough to be worth splitting.
 *
 * @return `true` if all three precisions initialised successfully.
 * @see ThreadsEnabled, PlanWithNumberOfThreads, CleanUpThreads, ThreadSession
 */
[[nodiscard]] inline bool InitialiseThreads() {
  const auto lock = PlannerLock{};
  const auto f = fftwf_init_threads();
  const auto d = fftw_init_threads();
  const auto l = fftwl_init_threads();
  return f != 0 && d != 0 && l != 0;
}

/**
 * @brief Sets the number of threads that subsequently created plans may use.
 * @details Wraps `fftw*_plan_with_nthreads` for every precision. It affects
 * only plans created after the call; existing plans keep the thread count they
 * were planned with. Requires a prior successful `InitialiseThreads()`.
 * @param numberOfThreads The maximum number of threads per plan. Must be
 * positive; a value of 1 restores single-threaded execution.
 * @throws std::invalid_argument if `numberOfThreads` is not positive.
 */
inline void PlanWithNumberOfThreads(int numberOfThreads) {
  if (numberOfThreads < 1) {
    throw std::invalid_argument("the number of FFTW threads must be positive");
  }
  const auto lock = PlannerLock{};
  fftwf_plan_with_nthreads(numberOfThreads);
  fftw_plan_with_nthreads(numberOfThreads);
  fftwl_plan_with_nthreads(numberOfThreads);
}

/**
 * @brief Releases the resources allocated by `InitialiseThreads`.
 * @details Wraps `fftw*_cleanup_threads`, which also performs the work of
 * `fftw*_cleanup` and therefore carries the same caveats: wisdom is discarded
 * and every live plan becomes undefined. Calling this makes `CleanUp()`
 * unnecessary.
 * @throws std::logic_error if any `Ranges::Plan` is still alive.
 * @see CleanUp, LivePlanCount
 */
inline void CleanUpThreads() {
  const auto live = LivePlanCount();
  if (live > 0) {
    throw std::logic_error(
        "FFTWpp::CleanUpThreads: " + std::to_string(live) +
        " plan(s) are still alive, and cleaning up would leave them "
        "undefined. Destroy every Ranges::Plan first.");
  }
  const auto lock = PlannerLock{};
  fftwf_cleanup_threads();
  fftw_cleanup_threads();
  fftwl_cleanup_threads();
}

/**
 * @class ThreadSession
 * @brief An RAII guard over FFTW's threading support.
 * @details Constructing one initialises FFTW threading and sets the per-plan
 * thread count; destroying one calls `CleanUpThreads`. Create it before any
 * plan and let it outlive every plan, since `CleanUpThreads` requires that all
 * plans have been destroyed.
 *
 * @code
 * auto threads = FFTWpp::ThreadSession(4);   // plans may use four threads
 * auto plan = FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Measure,
 *                                  FFTWpp::Forward);
 * plan.Execute();
 * @endcode
 *
 * @see ThreadsEnabled
 */
class ThreadSession {
 public:
  /**
   * @brief Initialises FFTW threading and sets the per-plan thread count.
   * @param numberOfThreads The maximum number of threads per plan.
   * @throws std::runtime_error if FFTW threading fails to initialise.
   * @throws std::invalid_argument if `numberOfThreads` is not positive.
   */
  explicit ThreadSession(int numberOfThreads) {
    if (!InitialiseThreads()) {
      throw std::runtime_error("FFTW failed to initialise threading support");
    }
    PlanWithNumberOfThreads(numberOfThreads);
  }

  ThreadSession(const ThreadSession&) = delete;
  ThreadSession(ThreadSession&&) = delete;
  ThreadSession& operator=(const ThreadSession&) = delete;
  ThreadSession& operator=(ThreadSession&&) = delete;

  /**
   * @brief Calls `CleanUpThreads`, unless plans are still alive.
   * @details A destructor must not throw, so a violated contract is reported
   * by assertion in a debug build and the cleanup is skipped. Skipping leaves
   * FFTW's reachable state in place, which is harmless; proceeding would leave
   * the surviving plans undefined, which is not.
   */
  ~ThreadSession() {
    if (LivePlanCount() > 0) {
      assert(false &&
             "plans outlived the ThreadSession; skipping CleanUpThreads");
      return;
    }
    CleanUpThreads();
  }
};

//----------------------------------------------------------//
//                         1D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 1D complex-to-complex Discrete Fourier Transform
 * (DFT).
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT. Use `FFTW_FORWARD` (-1) or
 * `FFTW_BACKWARD` (+1).
 * @param flag A bitwise OR of FFTW planner flags (e.g., `FFTW_MEASURE`,
 * `FFTW_ESTIMATE`).
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
}

/**
 * @brief Creates a plan for a 1D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array. The size should be n/2 + 1.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, std::complex<Real>* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 1D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the complex input array. The size should be n/2 + 1.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, Real* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 1D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n The size of the transform.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind The kind of R2R transform (e.g., `FFTW_REDFT00`, `FFTW_DHT`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, Real* out, fftw_r2r_kind kind, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_1d(n, in, out, kind, flag);
  }
}

//----------------------------------------------------------//
//                         2D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 2D complex-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT (`FFTW_FORWARD` or
 * `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
}

/**
 * @brief Creates a plan for a 2D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, std::complex<Real>* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 2D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, Real* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 2D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind0 The kind of R2R transform for the first dimension.
 * @param kind1 The kind of R2R transform for the second dimension.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
}

//----------------------------------------------------------//
//                         3D plans                         //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a 3D complex-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent in the DFT (`FFTW_FORWARD` or
 * `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in,
          std::complex<Real>* out, int sign, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
}

/**
 * @brief Creates a plan for a 3D real-to-complex (R2C) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, std::complex<Real>* out,
          unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a 3D complex-to-real (C2R) DFT.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in, Real* out,
          unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a 3D real-to-real (R2R) transform.
 * @tparam Real The floating-point precision of the data.
 * @param n0 The size of the transform in the first dimension.
 * @param n1 The size of the transform in the second dimension.
 * @param n2 The size of the transform in the third dimension.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind0 The kind of R2R transform for the first dimension.
 * @param kind1 The kind of R2R transform for the second dimension.
 * @param kind2 The kind of R2R transform for the third dimension.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
}

//----------------------------------------------------------//
//                   Multi-dimensional plans                //
//----------------------------------------------------------//

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) complex-to-complex
 * DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions for the transform.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign The sign of the exponent (`FFTW_FORWARD` or `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                         flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) real-to-complex DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, std::complex<Real>* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) complex-to-real DFT.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, Real* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Creates a plan for a multi-dimensional (rank > 0) real-to-real
 * transform.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of dimensions.
 * @param n Pointer to an array of size `rank` specifying the dimensions.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind Pointer to an array of `fftw_r2r_kind` of size `rank`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, Real* out, fftw_r2r_kind* kind,
          unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r(rank, n, in, out, kind, flag);
  }
}

//-------------------------------------------------------------//
//                      Advanced interface                     //
//-------------------------------------------------------------//

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * complex-to-complex DFTs.
 * @details This is a wrapper for `fftw_plan_many_dft` and its
 * precision-specific variants.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the complex input data.
 * @param inEmbed The "embedded" dimensions of the input array (for sub-arrays).
 * Can be `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the complex output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param sign The sign of the exponent (`FFTW_FORWARD` or `FFTW_BACKWARD`).
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
          int outStride, int outDist, int sign, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                              inStride, inDist, ComplexCast(out), outEmbed,
                              outStride, outDist, sign, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * real-to-complex DFTs.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the real input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the complex output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, std::complex<Real>* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                  inDist, ComplexCast(out), outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional
 * complex-to-real DFTs.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the complex input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the real output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, Real* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                  inStride, inDist, out, outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
}

/**
 * @brief Creates a plan for multiple, strided, multi-dimensional real-to-real
 * transforms.
 * @tparam Real The floating-point precision of the data.
 * @param rank Number of dimensions.
 * @param n Array of dimensions.
 * @param howMany The number of transforms to compute.
 * @param in Pointer to the real input data.
 * @param inEmbed The "embedded" dimensions of the input array. Can be
 * `nullptr`.
 * @param inStride The distance between consecutive elements in a dimension.
 * @param inDist The distance between the start of consecutive datasets.
 * @param out Pointer to the real output data.
 * @param outEmbed The "embedded" dimensions of the output array. Can be
 * `nullptr`.
 * @param outStride The distance between consecutive elements in a dimension of
 * the output.
 * @param outDist The distance between the start of consecutive output datasets.
 * @param kind Pointer to an array of `fftw_r2r_kind` of size `rank`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, Real* out, int* outEmbed, int outStride, int outDist,
          fftw_r2r_kind* kind, unsigned flag) {
  const auto lock = PlannerLock{};
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                              out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
  }
}

//----------------------------------------------------------//
//                        Guru plans                        //
//----------------------------------------------------------//

/**
 * @brief One dimension of a guru transform: an extent and a stride on each
 * side.
 * @details Mirrors FFTW's `fftw_iodim`, with two differences that exist to
 * make it harder to misuse.
 *
 * The members are named rather than positional, so a designated initialiser
 * says which stride is which and the pair cannot be transposed by accident:
 * @code
 * auto dim = FFTWpp::Dim{.n = 64, .inStride = 1, .outStride = 8};
 * @endcode
 *
 * The members are `std::ptrdiff_t` rather than `int`, so the choice between
 * FFTW's 32-bit and 64-bit guru entry points is made for you: a layout whose
 * extents and strides all fit in an `int` uses `fftw_plan_guru_*`, and one
 * that does not uses `fftw_plan_guru64_*`.
 *
 * A stride is measured in elements of the array it refers to, not in bytes,
 * and may be negative to traverse an axis backwards.
 */
struct Dim {
  /// @brief The number of elements along this dimension.
  std::ptrdiff_t n = 0;
  /// @brief The distance between consecutive elements in the input array.
  std::ptrdiff_t inStride = 0;
  /// @brief The distance between consecutive elements in the output array.
  std::ptrdiff_t outStride = 0;

  /** @brief Defaulted equality operator. */
  friend constexpr bool operator==(const Dim&, const Dim&) = default;
};

namespace Internal {

/**
 * @brief Reports whether a list of dimensions fits FFTW's 32-bit guru
 * interface.
 */
inline bool FitsGuru32(const Dim* dims, int rank) {
  constexpr auto limit =
      static_cast<std::ptrdiff_t>(std::numeric_limits<int>::max());
  constexpr auto floor =
      static_cast<std::ptrdiff_t>(std::numeric_limits<int>::min());
  for (int i = 0; i < rank; ++i) {
    const auto& d = dims[i];
    if (d.n > limit || d.inStride > limit || d.outStride > limit) return false;
    if (d.inStride < floor || d.outStride < floor) return false;
  }
  return true;
}

/** @brief Converts FFTWpp dimensions to FFTW's 32-bit `iodim`. */
inline std::vector<fftw_iodim> ToIoDim(const Dim* dims, int rank) {
  auto converted = std::vector<fftw_iodim>(static_cast<std::size_t>(rank));
  for (int i = 0; i < rank; ++i) {
    converted[i].n = static_cast<int>(dims[i].n);
    converted[i].is = static_cast<int>(dims[i].inStride);
    converted[i].os = static_cast<int>(dims[i].outStride);
  }
  return converted;
}

/** @brief Converts FFTWpp dimensions to FFTW's 64-bit `iodim64`. */
inline std::vector<fftw_iodim64> ToIoDim64(const Dim* dims, int rank) {
  auto converted = std::vector<fftw_iodim64>(static_cast<std::size_t>(rank));
  for (int i = 0; i < rank; ++i) {
    converted[i].n = dims[i].n;
    converted[i].is = dims[i].inStride;
    converted[i].os = dims[i].outStride;
  }
  return converted;
}

}  // namespace Internal

/**
 * @brief Creates a guru plan for a complex-to-complex DFT.
 * @details The guru interface generalises the advanced interface in two ways:
 * every dimension carries its own input and output stride, and the loop over
 * repeated transforms is itself multi-dimensional rather than a single
 * `(howMany, dist)` pair. Together those express layouts the advanced
 * interface cannot, the commonest being a transform along an interior axis of
 * an array of rank three or more.
 *
 * FFTW checks none of this. Prefer `Ranges::GuruPlan`, which validates the
 * layout before handing it over.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of transform dimensions.
 * @param dims The transform dimensions.
 * @param howManyRank The number of dimensions of the repetition loop. May be
 * zero, for a single transform.
 * @param howManyDims The dimensions of the repetition loop.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 * @param sign `FFTW_FORWARD` or `FFTW_BACKWARD`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, const Dim* dims, int howManyRank, const Dim* howManyDims,
          std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flag) {
  const auto lock = PlannerLock{};
  if (Internal::FitsGuru32(dims, rank) &&
      Internal::FitsGuru32(howManyDims, howManyRank)) {
    const auto d = Internal::ToIoDim(dims, rank);
    const auto h = Internal::ToIoDim(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru_dft(rank, d.data(), howManyRank, h.data(),
                                 ComplexCast(in), ComplexCast(out), sign, flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru_dft(rank, d.data(), howManyRank, h.data(),
                                ComplexCast(in), ComplexCast(out), sign, flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru_dft(rank, d.data(), howManyRank, h.data(),
                                 ComplexCast(in), ComplexCast(out), sign, flag);
    }
  } else {
    const auto d = Internal::ToIoDim64(dims, rank);
    const auto h = Internal::ToIoDim64(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru64_dft(rank, d.data(), howManyRank, h.data(),
                                   ComplexCast(in), ComplexCast(out), sign,
                                   flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru64_dft(rank, d.data(), howManyRank, h.data(),
                                  ComplexCast(in), ComplexCast(out), sign,
                                  flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru64_dft(rank, d.data(), howManyRank, h.data(),
                                   ComplexCast(in), ComplexCast(out), sign,
                                   flag);
    }
  }
}

/**
 * @brief Creates a guru plan for a real-to-complex DFT.
 * @details The extents in `dims` are those of the *real* array. The last
 * transform dimension is special: the complex array holds `n / 2 + 1` elements
 * along it, rounded down. `inStride` refers to the real array and `outStride`
 * to the complex one.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of transform dimensions.
 * @param dims The transform dimensions, with real-array extents.
 * @param howManyRank The number of dimensions of the repetition loop.
 * @param howManyDims The dimensions of the repetition loop.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, const Dim* dims, int howManyRank, const Dim* howManyDims,
          Real* in, std::complex<Real>* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if (Internal::FitsGuru32(dims, rank) &&
      Internal::FitsGuru32(howManyDims, howManyRank)) {
    const auto d = Internal::ToIoDim(dims, rank);
    const auto h = Internal::ToIoDim(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru_dft_r2c(rank, d.data(), howManyRank, h.data(), in,
                                     ComplexCast(out), flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru_dft_r2c(rank, d.data(), howManyRank, h.data(), in,
                                    ComplexCast(out), flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru_dft_r2c(rank, d.data(), howManyRank, h.data(), in,
                                     ComplexCast(out), flag);
    }
  } else {
    const auto d = Internal::ToIoDim64(dims, rank);
    const auto h = Internal::ToIoDim64(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru64_dft_r2c(rank, d.data(), howManyRank, h.data(),
                                       in, ComplexCast(out), flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru64_dft_r2c(rank, d.data(), howManyRank, h.data(), in,
                                      ComplexCast(out), flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru64_dft_r2c(rank, d.data(), howManyRank, h.data(),
                                       in, ComplexCast(out), flag);
    }
  }
}

/**
 * @brief Creates a guru plan for a complex-to-real DFT.
 * @details The extents in `dims` are those of the *real* array, as for the
 * real-to-complex case. Here `inStride` refers to the complex array and
 * `outStride` to the real one.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of transform dimensions.
 * @param dims The transform dimensions, with real-array extents.
 * @param howManyRank The number of dimensions of the repetition loop.
 * @param howManyDims The dimensions of the repetition loop.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, const Dim* dims, int howManyRank, const Dim* howManyDims,
          std::complex<Real>* in, Real* out, unsigned flag) {
  const auto lock = PlannerLock{};
  if (Internal::FitsGuru32(dims, rank) &&
      Internal::FitsGuru32(howManyDims, howManyRank)) {
    const auto d = Internal::ToIoDim(dims, rank);
    const auto h = Internal::ToIoDim(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                     ComplexCast(in), out, flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                    ComplexCast(in), out, flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                     ComplexCast(in), out, flag);
    }
  } else {
    const auto d = Internal::ToIoDim64(dims, rank);
    const auto h = Internal::ToIoDim64(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru64_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                       ComplexCast(in), out, flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru64_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                      ComplexCast(in), out, flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru64_dft_c2r(rank, d.data(), howManyRank, h.data(),
                                       ComplexCast(in), out, flag);
    }
  }
}

/**
 * @brief Creates a guru plan for a real-to-real transform.
 * @tparam Real The floating-point precision of the data.
 * @param rank The number of transform dimensions.
 * @param dims The transform dimensions.
 * @param howManyRank The number of dimensions of the repetition loop.
 * @param howManyDims The dimensions of the repetition loop.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 * @param kind Pointer to an array of `fftw_r2r_kind` of size `rank`.
 * @param flag A bitwise OR of FFTW planner flags.
 * @return An FFTW plan handle corresponding to the data precision.
 */
template <NumericConcepts::Real Real>
auto Plan(int rank, const Dim* dims, int howManyRank, const Dim* howManyDims,
          Real* in, Real* out, const fftw_r2r_kind* kind, unsigned flag) {
  const auto lock = PlannerLock{};
  if (Internal::FitsGuru32(dims, rank) &&
      Internal::FitsGuru32(howManyDims, howManyRank)) {
    const auto d = Internal::ToIoDim(dims, rank);
    const auto h = Internal::ToIoDim(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru_r2r(rank, d.data(), howManyRank, h.data(), in, out,
                                 kind, flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru_r2r(rank, d.data(), howManyRank, h.data(), in, out,
                                kind, flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru_r2r(rank, d.data(), howManyRank, h.data(), in, out,
                                 kind, flag);
    }
  } else {
    const auto d = Internal::ToIoDim64(dims, rank);
    const auto h = Internal::ToIoDim64(howManyDims, howManyRank);
    if constexpr (NumericConcepts::Float<Real>) {
      return fftwf_plan_guru64_r2r(rank, d.data(), howManyRank, h.data(), in,
                                   out, kind, flag);
    }
    if constexpr (NumericConcepts::Double<Real>) {
      return fftw_plan_guru64_r2r(rank, d.data(), howManyRank, h.data(), in,
                                  out, kind, flag);
    }
    if constexpr (NumericConcepts::LongDouble<Real>) {
      return fftwl_plan_guru64_r2r(rank, d.data(), howManyRank, h.data(), in,
                                   out, kind, flag);
    }
  }
}

//----------------------------------------------------------//
//                 Plan destruction functions               //
//----------------------------------------------------------//

/**
 * @brief Destroys a given FFTW plan and frees associated resources.
 * @tparam PlanType The type of the plan handle, constrained by `IsPlan`.
 * @param plan The plan to destroy. Must not be null.
 */
template <IsPlan PlanType>
void Destroy(PlanType plan) {
  assert(plan != nullptr);
  const auto lock = PlannerLock{};
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_destroy_plan(plan);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_destroy_plan(plan);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_destroy_plan(plan);
  }
}

//----------------------------------------------------------//
//                  Plan execution functions                //
//----------------------------------------------------------//

/**
 * @brief Executes a plan that was created for in-place transforms.
 * @details This is for plans where the input and output pointers were the same
 * during creation. It wraps the appropriate `fftw*_execute` function based on
 * the plan's precision.
 * @tparam PlanType The type of the plan handle, constrained by `IsPlan`.
 * @param plan The plan to execute. Must not be null.
 */
template <IsPlan PlanType>
void Execute(PlanType plan) {
  assert(plan != nullptr);
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute(plan);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute(plan);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute(plan);
  }
}

/**
 * @brief Executes an out-of-place complex-to-complex DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft` function based on the
 * plan's precision. This allows reusing a plan with different input/output
 * buffers that have the same alignment properties as the ones used for
 * planning.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the complex output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, std::complex<Real>* in, std::complex<Real>* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft(plan, ComplexCast(in), ComplexCast(out));
  }
}

/**
 * @brief Executes an out-of-place real-to-complex DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft_r2c` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the real input array.
 * @param out Pointer to the complex output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, Real* in, std::complex<Real>* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft_r2c(plan, in, ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft_r2c(plan, in, ComplexCast(out));
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft_r2c(plan, in, ComplexCast(out));
  }
}

/**
 * @brief Executes an out-of-place complex-to-real DFT plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_dft_c2r` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the complex input array.
 * @param out Pointer to the real output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, std::complex<Real>* in, Real* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_dft_c2r(plan, ComplexCast(in), out);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_dft_c2r(plan, ComplexCast(in), out);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_dft_c2r(plan, ComplexCast(in), out);
  }
}

/**
 * @brief Executes an out-of-place real-to-real transform plan with new arrays.
 * @details Wraps the appropriate `fftw*_execute_r2r` function based on the
 * plan's precision.
 * @tparam PlanType The type of the plan handle.
 * @tparam Real The floating-point precision of the data.
 * @param plan The plan to execute.
 * @param in Pointer to the real input array.
 * @param out Pointer to the real output array.
 */
template <typename PlanType, NumericConcepts::Real Real>
requires CheckPrecision<PlanType, Real>
void Execute(PlanType plan, Real* in, Real* out) {
  if constexpr (std::same_as<PlanType, fftwf_plan>) {
    fftwf_execute_r2r(plan, in, out);
  }
  if constexpr (std::same_as<PlanType, fftw_plan>) {
    fftw_execute_r2r(plan, in, out);
  }
  if constexpr (std::same_as<PlanType, fftwl_plan>) {
    fftwl_execute_r2r(plan, in, out);
  }
}

}  // namespace FFTWpp

#endif  // FFTWPP_CORE_GUARD_H
