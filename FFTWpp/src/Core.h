#ifndef FFTWPP_CORE_GUARD_H
#define FFTWPP_CORE_GUARD_H

#include <cassert>
#include <concepts>
#include <memory>
#include <variant>
#include <vector>

#include "Concepts.h"
#include "fftw3.h"

namespace FFTWpp {

// Define a custom allocator using the fftw3 versions of malloc and free.
template <typename T>
struct allocator {
  using value_type = T;
  allocator() noexcept {}
  template <class U>
  allocator(const allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    return static_cast<T*>(fftw_malloc(sizeof(T) * n));
  }
  void deallocate(T* p, std::size_t n) { fftw_free(p); }
};

template <class T, class U>
constexpr bool operator==(const allocator<T>&, const allocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const allocator<T>&, const allocator<U>&) noexcept {
  return false;
}

// Type alias for a std::vector using the custom allocator.
template <typename T>
using vector = std::vector<T, allocator<T>>;

// Reinterpret cast std::complex* to fftw_complex*.
template <std::floating_point Real>
auto ComplexCast(std::complex<Real>* z) {
  if constexpr (IsSingle<Real>) {
    return reinterpret_cast<fftwf_complex*>(z);
  }
  if constexpr (IsDouble<Real>) {
    return reinterpret_cast<fftw_complex*>(z);
  }
  if constexpr (IsLongDouble<Real>) {
    return reinterpret_cast<fftwl_complex*>(z);
  }
}

// Make C2C plans.
template <std::floating_point Real>
auto MakePlan(int rank, int* n, int howMany, std::complex<Real>* in,
              int* inEmbed, int inStride, int inDist, std::complex<Real>* out,
              int* outEmbed, int outStride, int outDist, int sign,
              unsigned flags) {
  if constexpr (IsSingle<Real>) {
    fftwf_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed, inStride,
                        inDist, ComplexCast(out), outEmbed, outStride, outDist,
                        sign, flags);
  }
  if constexpr (IsDouble<Real>) {
    fftw_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed, inStride,
                       inDist, ComplexCast(out), outEmbed, outStride, outDist,
                       sign, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    fftwl_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed, inStride,
                        inDist, ComplexCast(out), outEmbed, outStride, outDist,
                        sign, flags);
  }
}

// Make C2R plans.
template <std::floating_point Real>
auto MakePlan(int rank, int* n, int howMany, std::complex<Real>* in,
              int* inEmbed, int inStride, int inDist, Real* out, int* outEmbed,
              int outStride, int outDist, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    fftwf_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                            inStride, inDist, out, outEmbed, outStride, outDist,
                            flags);
  }
  if constexpr (IsDouble<Real>) {
    fftw_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed, inStride,
                           inDist, out, outEmbed, outStride, outDist, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    fftwl_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                            inStride, inDist, out, outEmbed, outStride, outDist,
                            flags);
  }
}

// Make R2C plans.
template <std::floating_point Real>
auto MakePlan(int rank, int* n, int howMany, Real* in, int* inEmbed,
              int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
              int outStride, int outDist, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    fftwf_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride, inDist,
                            ComplexCast(out), outEmbed, outStride, outDist,
                            flags);
  }
  if constexpr (IsDouble<Real>) {
    fftw_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride, inDist,
                           ComplexCast(out), outEmbed, outStride, outDist,
                           flags);
  }
  if constexpr (IsLongDouble<Real>) {
    fftwl_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride, inDist,
                            ComplexCast(out), outEmbed, outStride, outDist,
                            flags);
  }
}

// Make R2R plans.
template <std::floating_point Real>
auto MakePlan(int rank, int* n, int howMany, Real* in, int* inEmbed,
              int inStride, int inDist, Real* out, int* outEmbed, int outStride,
              int outDist, fftw_r2r_kind* kind, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    fftwf_plan_many_r2c(rank, n, howMany, in, inEmbed, inStride, inDist, out,
                        outEmbed, outStride, outDist, kind, flags);
  }
  if constexpr (IsDouble<Real>) {
    fftw_plan_many_r2c(rank, n, howMany, in, inEmbed, inStride, inDist, out,
                       outEmbed, outStride, outDist, kind, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    fftwl_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist, out,
                        outEmbed, outStride, outDist, kind, flags);
  }
}

// Execute a plan.
template <typename PlanType>
requires std::same_as<PlanType, fftwf_plan> or
         std::same_as<PlanType, fftw_plan> or std::same_as<PlanType, fftwl_plan>
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

// Execute C2C plan with new data.
template <typename PlanType, std::floating_point Real>
requires(std::same_as<PlanType, fftwf_plan> and IsSingle<Real>) or
        (std::same_as<PlanType, fftw_plan> and IsDouble<Real>) or
        (std::same_as<PlanType, fftwl_plan> and IsLongDouble<Real>)
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

// Execute R2C plan with new data.
template <typename PlanType, std::floating_point Real>
requires(std::same_as<PlanType, fftwf_plan> and IsSingle<Real>) or
        (std::same_as<PlanType, fftw_plan> and IsDouble<Real>) or
        (std::same_as<PlanType, fftwl_plan> and IsLongDouble<Real>)
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

// Execute C2R plan with new data.
template <typename PlanType, std::floating_point Real>
requires(std::same_as<PlanType, fftwf_plan> and IsSingle<Real>) or
        (std::same_as<PlanType, fftw_plan> and IsDouble<Real>) or
        (std::same_as<PlanType, fftwl_plan> and IsLongDouble<Real>)
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

// Execute R2R plan with new data.
template <typename PlanType, std::floating_point Real>
requires(std::same_as<PlanType, fftwf_plan> and IsSingle<Real>) or
        (std::same_as<PlanType, fftw_plan> and IsDouble<Real>) or
        (std::same_as<PlanType, fftwl_plan> and IsLongDouble<Real>)
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

// Clear up remaining memory. To be, optionally, called only when
// all plans have gone out of scope.
void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

}  // namespace FFTWpp

#endif  // FFTWPP_MEMORY_GUARD_H
