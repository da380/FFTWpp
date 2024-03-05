#ifndef FFTWPP_CORE_GUARD_H
#define FFTWPP_CORE_GUARD_H

#include <cassert>
#include <complex>
#include <concepts>
#include <memory>
#include <variant>
#include <vector>

#include "fftw3.h"

namespace FFTWpp {

//------------------------------------------------------//
//          Concepts for real and complex floats        //
//------------------------------------------------------//
template <typename T>
concept IsReal = std::floating_point<T>;

template <typename Float>
concept IsSingle = std::same_as<Float, float>;

template <typename Float>
concept IsDouble = std::same_as<Float, double>;

template <typename Float>
concept IsLongDouble = std::same_as<Float, long double>;

template <typename T>
struct IsComplexHelper : std::false_type {};

template <typename T>
struct IsComplexHelper<std::complex<T>> : std::true_type {};

template <typename T>
concept IsComplex = IsComplexHelper<T>::value;

template <typename T>
struct RemoveComplexHelper {
  using value_type = T;
};

template <typename T>
struct RemoveComplexHelper<std::complex<T>> {
  using value_type = T;
};

template <typename T>
using RemoveComplex = typename RemoveComplexHelper<T>::value_type;

template <typename T>
concept IsScalar = IsReal<T> or IsComplex<T>;

//--------------------------------------------------------------//
//                    Custom fftw3 allocator                    //
//--------------------------------------------------------------//
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

// Clean up internal fftw3 things.
void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

//-----------------------------------------------------//
//       Cast from std::complex to fftw_complex        //
//-----------------------------------------------------//
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

//----------------------------------------------------------//
//                         1D plans                         //
//----------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int n, std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flags);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int n, Real* in, std::complex<Real>* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_1d(n, in, ComplexCast(out), flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_1d(n, in, ComplexCast(out), flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_1d(n, in, ComplexCast(out), flags);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int n, std::complex<Real>* in, Real* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_1d(n, ComplexCast(in), out, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_1d(n, ComplexCast(in), out, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_1d(n, ComplexCast(in), out, flags);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int n, Real* in, Real* out, fftw_r2r_kind kind, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_1d(n, in, out, kind, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_1d(n, in, out, kind, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_1d(n, in, out, kind, flags);
  }
}

//----------------------------------------------------------//
//                         2D plans                         //
//----------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int n0, int n1, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                            flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flags);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int n0, int n1, Real* in, std::complex<Real>* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flags);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int n0, int n1, std::complex<Real>* in, Real* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flags);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int n0, int n1, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flags);
  }
}

//----------------------------------------------------------//
//                         3D plans                         //
//----------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in,
          std::complex<Real>* out, int sign, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out), sign,
                            flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flags);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, Real* in, std::complex<Real>* out,
          unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flags);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in, Real* out,
          unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flags);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flags);
  }
}

//----------------------------------------------------------//
//                   Multi-dimensional plans                //
//----------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                         flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flags);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, Real* in, std::complex<Real>* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c(rank, n, in, ComplexCast(out), flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c(rank, n, in, ComplexCast(out), flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c(rank, n, in, ComplexCast(out), flags);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, std::complex<Real>* in, Real* out, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r(rank, n, ComplexCast(in), out, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r(rank, n, ComplexCast(in), out, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r(rank, n, ComplexCast(in), out, flags);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, Real* in, Real* out, fftw_r2r_kind* kind,
          unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r(rank, n, in, out, kind, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r(rank, in, out, kind, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r(rank, n, in, out, kind, flags);
  }
}

//-------------------------------------------------------------//
//                      Advanced interface                     //
//-------------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
          int outStride, int outDist, int sign, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                              inStride, inDist, ComplexCast(out), outEmbed,
                              outStride, outDist, sign, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flags);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, Real* out, int* outEmbed, int outStride,
          int outDist, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                  inStride, inDist, out, outEmbed, outStride,
                                  outDist, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flags);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, std::complex<Real>* out, int* outEmbed, int outStride,
          int outDist, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                  inDist, ComplexCast(out), outEmbed, outStride,
                                  outDist, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flags);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, Real* out, int* outEmbed, int outStride, int outDist,
          fftw_r2r_kind* kind, unsigned flags) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_r2c(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flags);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_r2c(rank, n, howMany, in, inEmbed, inStride, inDist,
                              out, outEmbed, outStride, outDist, kind, flags);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flags);
  }
}

//----------------------------------------------------------//
//                 Plan distruction functions               //
//----------------------------------------------------------//

template <typename PlanType>
requires std::same_as<PlanType, fftwf_plan> or
         std::same_as<PlanType, fftw_plan> or std::same_as<PlanType, fftwl_plan>
void Destroy(PlanType plan) {
  assert(plan != nullptr);
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

}  // namespace FFTWpp

#endif  // FFTWPP_MEMORY_GUARD_H
