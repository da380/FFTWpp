
#ifndef FFTWPP_CORE_GUARD_H
#define FFTWPP_CORE_GUARD_H

#include <cassert>
#include <complex>
#include <concepts>
#include <memory>
#include <variant>
#include <vector>

#include "NumericConcepts/Numeric.hpp"
#include "fftw3.h"

namespace FFTWpp {

//------------------------------------------------------//
//              Define some useful concepts             //
//------------------------------------------------------//

template <typename T>
concept IsPlan = std::same_as<T, fftwf_plan> or std::same_as<T, fftw_plan> or
                 std::same_as<T, fftwl_plan>;

template <typename PlanType, typename Real>
concept CheckPrecision =
    (std::same_as<PlanType, fftwf_plan> and NumericConcepts::Float<Real>) or
    (std::same_as<PlanType, fftw_plan> and NumericConcepts::Double<Real>) or
    (std::same_as<PlanType, fftwl_plan> and NumericConcepts::LongDouble<Real>);

//--------------------------------------------------------------//
//                    Custom fftw3 allocator                    //
//--------------------------------------------------------------//

template <typename T>
class Allocator {
 public:
  using value_type = T;
  Allocator() noexcept {}
  template <class U>
  Allocator(const Allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    return static_cast<T*>(fftw_malloc(sizeof(T) * n));
  }
  void deallocate(T* p, std::size_t n) { fftw_free(p); }
};

template <class T, class U>
constexpr bool operator==(const Allocator<T>&, const Allocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const Allocator<T>&, const Allocator<U>&) noexcept {
  return false;
}

template <typename T>
using vector = std::vector<T, Allocator<T>>;

void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

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
//                         1D plans                         //
//----------------------------------------------------------//

template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, std::complex<Real>* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n, std::complex<Real>* in, Real* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n, Real* in, Real* out, fftw_r2r_kind kind, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, std::complex<Real>* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, std::complex<Real>* in, Real* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in,
          std::complex<Real>* out, int sign, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, std::complex<Real>* out,
          unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in, Real* out,
          unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int n0, int n1, int n2, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, std::complex<Real>* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, std::complex<Real>* in, Real* out, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, Real* in, Real* out, fftw_r2r_kind* kind,
          unsigned flag) {
  if constexpr (NumericConcepts::Float<Real>) {
    return fftwf_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::Double<Real>) {
    return fftw_plan_r2r(rank, in, out, kind, flag);
  }
  if constexpr (NumericConcepts::LongDouble<Real>) {
    return fftwl_plan_r2r(rank, n, in, out, kind, flag);
  }
}

//-------------------------------------------------------------//
//                      Advanced interface                     //
//-------------------------------------------------------------//

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
          int outStride, int outDist, int sign, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, std::complex<Real>* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, Real* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
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

template <NumericConcepts::Real Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, Real* out, int* outEmbed, int outStride, int outDist,
          fftw_r2r_kind* kind, unsigned flag) {
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
//                 Plan destruction functions               //
//----------------------------------------------------------//

template <IsPlan PlanType>
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

#endif  // FFTWPP_MEMORY_GUARD_H
