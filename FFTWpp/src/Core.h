
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

/**
 * @brief Concept for real floating point numbers.
 *
 */
template <typename T>
concept IsReal = std::floating_point<T>;

/**
 * @brief Concept for single precision floating point numbers.
 *
 */
template <typename Real>
concept IsSingle = std::same_as<Real, float>;

/**
 * @brief Concept for double precision floating point numbers.
 *
 */
template <typename Real>
concept IsDouble = std::same_as<Real, double>;

/**
 * @brief Concept for long double precision floating point numbers.
 *
 */
template <typename Real>
concept IsLongDouble = std::same_as<Real, long double>;

template <typename T>
struct IsComplexHelper : std::false_type {};

template <typename T>
struct IsComplexHelper<std::complex<T>> : std::true_type {};

/**
 * @brief Concept for complex floating point numbers.
 *
 */
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

/**
 * @brief Concept for determining if a type is real of complex floating point.
 *
 */
template <typename T>
concept IsScalar = IsReal<T> or IsComplex<T>;

//--------------------------------------------------------------//
//                    Custom fftw3 allocator                    //
//--------------------------------------------------------------//

/**
 * @brief Custom allocator that ensures correct alignment for `fftw3`.
 *
 * A custom allocator based on the function `fftw_malloc`. Use of
 * this function ensures that memory is aligned in such as way as
 * to allow for SIMD within the `fftw3` transformations.
 *
 * @tparam T
 */
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

/**
 * @brief Type allias for `std::vector<T,FFTWpp::Allocator<T>>`. This vector
 * can be used in place of `std::vector<T>` to ensure that data is correctly
 * alligned for the `fftw3` transformations.
 *
 * @tparam T
 */
template <typename T>
using vector = std::vector<T, Allocator<T>>;

/**
 * @brief Clean up internal data created by `fftw3`.
 *
 * `fftw3` plans internally save some other persistent data, such as the
 * accumulated wisdom and a list of algorithms available in the current
 * configuration. If you want to deallocate all of that and reset `fftw3` to
 *  the pristine state it was in when you started your program, you can call
 * this function.
 *
 * After calling FFTWpp::CleanUp, all existing plans become undefined, and you
should not attempt to execute them nor to destroy them. You can however create
and execute/destroy new plans, in which case `fftw3` starts accumulating wisdom
information again.

FFTWpp::ClearUp does not deallocate your plans, however. If working with raw
pointers, to prevent memory leaks, you must still call FFTWpp::Destroy before
executing this function.
 *
 */
void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

/**
 * @brief Cast from `std::<complex>*` to the corresponding `fftw3` complex
 * pointer type.
 *
 * @tparam Real Precision of the underlying complex number.
 * @param z pointer to std::complex.
 * @return `fttw3` complex pointer of the appropriate precision, e.g. result of
 * type `fftw_complex*` when the input is double precision.
 */
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

/**
 * @brief Returns a pointer to a plan for a 1D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_1d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *

 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param sign sign for the transformation.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 *
 */
template <std::floating_point Real>
auto Plan(int n, std::complex<Real>* in, std::complex<Real>* out, int sign,
          unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_1d(n, ComplexCast(in), ComplexCast(out), sign, flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 1D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_1d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *

 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 */
template <std::floating_point Real>
auto Plan(int n, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_1d(n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 1D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_1d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *

 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 *
 */
template <std::floating_point Real>
auto Plan(int n, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_1d(n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief  Returns a pointer to a plan for a 1D real-to-real
 * transformation of the specified kind.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_1d` and its variants
 * at different precisions. The API is identical.
 *
 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data.
 * @param out pointer to the end of the output data.
 * @param kind transformation type.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n, Real* in, Real* out, fftw_r2r_kind kind, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_1d(n, in, out, kind, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_1d(n, in, out, kind, flag);
  }
}

//----------------------------------------------------------//
//                         2D plans                         //
//----------------------------------------------------------//

/**
 * @brief Returns a pointer to a plan for a 2D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_2d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param sign sign for the transformation.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_2d(n0, n1, ComplexCast(in), ComplexCast(out), sign,
                             flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 2D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_2d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_2d(n0, n1, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 2D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_2d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag  planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_2d(n0, n1, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 2D real-to-real
 * transformation of the specified kind.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_2d` and its variants
 * at different precisions. The API is identical.
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param kind0 transformation type along zeroth-dimension.
 * @param kind1 transformation type along first-dimension.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_2d(n0, n1, in, out, kind0, kind1, flag);
  }
}

//----------------------------------------------------------//
//                         3D plans                         //
//----------------------------------------------------------//

/**
 * @brief Returns a pointer to a plan for a 3D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param sign sign for the transformation.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in,
          std::complex<Real>* out, int sign, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out), sign,
                            flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_3d(n0, n1, n2, ComplexCast(in), ComplexCast(out),
                             sign, flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 3D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, Real* in, std::complex<Real>* out,
          unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 3D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, std::complex<Real>* in, Real* out,
          unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r_3d(n0, n1, n2, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a 3D real-to-real
 * transformation of the specified kinds.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_3d` and its variants
 * at different precisions. The API is identical.
 *
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n1 size along second-dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param kind0 transformation type along zeroth-dimension.
 * @param kind1 transformation type along first-dimension.
 * @param kind2 transformation type along second-dimension.*
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 *
 */
template <std::floating_point Real>
auto Plan(int n0, int n1, int n2, Real* in, Real* out, fftw_r2r_kind kind0,
          fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r_3d(n0, n1, n2, in, out, kind0, kind1, kind2, flag);
  }
}

//----------------------------------------------------------//
//                   Multi-dimensional plans                //
//----------------------------------------------------------//

/**
 * @briefReturns a pointer to a plan for a multi-dimensional complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param rank the dimension of the data.
 * @param n pointer to the sizes along each dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param sign sign for the transformation.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int rank, int* n, std::complex<Real>* in, std::complex<Real>* out,
          int sign, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                         flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft(rank, n, ComplexCast(in), ComplexCast(out), sign,
                          flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a multi-dimensional real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param rank the dimension of the data.
 * @param n pointer to the sizes along each dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int rank, int* n, Real* in, std::complex<Real>* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_r2c(rank, n, in, ComplexCast(out), flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a multi-dimensional complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param rank the dimension of the data.
 * @param n pointer to the sizes along each dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int rank, int* n, std::complex<Real>* in, Real* out, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_dft_c2r(rank, n, ComplexCast(in), out, flag);
  }
}

/**
 * @brief Returns a pointer to a plan for a multi-dimensional real-to-real
 * transformation of the specifcied kinds.
 *
 * This function is a simple wrapper for `fftw_plan_r2r` and its variants
 * at different precisions. The API is identical.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param rank the dimension of the data.
 * @param n pointer to the sizes along each dimension.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param kind pointer to the transformation type along each dimension.
 * @param flag planner flag.
 * @return auto `fftw3` plan for the chosen precision.
 */
template <std::floating_point Real>
auto Plan(int rank, int* n, Real* in, Real* out, fftw_r2r_kind* kind,
          unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_r2r(rank, n, in, out, kind, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_r2r(rank, in, out, kind, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_r2r(rank, n, in, out, kind, flag);
  }
}

//-------------------------------------------------------------//
//                      Advanced interface                     //
//-------------------------------------------------------------//

// C2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, std::complex<Real>* out, int* outEmbed,
          int outStride, int outDist, int sign, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                              inStride, inDist, ComplexCast(out), outEmbed,
                              outStride, outDist, sign, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft(rank, n, howMany, ComplexCast(in), inEmbed,
                               inStride, inDist, ComplexCast(out), outEmbed,
                               outStride, outDist, sign, flag);
  }
}

// C2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, std::complex<Real>* in, int* inEmbed,
          int inStride, int inDist, Real* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                  inStride, inDist, out, outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft_c2r(rank, n, howMany, ComplexCast(in), inEmbed,
                                   inStride, inDist, out, outEmbed, outStride,
                                   outDist, flag);
  }
}

// R2C.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, std::complex<Real>* out, int* outEmbed, int outStride,
          int outDist, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                  inDist, ComplexCast(out), outEmbed, outStride,
                                  outDist, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_dft_r2c(rank, n, howMany, in, inEmbed, inStride,
                                   inDist, ComplexCast(out), outEmbed,
                                   outStride, outDist, flag);
  }
}

// R2R.
template <std::floating_point Real>
auto Plan(int rank, int* n, int howMany, Real* in, int* inEmbed, int inStride,
          int inDist, Real* out, int* outEmbed, int outStride, int outDist,
          fftw_r2r_kind* kind, unsigned flag) {
  if constexpr (IsSingle<Real>) {
    return fftwf_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (IsDouble<Real>) {
    return fftw_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                              out, outEmbed, outStride, outDist, kind, flag);
  }
  if constexpr (IsLongDouble<Real>) {
    return fftwl_plan_many_r2r(rank, n, howMany, in, inEmbed, inStride, inDist,
                               out, outEmbed, outStride, outDist, kind, flag);
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
