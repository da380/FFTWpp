
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
//              Define some useful concepts             //
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

/**
 * @brief Concept for determining if a type is an `fftw` plan.
 *
 */
template <typename T>
concept IsPlan = std::same_as<T, fftwf_plan> or std::same_as<T, fftw_plan> or
                 std::same_as<T, fftwl_plan>;

template <typename PlanType, typename Real>
concept CheckPrecision =
    (std::same_as<PlanType, fftwf_plan> and IsSingle<Real>) or
    (std::same_as<PlanType, fftw_plan> and IsDouble<Real>) or
    (std::same_as<PlanType, fftwl_plan> and IsLongDouble<Real>);

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
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html#SIMD-alignment-and-fftw_005fmalloc>
 *
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
 */
template <typename T>
using vector = std::vector<T, Allocator<T>>;

/**
 * @brief Clean up internal data created by `fftw3`.
 *
 * This is a wrapper for `fftw_cleanup` and its variants at different
 * precisions. This function can be called to clean up internal data linked to
 * `fftw3` plans. Doing so will invalidate any existing plans, but does not free
 * their associated memory.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Using-Plans.html>
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
 * Converts from `std::<complex>*` to the `fftw3` pointer for the
 * appropriate precision using a reinterpret_cast.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Complex-numbers.html#Complex-numbers>
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
 * @brief Returns a plan for a 1D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_1d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs>

 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n`.
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
 * @brief Returns a plan for a 1D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_1d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n/2+1`.
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
 * @brief Returns a plan for a 1D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_1d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>



 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n/2+1`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n`.
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
 * @brief  Returns a plan for a 1D real-to-real
 * transformation of the specified kind.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_1d` and its variants
 * at different precisions. The API is identical.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html#Real_002dto_002dReal-Transforms>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html#Real_002dto_002dReal-Transform-Kinds>
 *
 * @tparam Real precision of the data.
 * @param n size of the data.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n`.
 * @param out pointer to the end of the output data. Required size of underlying
 data equal to `n`.
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
 * @brief Returns a plan for a 2D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_2d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 *  It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs>
 *
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n0*n1`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n0*n1`.
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
 * @brief Returns a plan for a 2D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_2d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n0*n1`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n0*(n1/2+1)`.
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
 * @brief Returns a plan for a 2D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_2d` and its variants
 * at different precisions. The API is near identical, but here the relevant
 * data  pointer is of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data. Required size of underlying
 data equal to `n0*(n1/2+1)`.
 * @param out pointer to the start of the output data. Required size of
 underlying data equal to `n0*n1`.
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
 * @brief Returns a plan for a 2D real-to-real
 * transformation of the specified kind.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_2d` and its variants
 * at different precisions. The API is identical.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param in pointer to the start of the input data. Required
 * size of input data equal to `n0*n1`.
 * @param out pointer to the start of the output data. Required
 *  size of output data equal to `n0*n1`.
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
 * @brief Returns a plan for a 3D complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs>
 *

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data. Required size of
 * input data equal to `n0*n1*n2`.
 * @param out pointer to the start of the output data.  Required size
 * of output data equal to `n0*n1*n2`.
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
 * @brief Returns a plan for a 3D real-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_r2c_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data. Required
 *  size of input data equal to `n0*n1*n2`.
 * @param out pointer to the start of the output data. Required size of
 * output data equal to `n0*n1*(n2/2+1)`.
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
 * @brief Returns a plan for a 3D complex-to-real
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft_c2r_3d` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data. Required
 *  size of input data equal to `n0*n1*(n2/2+1)`.
 * @param out pointer to the start of the output data. Required
 * size of output data equal to `n0*n1*n2`.
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
 * @brief Returns a plan for a 3D real-to-real
 * transformation of the specified kinds.
 *
 * This function is a simple wrapper for `fftw_plan_r2r_3d` and its variants
 * at different precisions. The API is identical.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html#Real_002ddata-DFTs>
 *
 * <https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format>

 *
 *
 * @tparam Real precision of the data.
 * @param n0 size along zeroth-dimension.
 * @param n1 size along first-dimension.
 * @param n2 size along second-dimension.
 * @param in pointer to the start of the input data. Required
 * size of input data equal to `n0*n1*n2`.
 * @param out pointer to the start of the output data. Required
 *  size of output data equal to `n0*n1*n2`.
 * @param kind0 transformation type along zeroth-dimension.
 * @param kind1 transformation type along first-dimension.
 * @param kind2 transformation type along second-dimension.
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
 * @briefReturns a plan for a multi-dimensional complex-to-complex
 * transformation.
 *
 * This function is a simple wrapper for `fftw_plan_dft` and its variants
 * at different precisions. The API is near identical, but here the data
 * pointers are of `std::complex<Real>*` type.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * For further information see:
 *
 * <https://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs>
 *
 *
 * @tparam Real precision of the data.
 * @param rank dimension of the of the data (i.e., the number of axes
 * it is defined along).
 * @param n pointer to the sizes along each axis. Required size of underlying
 * data equal to rank.
 * @param in pointer to the start of the input data. Requires size of underlying
 * data equal to
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
 * @brief Returns a plan for a multi-dimensional real-to-complex
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
 * @param n pointer to the sizes along each axis.
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
 * @brief Returns a plan for a multi-dimensional complex-to-real
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
 * @param n pointer to the sizes along each axis.
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
 * @brief Returns a plan for a multi-dimensional real-to-real
 * transformation of the specifcied kinds.
 *
 * This function is a simple wrapper for `fftw_plan_r2r` and its variants
 * at different precisions. The API is identical.
 *
 * It is assumed that the data is stored contiguously in **row-major** order.
 *
 * @tparam Real precision of the data.
 * @param rank  dimension of the data.
 * @param n pointer to the sizes along each axis.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 * @param kind pointer to the transformation type along each axis.
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

/**
 * @brief Returns a plan for many complex-to-complex transforms
 * using the advanced interface from `fftw3` to specify the data layout.
 *
 *
 * This is a simple wrapper for `fftw_plan_many_dft` and its variants with
 * different precisions. For full details of the data layout see:
 *
 * <https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html#Advanced-Complex-DFTs>
 *
 * In brief, the offset within the data of the `j`th point for the `k`th
 * transform equal to `j*stride+k*dist`, while data along the `i`th axis are
 * embeded within a large block of size `embed[i]`.
 *
 *
 * @tparam Real precision of the transformation.
 * @param rank dimension of the data.
 * @param n pointer to the sizes along each axis.
 * @param howMany number of data blocks to transform.
 * @param in pointer to the start of the input data.
 * @param inEmbed pointer to sizes along each axis in which the input data are
 * embeded.
 * @param inStride stride of the input data, this meaning the distance between
 * adjacent points.
 * @param inDist distance between each subsequent block for the input data.
 * @param out pointer to the start of the output data.
 * @param outEmbed pointer to sizes along each axis in which the output data are
 * embeded.
 * @param outStride  stride of the output data, this meaning the distance
 * between adjacent points.
 * @param outDist distance between each subsequent block for the output data.
 * @param sign sign for the transformation.
 * @param flag planner flag
 * @return auto `fftw3` plan for the chosen precision.
 */
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

/**
 * @brief Returns a plan for many real-to-complex transforms
 * using the advanced interface from `fftw3` to specify the data layout.
 *
 *
 * This is a simple wrapper for `fftw_plan_many_dft_r2c` and its variants with
 * different precisions.
 *
 *
 * In brief, the offset within the data of the `j`th point for the `k`th
 * transform equal to `j*stride+k*dist`, while data along the `i`th axis are
 * embeded within a large block of size `embed[i]`.
 *
 *
 * @tparam Real precision of the transformation.
 * @param rank dimension of the data.
 * @param n pointer to the sizes along each axis.
 * @param howMany number of data blocks to transform.
 * @param in pointer to the start of the input data.
 * @param inEmbed pointer to sizes along each axis in which the input data are
 * embeded.
 * @param inStride stride of the input data, this meaning the distance between
 * adjacent points.
 * @param inDist distance between each subsequent block for the input data.
 * @param out pointer to the start of the output data.
 * @param outEmbed pointer to sizes along each axis in which the output data are
 * embeded.
 * @param outStride  stride of the output data, this meaning the distance
 * between adjacent points.
 * @param outDist distance between each subsequent block for the output data.
 * @param flag planner flag
 * @return auto `fftw3` plan for the chosen precision.
 */
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

/**
 * @brief Returns a plan for many complex-to-real transforms
 * using the advanced interface from `fftw3` to specify the data layout.
 *
 *
 * This is a simple wrapper for `fftw_plan_many_dft_c2r` and its variants with
 * different precisions.
 *
 *
 * In brief, the offset within the data of the `j`th point for the `k`th
 * transform equal to `j*stride+k*dist`, while data along the `i`th axis are
 * embeded within a large block of size `embed[i]`.
 *
 *
 * @tparam Real precision of the transformation.
 * @param rank dimension of the data.
 * @param n pointer to the sizes along each axis.
 * @param howMany number of data blocks to transform.
 * @param in pointer to the start of the input data.
 * @param inEmbed pointer to sizes along each axis in which the input data are
 * embeded.
 * @param inStride stride of the input data, this meaning the distance between
 * adjacent points.
 * @param inDist distance between each subsequent block for the input data.
 * @param out pointer to the start of the output data.
 * @param outEmbed pointer to sizes along each axis in which the output data are
 * embeded.
 * @param outStride  stride of the output data, this meaning the distance
 * between adjacent points.
 * @param outDist distance between each subsequent block for the output data.
 * @param flag planner flag
 * @return auto `fftw3` plan for the chosen precision.
 */
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

/**
 * @brief Returns a plan for many real-to-real transforms of the specificied
 * kinds using the advanced interface from `fftw3` to specify the data layout.
 *
 *
 * This is a simple wrapper for `fftw_plan_many_r2r` and its variants with
 * different precisions.
 *
 *
 * In brief, the offset within the data of the `j`th point for the `k`th
 * transform equal to `j*stride+k*dist`, while data along the `i`th axis are
 * embeded within a large block of size `embed[i]`.
 *
 *
 * @tparam Real precision of the transformation.
 * @param rank dimension of the data.
 * @param n pointer to the sizes along each axis.
 * @param howMany number of data blocks to transform.
 * @param in pointer to the start of the input data.
 * @param inEmbed pointer to sizes along each axis in which the input data are
 * embeded.
 * @param inStride stride of the input data, this meaning the distance between
 * adjacent points.
 * @param inDist distance between each subsequent block for the input data.
 * @param out pointer to the start of the output data.
 * @param outEmbed pointer to sizes along each axis in which the output data are
 * embeded.
 * @param outStride  stride of the output data, this meaning the distance
 * between adjacent points.
 * @param outDist distance between each subsequent block for the output data.
 * @param kind pointers to the transformation types along each axis.
 * @param flag planner flag
 * @return auto `fftw3` plan for the chosen precision.
 */
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

/**
 * @brief Frees memory associated with the input plan.
 *
 * This a wrapper for `fftw_destroy` and its variants at different precisions.
 */
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

/**
 * @brief Executes a transformation plan.
 *
 * This a wrapper for `fftw_execute` and its variants at different precisions.
 *
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
 * @brief Executes a complex-to-complex plan on new data.
 *
 * This is a wrapper for `fftw_execute_dft` and its variants at different
 * precisions.
 *
 * @param plan complex-to-complex plan for data of the appropriate type.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 */
template <typename PlanType, std::floating_point Real>
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
 * @brief Executes a real-to-complex plan on new data.
 *
 * This is a wrapper for `fftw_execute_dft_r2c` and its variants at different
 * precisions.
 *
 * @param plan real-to-complex plan for data of the appropriate type.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 */
template <typename PlanType, std::floating_point Real>
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
 * @brief Executes a complex-to-real plan on new data.
 *
 * This is a wrapper for `fftw_execute_dft_c2r` and its variants at different
 * precisions.
 *
 * @param plan complex-to-real plan for data of the appropriate type.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 */
template <typename PlanType, std::floating_point Real>
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
 * @brief Executes a real-to-real plan on new data.
 *
 * This is a wrapper for `fftw_execute_r2r` and its variants at different
 * precisions.
 *
 * @param plan real-to-real plan for data of the appropriate type.
 * @param in pointer to the start of the input data.
 * @param out pointer to the start of the output data.
 */
template <typename PlanType, std::floating_point Real>
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
