#ifndef FFTWallocator_GUARD_H
#define FFTWallocator_GUARD_H

#include <concepts>
#include <memory>
#include <vector>

#include "fftw3.h"

namespace FFTW {

// Define a custom allocator using the fftw3 versions of malloc and free.
template <typename T>
struct allocator {
  typedef T value_type;
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
template <std::floating_point Float>
auto ComplexCast(std::complex<Float>* z) {
  if constexpr (IsSingle<Float>) {
    return reinterpret_cast<fftwf_complex*>(z);
  }
  if constexpr (IsDouble<Float>) {
    return reinterpret_cast<fftw_complex*>(z);
  }
  if constexpr (IsLongDouble<Float>) {
    return reinterpret_cast<fftwl_complex*>(z);
  }
}

// Clear up remaining memory. To be, optionally, called only when all plans
// have gone out of scope.
void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

}  // namespace FFTW

#endif  // FFTWallocator_GUARD_H
