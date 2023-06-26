#ifndef FFTWAllocator_GUARD_H
#define FFTWAllocator_GUARD_H

#include <memory>
#include <vector>

#include "fftw3.h"

namespace FFTW {

// Define a custom allocator using the fftw3 versions of malloc and free.
template <typename T>
struct FFTWAllocator {
  typedef T value_type;
  FFTWAllocator() noexcept {}
  template <class U>
  FFTWAllocator(const FFTWAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    return static_cast<T*>(fftw_malloc(sizeof(T) * n));
  }
  void deallocate(T* p, std::size_t n) { fftw_free(p); }
};

template <class T, class U>
constexpr bool operator==(const FFTWAllocator<T>&,
                          const FFTWAllocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const FFTWAllocator<T>&,
                          const FFTWAllocator<U>&) noexcept {
  return false;
}

// Type alias for a std::vector using the custom allocator.
template <typename T>
using vector = std::vector<T, FFTWAllocator<T>>;

}  // namespace FFTW

#endif  // FFTWAllocator_GUARD_H
