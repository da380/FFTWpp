#ifndef FFTWConcepts_GUARD_H
#define FFTWConcepts_GUARD_H

#include <concepts>

namespace FFTW {

// Concepts for floating point types.
template <typename Precision>
concept IsSingle = std::same_as<Precision, float>;

template <typename Precision>
concept IsDouble = std::same_as<Precision, double>;

template <typename Precision>
concept IsQuadruple = std::same_as<Precision, long>;

// Concepts for real numbers.
template <typename T>
concept Real = std::floating_point<T>;

// Concepts for complex numbers.
template <typename T>
struct ComplexHelper : std::false_type {};

template <typename T>
struct ComplexHelper<std::complex<T>> : std::true_type {};

template <typename T>
concept Complex = requires() {
  requires ComplexHelper<T>::value;
  requires std::floating_point<typename T::value_type>;
};

// Concepts for real iterators
  template <typename Iter>
  concept RealIterator = requires(){    
    requires std::contiguous_iterator<Iter>;
    requires Real<typename Iter::value_type>;
  };
  
  template <typename Iter, typename Precision>
  concept RealIteratorWithPrecision = requires(){    
    requires RealIterator<Iter>;
    requires Real<Precision>;    
    requires std::same_as<typename Iter::value_type,Precision>;
  };



// Concepts for complex iterators
  template <typename Iter>
  concept ComplexIterator = requires(){    
    requires std::contiguous_iterator<Iter>;
    requires Complex<typename Iter::value_type>;
  };
  
  template <typename Iter, typename Precision>
  concept ComplexIteratorWithPrecision = requires(){    
    requires ComplexIterator<Iter>;
    requires Real<Precision>;    
    requires std::same_as<typename Iter::value_type::value_type,Precision>;
  };

}  // namespace FFTW

#endif  // FFTWConcepts_GUARD_H
