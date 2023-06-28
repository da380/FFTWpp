
#include <complex>
#include <iostream>
#include <iterator>
#include <vector>

#include "FFTW.h"

#include <limits>


int main()
{

  using Precision = double;
  using Complex = std::complex<Precision>;
  using RealVector = std::vector<Precision>;
  using RealVectorIterator = std::vector<Precision>::iterator;
  using ComplexVector = std::vector<Complex>;
  using ComplexVectorIterator = std::vector<Complex>::iterator;
  
  std::cout << FFTW::RealIterator<ComplexVectorIterator> << std::endl;
  std::cout << FFTW::ComplexIterator<ComplexVectorIterator> << std::endl;  


  std::cout << std::numeric_limits<float>::epsilon() << std::endl;
  std::cout << std::numeric_limits<double>::epsilon() << std::endl;
  std::cout << std::numeric_limits<long double>::epsilon() << std::endl;  

  
}

