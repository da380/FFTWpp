
#include <complex>
#include <iostream>
#include <iterator>
#include <list>
#include <vector>

#include "FFTW.h"


template <FFTW::ScalarIterator It>
void func(It in)
{

  std::cout << "Hello\n";
  
}


int main()
{

  using Real = double;
  using Complex = std::complex<Real>;
  using RealVector = std::vector<Real>;
  using RealVectorIterator = std::vector<Real>::iterator;
  using ComplexVector = std::vector<Complex>;
  using ComplexVectorIterator = std::vector<Complex>::iterator;

  using std::cout;
  using std::endl;
  




  
}

