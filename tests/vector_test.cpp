#include <iostream>
#include <complex>
#include <vector>

#include "FFTW.h"


int main(){

  FFTW::vector<double> x(10);

  for(auto val : x)
    std::cout << val << std::endl;


  FFTW::FFTWDirectionFlag direction = FFTW::FFTWDirectionFlag::Backward;


  
}
