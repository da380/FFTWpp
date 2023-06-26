#include <iostream>
#include <complex>
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>
#include <chrono>

#include "FFTW.h"


int main(){


  using Float = double;
  using Complex = std::complex<Float>;
  using Vector = FFTW::vector<Complex>;
  
  int n = 512;



  Vector in(n),out(n),check(n);

    
  FFTW::Plan<Float> forward_plan(in.begin(),in.end(),out.begin(),
				  FFTW::DirectionFlag::Forward,
				  FFTW::PlanFlag::Measure);
  


  
  FFTW::Plan<Float> backward_plan(out.begin(),out.end(),check.begin(),
				  FFTW::DirectionFlag::Backward,
				  FFTW::PlanFlag::Measure);



  for (int i = 0; i < n; i++)
    in[i] = {2.*i,3.*i};
  

  forward_plan.execute();
  backward_plan.execute();



  // Normalise the inverse transformation.
  auto norm = static_cast<Float>(1)/static_cast<Float>(n);
  std::transform(check.cbegin(),check.cend(),check.begin(),
  		 [norm](Complex x) -> Complex {return x*norm;});

  // Print out the residuals.
  for(int i = 0; i < n; i++)
    {
      std::cout <<  in[i]- check[i]   << std::endl;
    }


  
}
