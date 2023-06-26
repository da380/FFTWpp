#ifndef FFTWSimple_GUARD_H
#define FFTWSimple_GUARD_H

#include "FFTWConcepts.h"
#include "FFTWPlan.h"

namespace FFTW {


  template<typename ComplexContainer>
  ComplexContainer FFT1D(ComplexContainer& in)
  {
    using Float = typename ComplexContainer::value_type::value_type;
    int n = in.size();
    ComplexContainer out(n);
    Plan<Float> plan(in.begin(),in.end(),out.begin(),
		     DirectionFlag::Forward,
		     PlanFlag::Estimate);
    plan.execute();
    return out;
  }


  template<typename ComplexContainer>
  ComplexContainer IFFT1D(ComplexContainer& in)
  {
    using Float = typename ComplexContainer::value_type::value_type;
    int n = in.size();
    ComplexContainer out(n);
    Plan<Float> plan(in.begin(),in.end(),out.begin(),
		     DirectionFlag::Backward,
		     PlanFlag::Estimate);
    plan.execute();
    return out;
  }

  
}

#endif // FFTWSimple_GUARD_H
