#ifndef FFTWCleanUp_GUARD_H
#define FFTWCleanUp_GUARD_H

#include "fftw3.h"

namespace{


  void FFTWCleanUP()
  {
    fftwf_cleanup();
    fftw_cleanup();
    fftwl_cleanup();
  }

}

#endif // FFTWCleanUp_GUARD_H


