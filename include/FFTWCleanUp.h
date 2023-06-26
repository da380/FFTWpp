#ifndef FFTWCleanUp_GUARD_H
#define FFTWCleanUp_GUARD_H

#include "fftw3.h"

namespace FFTW {

void CleanUp() {
  fftwf_cleanup();
  fftw_cleanup();
  fftwl_cleanup();
}

}  // namespace FFTW

#endif  // FFTWCleanUp_GUARD_H
