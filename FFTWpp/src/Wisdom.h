#ifndef FFTWPP_WISDOM_GUARD_H
#define FFTWPP_WISDOM_GUARD_H

#include <cassert>
#include <string>

#include "fftw3.h"

void ExportWisdom(const std::string& filename) {
  int io = fftw_export_wisdom_to_filename(filename.c_str());
  assert(io == 0);
}

void ImportWisdom(const std::string& filename) {
  int io = fftw_import_wisdom_from_filename(filename.c_str());
  assert(io == 0);
}

void ForgetWisdom() { fftw_forget_wisdom(); }

#endif  // FFTWPP_WISDOM_GUARD_H
