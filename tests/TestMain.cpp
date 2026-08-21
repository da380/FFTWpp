#include <gtest/gtest.h>

#include <FFTWpp/Ranges>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto result = RUN_ALL_TESTS();

  // One of the few places CleanUp is worth calling: the sanitizer workflow
  // wants FFTW's still-reachable state released so that anything left over is
  // this library's doing. It is safe here only because every test-local plan
  // has been destroyed by now, which CleanUp itself verifies.
  FFTWpp::CleanUp();
  return result;
}
