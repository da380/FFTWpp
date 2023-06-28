#include <gtest/gtest.h>

#include "FFTW1DRealTest.h"
#include "FFTW1DTest.h"

// Tests for 1D complex transformations

TEST(FFTW1DTest, Float) { EXPECT_EQ(0, FFTW1DTest<float>()); }

TEST(FFTW1DTest, FloatNewData) { EXPECT_EQ(0, FFTW1DTest<float>(true)); }

TEST(FFTW1DTest, Double) { EXPECT_EQ(0, FFTW1DTest<double>()); }

TEST(FFTW1DTest, DoubleNewData) { EXPECT_EQ(0, FFTW1DTest<double>(true)); }

TEST(FFTW1DTest, LongDouble) { EXPECT_EQ(0, FFTW1DTest<long double>()); }

TEST(FFTW1DTest, LongDoubleNewData) {
  EXPECT_EQ(0, FFTW1DTest<long double>(true));
}

// Tests for 1D real-to-complex transformations.

TEST(FFTW1DRealTest, Float) { EXPECT_EQ(0, FFTW1DRealTest<float>()); }

TEST(FFTW1DRealTest, FloatNewData) {
  EXPECT_EQ(0, FFTW1DRealTest<float>(true));
}

TEST(FFTW1DRealTest, Double) { EXPECT_EQ(0, FFTW1DRealTest<double>()); }

TEST(FFTW1DRealTest, DoubleNewData) {
  EXPECT_EQ(0, FFTW1DRealTest<double>(true));
}

TEST(FFTW1DRealTest, LongDouble) {
  EXPECT_EQ(0, FFTW1DRealTest<long double>());
}

TEST(FFTW1DRealTest, LongDoubleNewData) {
  EXPECT_EQ(0, FFTW1DRealTest<long double>(true));
}
