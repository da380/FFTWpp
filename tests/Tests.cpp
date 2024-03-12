#include <gtest/gtest.h>

#include "Test1D.h"

// 1D C2C tests
TEST(Test1DC2C, FLOAT) {
  using Complex = std::complex<float>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DC2C, DOUBLE) {
  using Complex = std::complex<double>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DC2C, LONGDOUBLE) {
  using Complex = std::complex<long double>;
  auto result = Test1D<Complex, Complex>();
  EXPECT_TRUE(result);
}

// 1D R2C tests
TEST(Test1DR2C, FLOAT) {
  using Real = float;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2C, DOUBLE) {
  using Real = double;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2C, LONGDOUBLE) {
  using Real = long double;
  using Complex = std::complex<Real>;
  auto result = Test1D<Real, Complex>();
  EXPECT_TRUE(result);
}

// 1D R2R tests
TEST(Test1DR2R, FLOAT) {
  using Real = float;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2R, DOUBLE) {
  using Real = double;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}

TEST(Test1DR2R, LONGDOUBLE) {
  using Real = long double;
  auto result = Test1D<Real, Real>();
  EXPECT_TRUE(result);
}
