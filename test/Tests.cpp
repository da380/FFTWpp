#include <gtest/gtest.h>


#include "Test1DC2C.h"
#include "Test1DR2C.h"
#include "Test1DR2R.h"

// Tests for 1D complex transformations

TEST(Test1DC2C, Float) { EXPECT_EQ(0, Test1DC2C<float>()); }

TEST(Test1DC2C, FloatNewData) { EXPECT_EQ(0, Test1DC2C<float>(true)); }

TEST(Test1DC2C, Double) { EXPECT_EQ(0, Test1DC2C<double>()); }

TEST(Test1DC2C, DoubleNewData) { EXPECT_EQ(0, Test1DC2C<double>(true)); }

TEST(Test1DC2C, LongDouble) { EXPECT_EQ(0, Test1DC2C<long double>()); }

TEST(Test1DC2C, LongDoubleNewData) {
  EXPECT_EQ(0, Test1DC2C<long double>(true));
}

// Tests for 1D real-to-complex transformations.

TEST(Test1DR2C, Float) { EXPECT_EQ(0, Test1DR2C<float>()); }

TEST(Test1DR2C, FloatNewData) {
  EXPECT_EQ(0, Test1DR2C<float>(true));
}

TEST(Test1DR2C, Double) { EXPECT_EQ(0, Test1DR2C<double>()); }

TEST(Test1DR2C, DoubleNewData) {
  EXPECT_EQ(0, Test1DR2C<double>(true));
}

TEST(Test1DR2C, LongDouble) {
  EXPECT_EQ(0, Test1DR2C<long double>());
}

TEST(Test1DR2C, LongDoubleNewData) {
  EXPECT_EQ(0, Test1DR2C<long double>(true));
}



// Tests for 1D real-to-real transformations.

TEST(Test1DR2R, Float) { EXPECT_EQ(0, Test1DR2R<float>()); }

TEST(Test1DR2R, FloatNewData) {
  EXPECT_EQ(0, Test1DR2R<float>(true));
}

TEST(Test1DR2R, Double) { EXPECT_EQ(0, Test1DR2R<double>()); }

TEST(Test1DR2R, DoubleNewData) {
  EXPECT_EQ(0, Test1DR2R<double>(true));
}

TEST(Test1DR2R, LongDouble) {
  EXPECT_EQ(0, Test1DR2R<long double>());
}

TEST(Test1DR2R, LongDoubleNewData) {
  EXPECT_EQ(0, Test1DR2R<long double>(true));
}
