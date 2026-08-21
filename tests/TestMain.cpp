#include <gtest/gtest.h>

#include <FFTWpp/Ranges>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto result = RUN_ALL_TESTS();
  FFTWpp::CleanUp();
  return result;
}
