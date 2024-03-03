#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <typeinfo>
#include <vector>

int main() {
  using namespace FFTWpp::Testing;

  std::cout << Forward() << std::endl;

  auto flag = Estimate | WisdomOnly;
}
