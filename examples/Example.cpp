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

  auto type = Transformation<HC2R>(Forward);

  std::cout << type() << std::endl;
}
