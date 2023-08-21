#include <FFTWpp/All>
#include <iostream>

int main() {
  using namespace FFTWpp;

  auto flag1 = Estimate;
  auto flag2 = Measure;

  auto flag3 = flag1 | flag2 | flag2;

  std::cout << flag1.Convert() << std::endl;
}
