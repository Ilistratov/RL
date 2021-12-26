#include <iostream>

#include "base/base.h"

int main() {
  base::Base::Get().Init(base::DEFAULT_BASE_CONFIG);;
  std::cout << "Hello world!\n";
}
