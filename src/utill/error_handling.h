#pragma once

#include <stdexcept>

namespace utill {

#define PANIC(MSG)                                          \
  LOG << "Unrecoverable error: " << MSG << "\nTerminating"; \
  throw std::runtime_error("Unrecoverable error");

#define CHECK(COND)                           \
  if (!COND) {                                \
    PANIC("Condition: " #COND " unsatisfied") \
  }

#ifdef NDEBUG
#define DCHECK(COND) \
  {}
#else
#define DCHECK(COND) CHECK(COND)
#endif

}  // namespace utill
