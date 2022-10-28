#pragma once

#include <stdexcept>
#include <string_view>

#include "utill/logger.h"

namespace utill {

class DeathLogEntry : public LogEntry {
 public:
  DeathLogEntry(const char* src_location, uint32_t ln);
  DeathLogEntry(const DeathLogEntry&) = delete;
  void operator=(const DeathLogEntry&) = delete;
  ~DeathLogEntry();
};

#define DEATH_STREAM utill::DeathLogEntry(__FILE__, __LINE__)

#define CHECK(cond) \
  LAZY_STREAM(!(cond), DEATH_STREAM) << "Check " #cond " failed\n"

#define CHECK_VK_RESULT(RESULT)         \
  CHECK(RESULT == vk::Result::eSuccess) \
      << "Error: " << vk::to_string(RESULT) << " "

#ifdef NDEBUG
#define DCHECK(cond) utill::Voidify()
#else
#define DCHECK(COND) CHECK(COND)
#endif

}  // namespace utill
