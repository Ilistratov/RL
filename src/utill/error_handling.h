#pragma once

#include <stdexcept>
#include <string_view>

#include "utill/logger.h"

namespace utill {

class DeathLogEntry {
  std::ostringstream message_;
  bool cond_ = true;

 public:
  DeathLogEntry(std::string_view source_location, bool cond);
  DeathLogEntry(const DeathLogEntry&) = delete;
  void operator=(const DeathLogEntry&) = delete;

  template <typename T>
  DeathLogEntry& operator<<(const T& entry_contents) {
    if (!cond_) {
      message_ << entry_contents;
    }
    return *this;
  }

  ~DeathLogEntry();
};

#define PANIC                                                                 \
  utill::DeathLogEntry(std::string(__FILE__).substr(SOURCE_PATH_SIZE) + ":" + \
                           std::to_string(__LINE__),                          \
                       false)                                                 \
      << "Panic: "

#define CHECK(COND)                                                           \
  utill::DeathLogEntry(std::string(__FILE__).substr(SOURCE_PATH_SIZE) + ":" + \
                           std::to_string(__LINE__),                          \
                       COND)                                                  \
      << "Check " #COND " failed\n"

#define CHECK_VK_RESULT(RESULT)         \
  CHECK(RESULT == vk::Result::eSuccess) \
      << "Error: " << vk::to_string(RESULT) << " "

#ifdef NDEBUG
#define DCHECK(COND) \
  {}
#else
#define DCHECK(COND) CHECK(COND)
#endif

}  // namespace utill
