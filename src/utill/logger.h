#pragma once

#include <chrono>
#include <source_location>
#include <sstream>
#include <vector>

namespace utill {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << "[";
  bool first = true;
  for (const auto& el : vec) {
    if (!first) {
      out << ", ";
    }
    out << el;
    first = false;
  }
  out << "]";
  return out;
}

class Logger;

class LogEntry {
 protected:
  std::ostringstream message_;

 public:
  LogEntry(const char* src_location, uint32_t ln);
  LogEntry(const LogEntry&) = delete;
  void operator=(const LogEntry&) = delete;

  template <typename T>
  LogEntry& operator<<(const T& entry_contents) {
    message_ << entry_contents;
    return *this;
  }

  ~LogEntry();
};

class Voidify {
 public:
  Voidify() = default;
  template <typename T>
  Voidify& operator<<(const T&) {
    return *this;
  }
  template <typename T>
  void operator&(const T&) {}
};

class Logger {
  std::chrono::system_clock::time_point start_;
  Logger();

 public:
  static Logger& GetGlobal();

  std::chrono::system_clock::time_point GetStartTime() const;
  void Log(const std::string& message);
};

}  // namespace utill

#define LAZY_STREAM(cond, stream) \
  !(cond) ? (void)0 : utill::Voidify() & (stream)

#define LOG_STREAM utill::LogEntry(__FILE__, __LINE__)

#define LOG LAZY_STREAM(true, LOG_STREAM)

#ifdef NDEBUG
#define DLOG utill::Voidify()
#else
#define DLOG LAZY_STREAM(true, LOG_STREAM)
#endif
