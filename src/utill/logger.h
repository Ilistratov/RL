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
  std::ostringstream message_;

 public:
  LogEntry(std::string_view source_location);
  LogEntry(const LogEntry&) = delete;
  void operator=(const LogEntry&) = delete;

  template <typename T>
  LogEntry& operator<<(const T& entry_contents) {
    message_ << entry_contents;
    return *this;
  }

  ~LogEntry();
};

class EmptyEntry {
 public:
  template <typename T>
  LogEntry& operator<<(const T&) {
    return *this;
  }
};

class Logger {
  std::chrono::system_clock::time_point start_;

 public:
  Logger();

  std::chrono::system_clock::time_point GetStartTime() const;
  void Log(const std::string& message);
  static void AddInitialLogTag(std::ostringstream& out,
                               const std::string_view source_location);
};

extern Logger GlobalLogger;

}  // namespace utill

#define LOG                                                              \
  utill::LogEntry(std::string(__FILE__).substr(SOURCE_PATH_SIZE) + ":" + \
                  std::to_string(__LINE__))

#ifdef NDEBUG
#define DLOG utill::EmptyEntry()
#else
#define DLOG LOG
#endif
