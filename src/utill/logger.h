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
  Logger* dst_logger_ = nullptr;
  std::ostringstream message_;

 public:
  LogEntry(Logger* dst_logger, std::string_view source_location);
  LogEntry(LogEntry&& other);

  template <typename T>
  LogEntry& operator<<(const T& entry_contents) {
    message_ << entry_contents;
    return *this;
  }

  ~LogEntry();
};

class Logger {
  std::chrono::system_clock::time_point start_;

 public:
  Logger();

  LogEntry NewEntry(std::string_view location_string);
  std::chrono::system_clock::time_point GetStartTime() const;
  void Log(const std::string& message);
};

extern Logger GlobalLogger;

}  // namespace utill

#define LOG(TAG)                                             \
  utill::GlobalLogger.NewEntry(                              \
      std::string(__FILE__).substr(SOURCE_PATH_SIZE) + ":" + \
      std::to_string(__LINE__))                              \
      << "[" #TAG "] "
