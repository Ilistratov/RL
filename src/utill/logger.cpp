#include "utill/logger.h"

#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace utill {

LogEntry::LogEntry(const std::string_view source_location) {
  int64_t miliseconds_since_start =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - GlobalLogger.GetStartTime())
          .count();
  int64_t seconds_since_start = miliseconds_since_start / 1000;
  int64_t minutes_since_start = seconds_since_start / 60;
  int64_t hours_since_start = minutes_since_start / 60;
  message_ << "[" << hours_since_start << ":" << minutes_since_start << ":"
           << seconds_since_start << ":" << miliseconds_since_start << "]";
  message_ << "[" << source_location << "] ";
}

LogEntry::LogEntry(LogEntry&& other) {
  message_.swap(other.message_);
}

LogEntry::~LogEntry() {
  if (!message_.str().empty()) {
    GlobalLogger.Log(message_.str());
  }
}

Logger::Logger() {
  start_ = std::chrono::system_clock::now();
}

std::chrono::system_clock::time_point Logger::GetStartTime() const {
  return start_;
}

void Logger::Log(const std::string& message) {
  std::cout << message << '\n';
}

Logger GlobalLogger;

}  // namespace utill
