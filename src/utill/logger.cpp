#include "utill/logger.h"

#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "common_def.h"

namespace utill {

namespace {

std::string GetTimeStr(uint64_t miliseconds_since_start) {
  uint64_t seconds_since_start = miliseconds_since_start / 1000;
  miliseconds_since_start %= 1000;
  uint64_t minutes_since_start = seconds_since_start / 60;
  seconds_since_start %= 60;
  uint64_t hours_since_start = minutes_since_start / 60;
  minutes_since_start %= 60;

  std::string res;
  res.resize(sizeof("0000:00:00:000"));
  sprintf_s(res.data(), res.size(), "%04u:%02u:%02u:%03u", hours_since_start,
            minutes_since_start, seconds_since_start, miliseconds_since_start);
  return res;
}

std::string GetSrcLocationStr(const char* raw_src, uint32_t line) {
  std::string res(raw_src + RL::BASE_PATH_LEN);
  res += ':';
  res += std::to_string(line);
  return res;
}

}  // namespace

LogEntry::LogEntry(const char* src_file, uint32_t src_line) {
  int64_t miliseconds_since_start =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - Logger::GetGlobal().GetStartTime())
          .count();
  message_ << "[" << GetTimeStr(miliseconds_since_start) << "]";
  message_ << "[" << GetSrcLocationStr(src_file, src_line) << "] ";
}

LogEntry::~LogEntry() {
  if (!message_.str().empty()) {
    message_ << '\n';
    Logger::GetGlobal().Log(message_.str());
  }
}

Logger::Logger() {
  start_ = std::chrono::system_clock::now();
}

Logger& Logger::GetGlobal() {
  static Logger logger;
  return logger;
}

std::chrono::system_clock::time_point Logger::GetStartTime() const {
  return start_;
}

void Logger::Log(const std::string& message) {
  std::cout << message;
}

}  // namespace utill
