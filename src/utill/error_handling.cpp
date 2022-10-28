#include "utill/error_handling.h"
#include <memory>
#include "utill/logger.h"

namespace utill {

DeathLogEntry::DeathLogEntry(const char* src_location, uint32_t ln)
    : LogEntry(src_location, ln) {}

DeathLogEntry::~DeathLogEntry() {
  Logger::GetGlobal().Log(message_.str());
  _exit(1);
}

}  // namespace utill
