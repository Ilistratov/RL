#include "utill/error_handling.h"

namespace utill {

DeathLogEntry::DeathLogEntry(const std::string_view source_location, bool cond)
    : cond_(cond) {
  if (cond) {
    return;
  }
  Logger::AddInitialLogTag(message_, source_location);
}

DeathLogEntry::~DeathLogEntry() {
  if (cond_) {
    return;
  }
  if (!message_.str().empty()) {
    message_ << '\n';
    GlobalLogger.Log(message_.str());
  }
#ifdef NDEBUG
  throw std::runtime_error("Unrecoverable error");
#else
  _exit(1);
#endif
}

}  // namespace utill
