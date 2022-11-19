#pragma once

namespace gpu_resources {

namespace error_messages {

extern const char* kErrNotInitialized;
extern const char* kErrAlreadyInitialized;
extern const char* kErrCantBeEmpty;
extern const char* kErrMemoryNotRequested;
extern const char* kErrMemoryAlreadyRequested;
extern const char* kErrMemoryNotAllocated;
extern const char* kErrMemoryNotMapped;
extern const char* kErrResourceIsNull;
extern const char* kErrInvalidResourceIdx;
extern const char* kErrInvalidPassIdx;
extern const char* kErrLayoutsIncompatible;
extern const char* kErrNotEnoughSpace;
extern const char* kErrSyncronizerNotProvided;

}  // namespace error_messages

}  // namespace gpu_resources
