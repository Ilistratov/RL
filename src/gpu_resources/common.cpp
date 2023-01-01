#include "gpu_resources/common.h"

namespace gpu_resources {

namespace error_messages {

const char* kErrNotInitialized = "resource is not initialized";
const char* kErrAlreadyInitialized = "resource is already initialized";
const char* kErrCantBeEmpty = "can not create empty resource";
const char* kErrMemoryNotRequested = "device memory was not requested";
const char* kErrMemoryAlreadyRequested = "device memory is already requested";
const char* kErrMemoryNotAllocated = "device memory was not allocated";
const char* kErrMemoryNotMapped = "device memory was not mapped to host";
const char* kErrResourceIsNull = "passed resource is null";
const char* kErrInvalidResourceIdx = "passed resource has invalid idx";
const char* kErrLayoutsIncompatible =
    "two accesses conflict due to different expected image layouts";
const char* kErrInvalidPassIdx = "invalid pass idx";
const char* kErrNotEnoughSpace = "not enough space";
const char* kErrSyncronizerNotProvided =
    "pass access syncronizer was not provided";

}  // namespace error_messages

}  // namespace gpu_resources
