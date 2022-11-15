#include <vector>

namespace utill {

template <typename T>
class RingBuffer {
  std::vector<T> data_;
  size_t start_ = 0;
  size_t size_ = 0;

  size_t GetAbsoluteElementInd(size_t ind) {
    return (start_ + ind) % data_.size();
  }

 public:
  RingBuffer(size_t cap) : data_(cap) {}

  bool PushBack(T&& val) {
    if (size_ == data_.size()) {
      return false;
    }
    data_[GetAbsoluteElementInd(size_)] = std::move(val);
    return true;
  }

  T PopFront() {
    --size_;
    return data[start_++];
  }

  T& GetFront() { return data[start_]; }
  T& GetBack() { return data[GetAbsoluteElementInd(size_ - 1)]; }
  size_t GetSize() const { return size_; }
  size_t GetCapacity() const { return data_.size(); }
  bool IsFull() const { return GetSize() == GetCapacity(); }
};

}  // namespace utill
