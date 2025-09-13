#pragma once

#include <string>

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace starpu_server {

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) : label_(name)
  {
#ifdef HAVE_NVTX
    nvtxRangePushA(label_.c_str());
#endif
  }

  explicit NvtxRange(std::string name) : label_(std::move(name))
  {
#ifdef HAVE_NVTX
    nvtxRangePushA(label_.c_str());
#endif
  }

  ~NvtxRange()
  {
#ifdef HAVE_NVTX
    nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange&) = delete;
  auto operator=(const NvtxRange&) -> NvtxRange& = delete;

 private:
  std::string label_{};
};

}  // namespace starpu_server
