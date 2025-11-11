#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "core/input_slot_pool.hpp"
#include "core/starpu_setup.hpp"
#include "support/input_slot_pool_test_hooks.hpp"
#include "support/output_slot_pool_test_hooks.hpp"
#include "test_utils.hpp"
#include "utils/runtime_config.hpp"

namespace {

std::vector<std::byte*> g_observed_base_ptrs;
std::vector<starpu_data_handle_t> g_observed_handles;
bool g_failure_observer_called = false;

int g_succeed_then_fail_register_call_count = 0;
starpu_server::testing::StarpuVectorRegisterFn
    g_succeed_then_fail_previous_hook = nullptr;
std::vector<starpu_data_handle_t>* g_pre_cleanup_registered_handles = nullptr;

std::vector<std::byte*> g_output_observed_base_ptrs;
std::vector<starpu_data_handle_t> g_output_observed_handles;
std::vector<starpu_server::OutputSlotPool::HostBufferInfo>
    g_output_observed_host_buffer_infos;
bool g_output_failure_observer_called = false;

using StarpuInitRawFn = int (*)(struct starpu_conf*);
using StarpuWorkerGetIdHook = int (*)();
using StarpuWorkerGetDevidHook = int (*)(int);
using StarpuCudaGetLocalStreamHook = cudaStream_t (*)();
using HwlocTopologyInitFn = int (*)(hwloc_topology_t*);
using HwlocTopologyLoadFn = int (*)(hwloc_topology_t);
using HwlocTopologyDestroyFn = void (*)(hwloc_topology_t);
using HwlocGetTypeDepthFn = int (*)(hwloc_topology_t, hwloc_obj_type_t);
using HwlocGetDepthTypeFn = hwloc_obj_type_t (*)(hwloc_topology_t, int);
using HwlocGetNbobjsByDepthFn = unsigned (*)(hwloc_topology_t, int);
using HwlocGetObjByDepthFn = hwloc_obj_t (*)(hwloc_topology_t, int, unsigned);
using HwlocBitmapFirstFn = int (*)(hwloc_const_bitmap_t);
using EstimateNonCpuWorkersFn =
    unsigned (*)(const starpu_server::RuntimeConfig&);
using SetenvFn = int (*)(const char*, const char*, int);

bool g_starpu_init_stub_called = false;
bool g_worker_stream_stub_called = false;
StarpuWorkerGetIdHook g_worker_get_id_hook = nullptr;
StarpuWorkerGetDevidHook g_worker_get_devid_hook = nullptr;
StarpuCudaGetLocalStreamHook g_cuda_get_local_stream_hook = nullptr;
SetenvFn g_setenv_override = nullptr;

int g_mock_worker_id = 0;
int g_mock_device_id = 0;
cudaStream_t g_mock_cuda_stream = nullptr;

int
mock_starpu_worker_get_id()
{
  return g_mock_worker_id;
}

int
mock_starpu_worker_get_devid(int /*unused*/)
{
  return g_mock_device_id;
}

cudaStream_t
mock_starpu_cuda_get_local_stream()
{
  return g_mock_cuda_stream;
}

int
starpu_init_stub_for_null(struct starpu_conf*)
{
  g_starpu_init_stub_called = true;
  return 0;
}

int
worker_stream_stub_for_null(
    unsigned int /*device_id*/, int* /*workerids*/,
    enum starpu_worker_archtype /*worker*/)
{
  g_worker_stream_stub_called = true;
  return -1;
}

int
failing_setenv_stub(
    const char* /*name*/, const char* /*value*/, int /*overwrite*/)
{
  errno = EPERM;
  return -1;
}

auto
resolve_real_starpu_worker_get_id() -> StarpuWorkerGetIdHook
{
  static StarpuWorkerGetIdHook fn = []() -> StarpuWorkerGetIdHook {
    auto candidate = reinterpret_cast<StarpuWorkerGetIdHook>(
        dlsym(RTLD_NEXT, "starpu_worker_get_id"));
    if (candidate == nullptr) {
      candidate = reinterpret_cast<StarpuWorkerGetIdHook>(
          dlsym(RTLD_DEFAULT, "starpu_worker_get_id"));
    }
    return candidate;
  }();
  return fn;
}

auto
resolve_real_starpu_worker_get_devid() -> StarpuWorkerGetDevidHook
{
  static StarpuWorkerGetDevidHook fn = []() -> StarpuWorkerGetDevidHook {
    auto candidate = reinterpret_cast<StarpuWorkerGetDevidHook>(
        dlsym(RTLD_NEXT, "starpu_worker_get_devid"));
    if (candidate == nullptr) {
      candidate = reinterpret_cast<StarpuWorkerGetDevidHook>(
          dlsym(RTLD_DEFAULT, "starpu_worker_get_devid"));
    }
    return candidate;
  }();
  return fn;
}

auto
resolve_real_starpu_cuda_get_local_stream() -> StarpuCudaGetLocalStreamHook
{
  static StarpuCudaGetLocalStreamHook fn =
      []() -> StarpuCudaGetLocalStreamHook {
    auto candidate = reinterpret_cast<StarpuCudaGetLocalStreamHook>(
        dlsym(RTLD_NEXT, "starpu_cuda_get_local_stream"));
    if (candidate == nullptr) {
      candidate = reinterpret_cast<StarpuCudaGetLocalStreamHook>(
          dlsym(RTLD_DEFAULT, "starpu_cuda_get_local_stream"));
    }
    return candidate;
  }();
  return fn;
}

auto
resolve_real_hwloc_topology_init() -> HwlocTopologyInitFn
{
  static HwlocTopologyInitFn fn = []() -> HwlocTopologyInitFn {
    return reinterpret_cast<HwlocTopologyInitFn>(
        dlsym(RTLD_NEXT, "hwloc_topology_init"));
  }();
  return fn;
}

auto
resolve_real_hwloc_topology_load() -> HwlocTopologyLoadFn
{
  static HwlocTopologyLoadFn fn = []() -> HwlocTopologyLoadFn {
    return reinterpret_cast<HwlocTopologyLoadFn>(
        dlsym(RTLD_NEXT, "hwloc_topology_load"));
  }();
  return fn;
}

auto
resolve_real_hwloc_topology_destroy() -> HwlocTopologyDestroyFn
{
  static HwlocTopologyDestroyFn fn = []() -> HwlocTopologyDestroyFn {
    return reinterpret_cast<HwlocTopologyDestroyFn>(
        dlsym(RTLD_NEXT, "hwloc_topology_destroy"));
  }();
  return fn;
}

auto
resolve_real_hwloc_get_type_depth() -> HwlocGetTypeDepthFn
{
  static HwlocGetTypeDepthFn fn = []() -> HwlocGetTypeDepthFn {
    return reinterpret_cast<HwlocGetTypeDepthFn>(
        dlsym(RTLD_NEXT, "hwloc_get_type_depth"));
  }();
  return fn;
}

auto
resolve_real_hwloc_get_depth_type() -> HwlocGetDepthTypeFn
{
  static HwlocGetDepthTypeFn fn = []() -> HwlocGetDepthTypeFn {
    return reinterpret_cast<HwlocGetDepthTypeFn>(
        dlsym(RTLD_NEXT, "hwloc_get_depth_type"));
  }();
  return fn;
}

auto
resolve_real_hwloc_get_nbobjs_by_depth() -> HwlocGetNbobjsByDepthFn
{
  static HwlocGetNbobjsByDepthFn fn = []() -> HwlocGetNbobjsByDepthFn {
    return reinterpret_cast<HwlocGetNbobjsByDepthFn>(
        dlsym(RTLD_NEXT, "hwloc_get_nbobjs_by_depth"));
  }();
  return fn;
}

auto
resolve_real_hwloc_get_obj_by_depth() -> HwlocGetObjByDepthFn
{
  static HwlocGetObjByDepthFn fn = []() -> HwlocGetObjByDepthFn {
    return reinterpret_cast<HwlocGetObjByDepthFn>(
        dlsym(RTLD_NEXT, "hwloc_get_obj_by_depth"));
  }();
  return fn;
}

auto
resolve_real_hwloc_bitmap_first() -> HwlocBitmapFirstFn
{
  static HwlocBitmapFirstFn fn = []() -> HwlocBitmapFirstFn {
    return reinterpret_cast<HwlocBitmapFirstFn>(
        dlsym(RTLD_NEXT, "hwloc_bitmap_first"));
  }();
  return fn;
}

auto
resolve_estimate_non_cpu_workers() -> EstimateNonCpuWorkersFn
{
  static EstimateNonCpuWorkersFn fn = []() -> EstimateNonCpuWorkersFn {
    constexpr const char* symbol_name =
        "_ZN13starpu_server12_GLOBAL__N_124estimate_non_cpu_workersERKNS_"
        "13RuntimeConfigE";
    if (void* sym = dlsym(RTLD_DEFAULT, symbol_name); sym != nullptr) {
      return reinterpret_cast<EstimateNonCpuWorkersFn>(sym);
    }
    return nullptr;
  }();
  return fn;
}

auto
resolve_real_setenv() -> SetenvFn
{
  static SetenvFn fn = []() -> SetenvFn {
    return reinterpret_cast<SetenvFn>(dlsym(RTLD_NEXT, "setenv"));
  }();
  return fn;
}

class ScopedWorkerContextOverride {
 public:
  ScopedWorkerContextOverride(
      StarpuWorkerGetIdHook id_hook, StarpuWorkerGetDevidHook devid_hook,
      StarpuCudaGetLocalStreamHook stream_hook)
      : previous_id_{g_worker_get_id_hook},
        previous_devid_{g_worker_get_devid_hook},
        previous_stream_{g_cuda_get_local_stream_hook}
  {
    g_worker_get_id_hook = id_hook;
    g_worker_get_devid_hook = devid_hook;
    g_cuda_get_local_stream_hook = stream_hook;
  }

  ~ScopedWorkerContextOverride()
  {
    g_worker_get_id_hook = previous_id_;
    g_worker_get_devid_hook = previous_devid_;
    g_cuda_get_local_stream_hook = previous_stream_;
  }

  ScopedWorkerContextOverride(const ScopedWorkerContextOverride&) = delete;
  auto operator=(const ScopedWorkerContextOverride&)
      -> ScopedWorkerContextOverride& = delete;
  ScopedWorkerContextOverride(ScopedWorkerContextOverride&&) = delete;
  auto operator=(ScopedWorkerContextOverride&&)
      -> ScopedWorkerContextOverride& = delete;

 private:
  StarpuWorkerGetIdHook previous_id_;
  StarpuWorkerGetDevidHook previous_devid_;
  StarpuCudaGetLocalStreamHook previous_stream_;
};

class ScopedCudaStream {
 public:
  explicit ScopedCudaStream(cudaStream_t stream) : stream_{stream} {}
  ~ScopedCudaStream()
  {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }
  ScopedCudaStream(const ScopedCudaStream&) = delete;
  auto operator=(const ScopedCudaStream&) -> ScopedCudaStream& = delete;
  ScopedCudaStream(ScopedCudaStream&& other) noexcept : stream_{other.stream_}
  {
    other.stream_ = nullptr;
  }
  auto operator=(ScopedCudaStream&& other) noexcept -> ScopedCudaStream&
  {
    if (this != &other) {
      if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
      }
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }
  [[nodiscard]] auto get() const -> cudaStream_t { return stream_; }

 private:
  cudaStream_t stream_;
};

class DeviceBufferGuard {
 public:
  DeviceBufferGuard() = default;
  explicit DeviceBufferGuard(void* ptr) : ptr_{ptr} {}
  ~DeviceBufferGuard()
  {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DeviceBufferGuard(const DeviceBufferGuard&) = delete;
  auto operator=(const DeviceBufferGuard&) -> DeviceBufferGuard& = delete;
  DeviceBufferGuard(DeviceBufferGuard&& other) noexcept : ptr_{other.ptr_}
  {
    other.ptr_ = nullptr;
  }
  auto operator=(DeviceBufferGuard&& other) noexcept -> DeviceBufferGuard&
  {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  void reset(void* ptr)
  {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
    ptr_ = ptr;
  }
  [[nodiscard]] auto get() const -> void* { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

struct CapturedStarpuConf {
  bool called = false;
  starpu_conf conf{};
};

CapturedStarpuConf g_captured_starpu_conf;

auto
resolve_real_starpu_init() -> StarpuInitRawFn
{
  static StarpuInitRawFn fn = nullptr;
  if (fn == nullptr) {
    fn = reinterpret_cast<StarpuInitRawFn>(dlsym(RTLD_NEXT, "starpu_init"));
    if (fn == nullptr) {
      fn =
          reinterpret_cast<StarpuInitRawFn>(dlsym(RTLD_DEFAULT, "starpu_init"));
    }
  }
  return fn;
}

auto
capturing_starpu_init(struct starpu_conf* conf) -> int
{
  g_captured_starpu_conf.called = true;
  g_captured_starpu_conf.conf = *conf;
  if (auto* real_init = resolve_real_starpu_init(); real_init != nullptr) {
    return real_init(conf);
  }
  return -1;
}

class StarpuInitCaptureGuard {
 public:
  StarpuInitCaptureGuard()
  {
    g_captured_starpu_conf = {};
    starpu_server::StarPUSetup::set_starpu_init_fn(&capturing_starpu_init);
  }

  ~StarpuInitCaptureGuard()
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }

  StarpuInitCaptureGuard(const StarpuInitCaptureGuard&) = delete;
  auto operator=(const StarpuInitCaptureGuard&) -> StarpuInitCaptureGuard& =
                                                       delete;
  StarpuInitCaptureGuard(StarpuInitCaptureGuard&&) = delete;
  auto operator=(StarpuInitCaptureGuard&&) -> StarpuInitCaptureGuard& = delete;

  [[nodiscard]] auto called() const -> bool
  {
    return g_captured_starpu_conf.called;
  }

  [[nodiscard]] auto conf() const -> const starpu_conf&
  {
    return g_captured_starpu_conf.conf;
  }
};

auto
capturing_starpu_init_noop(struct starpu_conf* conf) -> int
{
  g_captured_starpu_conf.called = true;
  g_captured_starpu_conf.conf = *conf;
  return 0;
}

class StarpuInitCaptureStubGuard {
 public:
  StarpuInitCaptureStubGuard()
  {
    g_captured_starpu_conf = {};
    starpu_server::StarPUSetup::set_starpu_init_fn(&capturing_starpu_init_noop);
  }

  ~StarpuInitCaptureStubGuard()
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }

  StarpuInitCaptureStubGuard(const StarpuInitCaptureStubGuard&) = delete;
  auto operator=(const StarpuInitCaptureStubGuard&)
      -> StarpuInitCaptureStubGuard& = delete;
  StarpuInitCaptureStubGuard(StarpuInitCaptureStubGuard&&) = delete;
  auto operator=(StarpuInitCaptureStubGuard&&) -> StarpuInitCaptureStubGuard& =
                                                      delete;

  [[nodiscard]] auto called() const -> bool
  {
    return g_captured_starpu_conf.called;
  }

  [[nodiscard]] auto conf() const -> const starpu_conf&
  {
    return g_captured_starpu_conf.conf;
  }
};

class EnvVarGuard {
 public:
  EnvVarGuard(std::string name, std::string value) : name_{std::move(name)}
  {
    if (const char* current = std::getenv(name_.c_str()); current != nullptr) {
      previous_ = std::string(current);
    }
    if (setenv(name_.c_str(), value.c_str(), 1) != 0) {
      ADD_FAILURE() << "Failed to set environment variable " << name_;
    }
  }

  ~EnvVarGuard()
  {
    if (previous_) {
      if (setenv(name_.c_str(), previous_->c_str(), 1) != 0) {
        ADD_FAILURE() << "Failed to restore environment variable " << name_;
      }
    } else if (unsetenv(name_.c_str()) != 0) {
      ADD_FAILURE() << "Failed to unset environment variable " << name_;
    }
  }

  EnvVarGuard(const EnvVarGuard&) = delete;
  auto operator=(const EnvVarGuard&) -> EnvVarGuard& = delete;
  EnvVarGuard(EnvVarGuard&&) = delete;
  auto operator=(EnvVarGuard&&) -> EnvVarGuard& = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

class SetenvOverrideGuard {
 public:
  explicit SetenvOverrideGuard(SetenvFn override_fn)
      : previous_{g_setenv_override}
  {
    g_setenv_override = override_fn;
  }

  ~SetenvOverrideGuard() { g_setenv_override = previous_; }

  SetenvOverrideGuard(const SetenvOverrideGuard&) = delete;
  auto operator=(const SetenvOverrideGuard&) -> SetenvOverrideGuard& = delete;
  SetenvOverrideGuard(SetenvOverrideGuard&&) = delete;
  auto operator=(SetenvOverrideGuard&&) -> SetenvOverrideGuard& = delete;

 private:
  SetenvFn previous_;
};

void
capture_slot_state(const starpu_server::InputSlotPool::SlotInfo& slot)
{
  g_failure_observer_called = true;
  g_observed_base_ptrs.assign(slot.base_ptrs.begin(), slot.base_ptrs.end());
  g_observed_handles.assign(slot.handles.begin(), slot.handles.end());
}

void
capture_output_slot_state(
    const starpu_server::OutputSlotPool::SlotInfo& slot,
    const std::vector<starpu_server::OutputSlotPool::HostBufferInfo>&
        buffer_infos)
{
  g_output_failure_observer_called = true;
  g_output_observed_base_ptrs.assign(
      slot.base_ptrs.begin(), slot.base_ptrs.end());
  g_output_observed_handles.assign(slot.handles.begin(), slot.handles.end());
  g_output_observed_host_buffer_infos.assign(
      buffer_infos.begin(), buffer_infos.end());
}

template <typename Fn>
struct FunctionArguments;

template <typename R, typename... Args>
struct FunctionArguments<R (*)(Args...)> {
  using Tuple = std::tuple<Args...>;
};

using StarpuRegisterArgs =
    FunctionArguments<starpu_server::testing::StarpuVectorRegisterFn>::Tuple;

void
failing_starpu_vector_register(
    std::tuple_element_t<0, StarpuRegisterArgs> handle,
    std::tuple_element_t<1, StarpuRegisterArgs> /*home_node*/,
    std::tuple_element_t<2, StarpuRegisterArgs> /*ptr*/,
    std::tuple_element_t<3, StarpuRegisterArgs> /*numel*/,
    std::tuple_element_t<4, StarpuRegisterArgs> /*element_size*/)
{
  *handle = nullptr;
}

void
succeed_then_fail_starpu_vector_register(
    std::tuple_element_t<0, StarpuRegisterArgs> handle,
    std::tuple_element_t<1, StarpuRegisterArgs> home_node,
    std::tuple_element_t<2, StarpuRegisterArgs> ptr,
    std::tuple_element_t<3, StarpuRegisterArgs> numel,
    std::tuple_element_t<4, StarpuRegisterArgs> element_size)
{
  ++g_succeed_then_fail_register_call_count;
  if (g_succeed_then_fail_register_call_count == 1) {
    if (g_succeed_then_fail_previous_hook != nullptr) {
      g_succeed_then_fail_previous_hook(
          handle, home_node, ptr, numel, element_size);
    }
    if (g_pre_cleanup_registered_handles != nullptr) {
      g_pre_cleanup_registered_handles->push_back(*handle);
    }
    return;
  }

  *handle = nullptr;
}

auto
failing_host_allocator(void** ptr, size_t /*alignment*/, size_t /*size*/) -> int
{
  if (ptr != nullptr) {
    *ptr = nullptr;
  }
  return 1;
}

auto
force_cuda_host_alloc_failure(
    size_t /*bytes*/, bool /*use_pinned*/, bool /*default_cuda_pinned*/) -> bool
{
  return false;
}

auto
starpu_memory_pin_success(void* /*ptr*/, size_t /*size*/) -> int
{
  return 0;
}

constexpr int kStarpuPinTestError = -42;

auto
starpu_memory_pin_failure(void* /*ptr*/, size_t /*size*/) -> int
{
  return kStarpuPinTestError;
}

auto
failing_starpu_init(struct starpu_conf*) -> int
{
  return -1;
}

auto
stub_starpu_init(struct starpu_conf* conf) -> int
{
  const char* const key = "STARPU_NWORKER_PER_CUDA";
  std::optional<std::string> previous_value;

  if (const char* current = std::getenv(key); current != nullptr) {
    previous_value = std::string(current);
  }

  if (setenv(key, "1", 1) != 0) {
    ADD_FAILURE() << "Failed to set environment variable " << key;
    return -1;
  }

  const StarpuInitRawFn real_init = resolve_real_starpu_init();
  const int rc = real_init != nullptr ? real_init(conf) : 0;

  if (previous_value) {
    if (setenv(key, previous_value->c_str(), 1) != 0) {
      ADD_FAILURE() << "Failed to restore environment variable " << key;
    }
  } else if (unsetenv(key) != 0) {
    ADD_FAILURE() << "Failed to unset environment variable " << key;
  }

  return rc;
}

class StarPUSetupInitOverrideTest : public ::testing::Test {
 protected:
  void TearDown() override
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }
};

class StarPUSetupInitStubTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    starpu_server::StarPUSetup::set_starpu_init_fn(&stub_starpu_init);
  }

  void TearDown() override
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }
};

}  // namespace

extern "C" int
starpu_worker_get_id()
{
  if (g_worker_get_id_hook != nullptr) {
    return g_worker_get_id_hook();
  }
  if (auto fn = resolve_real_starpu_worker_get_id(); fn != nullptr) {
    return fn();
  }
  return -1;
}

extern "C" int
starpu_worker_get_devid(int workerid)
{
  if (g_worker_get_devid_hook != nullptr) {
    return g_worker_get_devid_hook(workerid);
  }
  if (auto fn = resolve_real_starpu_worker_get_devid(); fn != nullptr) {
    return fn(workerid);
  }
  return -1;
}

extern "C" cudaStream_t
starpu_cuda_get_local_stream()
{
  if (g_cuda_get_local_stream_hook != nullptr) {
    return g_cuda_get_local_stream_hook();
  }
  if (auto fn = resolve_real_starpu_cuda_get_local_stream(); fn != nullptr) {
    return fn();
  }
  return nullptr;
}

extern "C" int
setenv(const char* name, const char* value, int overwrite)
{
  if (g_setenv_override != nullptr) {
    return g_setenv_override(name, value, overwrite);
  }
  if (auto fn = resolve_real_setenv(); fn != nullptr) {
    return fn(name, value, overwrite);
  }
  return -1;
}

TEST(ConfigureCpu, DisablesCpuWorkersWhenCpuUsageDisabled)
{
  StarpuInitCaptureGuard capture_guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cpu = false;
  opts.devices.group_cpu_by_numa = true;

  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const starpu_server::StarPUInitializationException& ex) {
          EXPECT_STREQ("[ERROR] StarPU initialization error", ex.what());
          throw;
        }
      },
      starpu_server::StarPUInitializationException);

  ASSERT_TRUE(capture_guard.called());
  const starpu_conf captured = capture_guard.conf();
  EXPECT_EQ(0, captured.ncpus);
  EXPECT_EQ(0U, captured.use_explicit_workers_bindid);
  EXPECT_EQ(0, captured.precedence_over_environment_variables);
}

TEST(ConfigureCpu, BindsCpuWorkersWithExplicitBinding)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:2"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  StarpuInitCaptureGuard capture_guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;

  {
    starpu_server::StarPUSetup setup(opts);
  }

  ASSERT_TRUE(capture_guard.called());
  const starpu_conf captured = capture_guard.conf();
  EXPECT_EQ(1, captured.ncpus);
  EXPECT_EQ(1U, captured.use_explicit_workers_bindid);
  EXPECT_EQ(1, captured.precedence_over_environment_variables);
  EXPECT_EQ(0U, captured.workers_bindid[0]);
  EXPECT_EQ(captured.workers_bindid[0], captured.workers_bindid[1]);
  EXPECT_EQ(captured.workers_bindid[0], captured.workers_bindid[2]);
}

TEST(ConfigureCpu, FallbacksToAllCpuIdsWhenNoGpuCandidates)
{
  ASSERT_GT(STARPU_NMAXWORKERS, 1);

  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  StarpuInitCaptureGuard capture_guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;

  {
    starpu_server::StarPUSetup setup(opts);
  }

  ASSERT_TRUE(capture_guard.called());
  const starpu_conf captured = capture_guard.conf();
  EXPECT_EQ(1, captured.ncpus);
  EXPECT_EQ(1U, captured.use_explicit_workers_bindid);
  EXPECT_EQ(1, captured.precedence_over_environment_variables);
  EXPECT_EQ(0U, captured.workers_bindid[0]);
  EXPECT_EQ(0U, captured.workers_bindid[1]);
}

TEST_F(StarPUSetupInitStubTest, ParseUnsignedAcceptsMaxUnsignedConfigValue)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};
  opts.starpu_env["STARPU_NWORKER_PER_CUDA"] =
      std::to_string(std::numeric_limits<unsigned>::max());

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  EXPECT_EQ(log.find("Invalid value"), std::string::npos);
  EXPECT_NE(
      log.find("group_cpu_by_numa requested, but non-CPU workers already reach "
               "StarPU's worker limit"),
      std::string::npos);
}

TEST_F(
    StarPUSetupInitStubTest, ParseUnsignedLogsWarningForNonNumericConfigValue)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};
  constexpr const char* kInvalid = "not-a-number";
  opts.starpu_env["STARPU_NWORKER_PER_CUDA"] = kInvalid;

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  const std::string expected = std::format(
      "Invalid value '{}' for {} in configuration; ignoring binding hint",
      kInvalid, "STARPU_NWORKER_PER_CUDA");
  EXPECT_NE(log.find(expected), std::string::npos);
}

TEST_F(StarPUSetupInitStubTest, ParseUnsignedLogsWarningForOverflowConfigValue)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};
  const std::string overflow_value = std::to_string(
      static_cast<unsigned long>(std::numeric_limits<unsigned>::max()) + 1UL);
  opts.starpu_env["STARPU_NWORKER_PER_CUDA"] = overflow_value;

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  const std::string expected = std::format(
      "Invalid value '{}' for {} in configuration; ignoring binding hint",
      overflow_value, "STARPU_NWORKER_PER_CUDA");
  EXPECT_NE(log.find(expected), std::string::npos);
}

TEST_F(StarPUSetupInitStubTest, GetEnvUnsignedUsesEnvironmentValue)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};
  EnvVarGuard workers_guard{
      "STARPU_NWORKER_PER_CUDA", std::to_string(STARPU_NMAXWORKERS)};

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cpu = true;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  EXPECT_NE(
      log.find("group_cpu_by_numa requested, but non-CPU workers already reach "
               "StarPU's worker limit"),
      std::string::npos);
}

TEST_F(StarPUSetupInitStubTest, GetEnvUnsignedHandlesInvalidEnvironmentValue)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};
  EnvVarGuard workers_guard{"STARPU_NWORKER_PER_CUDA", "invalid"};

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cpu = true;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  EXPECT_NE(
      log.find("Invalid value 'invalid' for environment variable "
               "STARPU_NWORKER_PER_CUDA; ignoring binding hint"),
      std::string::npos);
  EXPECT_EQ(
      log.find("group_cpu_by_numa requested, but non-CPU workers already reach "
               "StarPU's worker limit"),
      std::string::npos);
}

TEST(ApplyStarpuEnv, EmptyNameThrowsInitializationException)
{
  StarpuInitCaptureStubGuard capture_guard;
  starpu_server::RuntimeConfig opts;
  opts.starpu_env[""] = "value";

  try {
    starpu_server::StarPUSetup setup(opts);
    FAIL() << "Expected StarPUInitializationException";
  }
  catch (const starpu_server::StarPUInitializationException& ex) {
    const std::string_view message(ex.what());
    EXPECT_NE(
        message.find("Environment variable name cannot be empty"),
        std::string::npos);
  }
}

TEST(ApplyStarpuEnv, SetenvFailureThrowsInitializationException)
{
  StarpuInitCaptureStubGuard capture_guard;
  SetenvOverrideGuard setenv_guard(&failing_setenv_stub);

  starpu_server::RuntimeConfig opts;
  opts.starpu_env["STARPU_TEST_VAR"] = "1";

  try {
    starpu_server::StarPUSetup setup(opts);
    FAIL() << "Expected StarPUInitializationException";
  }
  catch (const starpu_server::StarPUInitializationException& ex) {
    const std::string_view message(ex.what());
    EXPECT_NE(
        message.find("Failed to set environment variable STARPU_TEST_VAR"),
        std::string::npos);
  }
}

TEST(EstimateNonCpuWorkers, ReturnsMaxUnsignedOnOverflow)
{
  EnvVarGuard component_guard{"HWLOC_COMPONENTS", "synthetic"};
  EnvVarGuard synthetic_guard{"HWLOC_SYNTHETIC", "numa:1 pu:1"};
  EnvVarGuard thissystem_guard{"HWLOC_THISSYSTEM", "0"};

  StarpuInitCaptureStubGuard capture_guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.group_cpu_by_numa = true;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 1};
  const unsigned workers_per_gpu =
      (std::numeric_limits<unsigned>::max() / 2U) + 1U;
  opts.starpu_env["STARPU_NWORKER_PER_CUDA"] = std::to_string(workers_per_gpu);

  std::string log;
  {
    starpu_server::CaptureStream capture{std::cerr};
    {
      starpu_server::StarPUSetup setup(opts);
    }
    log = capture.str();
  }

  ASSERT_TRUE(capture_guard.called());
  EXPECT_NE(
      log.find("group_cpu_by_numa requested, but non-CPU workers already reach "
               "StarPU's worker limit"),
      std::string::npos);
}

TEST(StarPUSetupHooks, SetStarpuInitFnNullRestoresDefault)
{
  g_starpu_init_stub_called = false;
  starpu_server::StarPUSetup::set_starpu_init_fn(&starpu_init_stub_for_null);
  starpu_server::StarPUSetup::set_starpu_init_fn(nullptr);

  starpu_server::RuntimeConfig opts;
  {
    starpu_server::StarPUSetup setup(opts);
  }

  EXPECT_FALSE(g_starpu_init_stub_called);
  starpu_server::StarPUSetup::reset_starpu_init_fn();
}

TEST(StarPUSetupHooks, SetWorkerStreamQueryFnNullRestoresDefault)
{
  g_worker_stream_stub_called = false;
  starpu_server::StarPUSetup::set_worker_stream_query_fn(
      &worker_stream_stub_for_null);
  starpu_server::StarPUSetup::set_worker_stream_query_fn(nullptr);

  {
    StarpuRuntimeGuard guard;
    try {
      std::ignore = starpu_server::StarPUSetup::get_cuda_workers_by_device({0});
    }
    catch (const std::exception&) {
      // Ignore failures; only verifying that the stub was not invoked.
    }
  }

  EXPECT_FALSE(g_worker_stream_stub_called);
  starpu_server::StarPUSetup::reset_worker_stream_query_fn();
}

TEST_F(StarPUSetupInitOverrideTest, FailingStarpuInitThrows)
{
  starpu_server::RuntimeConfig opts;

  starpu_server::StarPUSetup::set_starpu_init_fn(&failing_starpu_init);

  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const starpu_server::StarPUInitializationException& ex) {
          EXPECT_STREQ("[ERROR] StarPU initialization error", ex.what());
          throw;
        }
      },
      starpu_server::StarPUInitializationException);
}

TEST(StarPUSetup_Unit, DuplicateDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 0};
  EXPECT_THROW(
      { starpu_server::StarPUSetup setup(opts); }, std::invalid_argument);
}

TEST(InferenceCodelet, RunCodeletInferenceLogsTraceMessage)
{
  const StarpuRuntimeGuard guard;

  auto params = starpu_server::make_basic_params(3);
  params.verbosity = starpu_server::VerbosityLevel::Trace;
  params.request_id = 99;

  torch::jit::script::Module module{"trace_logger"};
  module.define(R"JIT(
        def forward(self, x):
            return x
    )JIT");
  params.models.model_cpu = &module;

  TestBuffers buffers = make_test_buffers();

  g_mock_worker_id = 11;
  g_mock_device_id = 5;
  ScopedWorkerContextOverride worker_guard(
      &mock_starpu_worker_get_id, &mock_starpu_worker_get_devid, nullptr);

  starpu_server::InferenceCodelet codelet;
  auto* cpu_func = codelet.get_codelet()->cpu_funcs[0];
  ASSERT_NE(cpu_func, nullptr);

  starpu_server::CaptureStream capture{std::cout};
  cpu_func(reinterpret_cast<void**>(buffers.buffers.data()), &params);
  const std::string log = capture.str();

  EXPECT_NE(
      log.find(std::format(
          "CPU device id {}, worker id {}, job id {}", g_mock_device_id,
          g_mock_worker_id, params.request_id)),
      std::string::npos);

  for (float value : buffers.output_data) {
    EXPECT_NE(value, 0.0F);
  }

  g_mock_worker_id = 0;
  g_mock_device_id = 0;
}

TEST(InferenceCodelet, CudaInferenceFuncCopiesResultsToDeviceBuffer)
{
  skip_if_no_cuda();

  constexpr int kDeviceId = 0;
  constexpr int kWorkerId = 77;
  constexpr size_t kElementCount = 3;
  constexpr size_t kBufferBytes = kElementCount * sizeof(float);

  ASSERT_EQ(cudaSetDevice(kDeviceId), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ScopedCudaStream stream_guard{stream};

  g_mock_worker_id = kWorkerId;
  g_mock_device_id = kDeviceId;
  g_mock_cuda_stream = stream;
  ScopedWorkerContextOverride worker_guard(
      &mock_starpu_worker_get_id, &mock_starpu_worker_get_devid,
      &mock_starpu_cuda_get_local_stream);

  DeviceBufferGuard input_buffer;
  DeviceBufferGuard output_buffer;

  void* raw_input = nullptr;
  ASSERT_EQ(cudaMalloc(&raw_input, kBufferBytes), cudaSuccess);
  input_buffer.reset(raw_input);

  void* raw_output = nullptr;
  ASSERT_EQ(cudaMalloc(&raw_output, kBufferBytes), cudaSuccess);
  output_buffer.reset(raw_output);

  std::array<float, kElementCount> host_input{1.0F, 2.0F, 3.0F};
  ASSERT_EQ(
      cudaMemcpy(
          raw_input, host_input.data(), kBufferBytes, cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemset(raw_output, 0, kBufferBytes), cudaSuccess);

  auto params =
      starpu_server::make_basic_params(static_cast<int>(kElementCount));
  params.models.models_gpu.resize(kDeviceId + 1, nullptr);

  torch::jit::script::Module module{"m"};
  module.define(R"JIT(
        def forward(self, x):
            return x + 1.5
    )JIT");
  module.to(torch::Device(torch::kCUDA, kDeviceId));
  params.models.models_gpu[kDeviceId] = &module;
  params.models.num_models_gpu = params.models.models_gpu.size();

  int recorded_device_id = -1;
  int recorded_worker_id = -1;
  starpu_server::DeviceType executed_on = starpu_server::DeviceType::Unknown;
  params.device.device_id = &recorded_device_id;
  params.device.worker_id = &recorded_worker_id;
  params.device.executed_on = &executed_on;

  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_tp{};
  Clock::time_point end_tp{};
  params.timing.codelet_start_time = &start_tp;
  params.timing.codelet_end_time = &end_tp;

  starpu_variable_interface input_iface{};
  input_iface.ptr = reinterpret_cast<uintptr_t>(raw_input);
  starpu_variable_interface output_iface{};
  output_iface.ptr = reinterpret_cast<uintptr_t>(raw_output);

  std::array<starpu_server::StarpuBufferPtr, 2> buffers{
      &input_iface, &output_iface};

  starpu_server::InferenceCodelet codelet;
  auto* cuda_func = codelet.get_codelet()->cuda_funcs[0];
  ASSERT_NE(cuda_func, nullptr);

  cuda_func(reinterpret_cast<void**>(buffers.data()), &params);
  ASSERT_EQ(cudaStreamSynchronize(stream_guard.get()), cudaSuccess);

  std::array<float, kElementCount> host_output{};
  ASSERT_EQ(
      cudaMemcpy(
          host_output.data(), raw_output, kBufferBytes, cudaMemcpyDeviceToHost),
      cudaSuccess);

  for (size_t idx = 0; idx < kElementCount; ++idx) {
    EXPECT_FLOAT_EQ(host_output[idx], host_input[idx] + 1.5F);
  }
  EXPECT_EQ(recorded_device_id, kDeviceId);
  EXPECT_EQ(recorded_worker_id, kWorkerId);
  EXPECT_EQ(executed_on, starpu_server::DeviceType::CUDA);
  EXPECT_NE(start_tp.time_since_epoch().count(), 0);
  EXPECT_NE(end_tp.time_since_epoch().count(), 0);

  g_mock_worker_id = 0;
  g_mock_device_id = 0;
  g_mock_cuda_stream = nullptr;
}

TEST(InferenceCodelet, CudaInferenceFuncThrowsWhenGpuModuleMissing)
{
  skip_if_no_cuda();

  constexpr int kDeviceId = 0;
  constexpr int kWorkerId = 91;

  ASSERT_EQ(cudaSetDevice(kDeviceId), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ScopedCudaStream stream_guard{stream};

  g_mock_worker_id = kWorkerId;
  g_mock_device_id = kDeviceId;
  g_mock_cuda_stream = stream;
  ScopedWorkerContextOverride worker_guard(
      &mock_starpu_worker_get_id, &mock_starpu_worker_get_devid,
      &mock_starpu_cuda_get_local_stream);

  starpu_server::InferenceCodelet codelet;
  starpu_server::InferenceParams params{};
  params.num_inputs = 0;
  params.num_outputs = 0;
  params.limits.max_inputs = starpu_server::InferLimits::MaxInputs;
  params.limits.max_dims = starpu_server::InferLimits::MaxDims;
  params.limits.max_models_gpu = starpu_server::InferLimits::MaxModelsGPU;

  auto* cuda_func = codelet.get_codelet()->cuda_funcs[0];
  ASSERT_NE(cuda_func, nullptr);

  EXPECT_THROW(
      cuda_func(nullptr, &params), starpu_server::StarPUCodeletException);

  g_mock_worker_id = 0;
  g_mock_device_id = 0;
  g_mock_cuda_stream = nullptr;
}

TEST(OutputSlotPool_Unit, CheckedTotalNumelGuard)
{
  EXPECT_NO_THROW(
      starpu_server::OutputSlotPoolTestHook::checked_total_numel(16, 8));

  EXPECT_THROW(
      {
        starpu_server::OutputSlotPoolTestHook::checked_total_numel(
            std::numeric_limits<size_t>::max(), 2);
      },
      std::overflow_error);
}

TEST(OutputSlotPool_Unit, FreeHostBufferNullPointerNoWarning)
{
  starpu_server::OutputSlotPool::HostBufferInfo info{};
  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(
      nullptr, info);
  EXPECT_TRUE(capture.str().empty());
}

TEST(OutputSlotPool_Unit, HostDeallocatorHookIgnoresNullptr)
{
  auto& deallocator_hook =
      starpu_server::OutputSlotPoolTestHook::host_deallocator_hook_ref();
  ASSERT_TRUE(static_cast<bool>(deallocator_hook));

  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_NO_THROW(deallocator_hook(nullptr));
  EXPECT_TRUE(capture.str().empty());
}

TEST(OutputSlotPool_Unit, FreeHostBufferStarpuUnpinFailureLogsWarning)
{
  constexpr size_t kBytes = 32;
  auto* ptr = static_cast<std::byte*>(std::malloc(kBytes));
  ASSERT_NE(ptr, nullptr);

  starpu_server::OutputSlotPool::HostBufferInfo info{};
  info.starpu_pinned = true;
  info.bytes = kBytes;

  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(ptr, info);
  const std::string log = capture.str();
  EXPECT_NE(log.find("starpu_memory_unpin failed"), std::string::npos);
}

TEST(OutputSlotPool_Unit, FreeHostBufferCudaFreeHostFailureLogsWarning)
{
  int device_count = 0;
  const cudaError_t device_rc = cudaGetDeviceCount(&device_count);
  if (device_rc == cudaErrorInsufficientDriver ||
      device_rc == cudaErrorNoDevice) {
    GTEST_SKIP() << "CUDA runtime unavailable for test: rc="
                 << static_cast<int>(device_rc);
  }
  if (device_rc != cudaSuccess) {
    GTEST_SKIP() << "CUDA runtime check failed: rc="
                 << static_cast<int>(device_rc);
  }

  constexpr size_t kBytes = 32;
  auto* ptr = static_cast<std::byte*>(std::malloc(kBytes));
  ASSERT_NE(ptr, nullptr);

  starpu_server::OutputSlotPool::HostBufferInfo info{};
  info.cuda_pinned = true;
  info.bytes = kBytes;

  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(ptr, info);
  const std::string log = capture.str();
  EXPECT_NE(log.find("cudaFreeHost failed"), std::string::npos);

  if (log.find("cudaFreeHost failed") != std::string::npos) {
    std::free(ptr);
  }
}

TEST(StarPUSetup_Unit, TooManyDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;

  opts.devices.ids.reserve(STARPU_NMAXWORKERS + 1);
  for (int idx = 0; idx < STARPU_NMAXWORKERS + 1; ++idx) {
    opts.devices.ids.push_back(idx);
  }

  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ(
              std::format(
                  "[ERROR] Number of CUDA device IDs exceeds maximum of {}",
                  STARPU_NMAXWORKERS)
                  .c_str(),
              ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST_F(
    StarPUSetupInitStubTest, StarPUSetup_InputPoolInitFailureLogsAndPropagates)
{
  starpu_server::RuntimeConfig opts;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig invalid_input;
  invalid_input.name = "invalid_input";
  invalid_input.dims = {0, 1};
  invalid_input.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_input_model";
  model.inputs.push_back(invalid_input);
  opts.models.push_back(model);

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log_output.find("Failed to initialize InputSlotPool"), std::string::npos);
}

TEST_F(
    StarPUSetupInitStubTest, StarPUSetup_OutputPoolInitFailureLogsAndPropagates)
{
  starpu_server::RuntimeConfig opts;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig invalid_output;
  invalid_output.name = "invalid_output";
  invalid_output.dims = {0, 1};
  invalid_output.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_output_model";
  model.outputs.push_back(invalid_output);
  opts.models.push_back(model);

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log_output.find("Failed to initialize OutputSlotPool"),
      std::string::npos);
}

TEST(InputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = std::numeric_limits<int>::max();
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "large_input";
  tensor.dims = {1, 65536, 65536};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, AllocateSlotBuffersNumelOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "numel_overflow_input";
  tensor.dims = {1, 4611686018427387905};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "numel_overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, ConstructionWithoutModelsThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::invalid_argument);
}

TEST(InputSlotPool_Unit, NonPositiveBatchDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "invalid_batch_input";
  tensor.dims = {0, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_batch_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, BatchDimensionExceedsIntMaxThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "exceeds_int_max_input";
  tensor.dims = {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "exceeds_int_max_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) exceeds int max", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, NonPositiveDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "non_positive_dims";
  tensor.dims = {1, 0, 8};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "non_positive_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_NE(
              std::string(ex.what()).find("dims must be positive"),
              std::string::npos);
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, DimensionProductOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "product_overflow_input";
  tensor.dims = {1, std::numeric_limits<int64_t>::max(), 3};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "product_overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, SlotInfoProvidesConsistentReferences)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& info = pool.slot_info(slot_id);

  EXPECT_EQ(info.id, slot_id);
  ASSERT_EQ(info.base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(info.handles.size(), model.inputs.size());

  const auto& base_ptrs_ref = pool.base_ptrs(slot_id);
  const auto& handles_ref = pool.handles(slot_id);

  EXPECT_EQ(base_ptrs_ref.size(), info.base_ptrs.size());
  EXPECT_EQ(handles_ref.size(), info.handles.size());

  EXPECT_EQ(
      static_cast<const void*>(&base_ptrs_ref),
      static_cast<const void*>(&info.base_ptrs));
  EXPECT_EQ(
      static_cast<const void*>(&handles_ref),
      static_cast<const void*>(&info.handles));

  EXPECT_THROW(
      static_cast<void>(pool.slot_info(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(pool.base_ptrs(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(pool.handles(slot_id + 1)), std::out_of_range);

  pool.release(slot_id);
}

TEST(InputSlotPool_Unit, TryAcquireEmptyPoolReturnsNullopt)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  EXPECT_EQ(pool.try_acquire(), std::nullopt);

  pool.release(slot_id);

  const auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, slot_id);
  pool.release(*reacquired);
}

TEST(InputSlotPool_Unit, HostBufferInfoIndicatesCudaPinningAttempt)
{
  StarpuRuntimeGuard starpu_guard;

  void* probe_ptr = nullptr;
  const cudaError_t probe_err =
      cudaHostAlloc(&probe_ptr, 1, cudaHostAllocPortable);
  if (probe_err == cudaSuccess) {
    cudaFreeHost(probe_ptr);
  } else if (
      probe_err == cudaErrorNotSupported ||
      probe_err == cudaErrorInsufficientDriver ||
      probe_err == cudaErrorNoDevice) {
    GTEST_SKIP() << "cudaHostAlloc unsupported: "
                 << cudaGetErrorString(probe_err);
  }

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "cuda_probe_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "cuda_probe_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& buffer_infos = pool.host_buffer_infos(slot_id);
  ASSERT_EQ(buffer_infos.size(), 1);

  const auto& info = buffer_infos.front();

  if (info.cuda_pinned) {
    EXPECT_TRUE(info.cuda_pinned);
  } else {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned || info.starpu_pin_rc != 0)
        << "Fallback StarPU pinning should report a result";
  }

  pool.release(slot_id);
}

TEST(OutputSlotPool_Unit, HostBufferInfoIndicatesCudaPinningAttempt)
{
  StarpuRuntimeGuard starpu_guard;

  void* probe_ptr = nullptr;
  const cudaError_t probe_err =
      cudaHostAlloc(&probe_ptr, 1, cudaHostAllocPortable);
  if (probe_err == cudaSuccess) {
    cudaFreeHost(probe_ptr);
  } else if (
      probe_err == cudaErrorNotSupported ||
      probe_err == cudaErrorInsufficientDriver ||
      probe_err == cudaErrorNoDevice) {
    GTEST_SKIP() << "cudaHostAlloc unsupported: "
                 << cudaGetErrorString(probe_err);
  }

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "cuda_probe_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "cuda_probe_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& buffer_infos =
      starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
  ASSERT_EQ(buffer_infos.size(), 1);

  const auto& info = buffer_infos.front();

  if (info.cuda_pinned) {
    EXPECT_TRUE(info.cuda_pinned);
  } else {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned || info.starpu_pin_rc != 0)
        << "Fallback StarPU pinning should report a result";
  }

  pool.release(slot_id);
}

TEST(
    OutputSlotPool_Unit,
    HostBufferInfoReportsStarpuPinSuccessWhenCudaHostAllocFails)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "starpu_pin_success";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "starpu_pin_success_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_cuda_override =
      starpu_server::testing::set_output_cuda_pinned_override_for_tests(
          &force_cuda_host_alloc_failure);
  const auto previous_pin_hook =
      starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
          &starpu_memory_pin_success);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
        previous_pin_hook);
    starpu_server::testing::set_output_cuda_pinned_override_for_tests(
        previous_cuda_override);
  };

  try {
    starpu_server::OutputSlotPool pool(opts, 1);

    const int slot_id = pool.acquire();
    const auto& buffer_infos =
        starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
    ASSERT_EQ(buffer_infos.size(), 1);

    const auto& info = buffer_infos.front();
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, 0);

    pool.release(slot_id);
    restore_hooks();
  }
  catch (...) {
    restore_hooks();
    throw;
  }
}

TEST(
    OutputSlotPool_Unit,
    HostBufferInfoCapturesStarpuPinFailureWhenCudaHostAllocFails)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "starpu_pin_failure";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "starpu_pin_failure_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_cuda_override =
      starpu_server::testing::set_output_cuda_pinned_override_for_tests(
          &force_cuda_host_alloc_failure);
  const auto previous_pin_hook =
      starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
          &starpu_memory_pin_failure);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
        previous_pin_hook);
    starpu_server::testing::set_output_cuda_pinned_override_for_tests(
        previous_cuda_override);
  };

  try {
    starpu_server::OutputSlotPool pool(opts, 1);

    const int slot_id = pool.acquire();
    const auto& buffer_infos =
        starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
    ASSERT_EQ(buffer_infos.size(), 1);

    const auto& info = buffer_infos.front();
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_FALSE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, kStarpuPinTestError);

    pool.release(slot_id);
    restore_hooks();
  }
  catch (...) {
    restore_hooks();
    throw;
  }
}

TEST(OutputSlotPool_Unit, HostAllocatorFailureThrowsBadAlloc)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "host_allocator_failure";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "host_allocator_failure_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_allocator =
      starpu_server::testing::set_output_host_allocator_for_tests(
          &failing_host_allocator);

  auto restore_allocator = [&]() {
    starpu_server::testing::set_output_host_allocator_for_tests(
        previous_allocator);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
          restore_allocator();
          FAIL() << "Expected host allocator failure";
        }
        catch (...) {
          restore_allocator();
          throw;
        }
      },
      std::bad_alloc);
}

TEST(InputSlotPool_Unit, RegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "failing_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "failing_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  g_observed_base_ptrs.clear();
  g_observed_handles.clear();
  g_failure_observer_called = false;

  const auto previous_hook =
      starpu_server::testing::set_starpu_vector_register_hook_for_tests(
          &failing_starpu_vector_register);
  const auto previous_observer =
      starpu_server::testing::set_starpu_register_failure_observer_for_tests(
          &capture_slot_state);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_starpu_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_TRUE(g_failure_observer_called);
  ASSERT_EQ(g_observed_base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(g_observed_handles.size(), model.inputs.size());

  for (std::byte* base_ptr : g_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }
}

TEST(InputSlotPool_Unit, PartialRegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig first_tensor;
  first_tensor.name = "first_input";
  first_tensor.dims = {1, 1};
  first_tensor.type = at::ScalarType::Float;

  starpu_server::TensorConfig second_tensor;
  second_tensor.name = "second_input";
  second_tensor.dims = {1, 1};
  second_tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "partial_failure_model";
  model.inputs.push_back(first_tensor);
  model.inputs.push_back(second_tensor);
  opts.models.push_back(model);

  g_observed_base_ptrs.clear();
  g_observed_handles.clear();
  g_failure_observer_called = false;

  g_succeed_then_fail_register_call_count = 0;
  std::vector<starpu_data_handle_t> registered_handles_before_cleanup;
  g_pre_cleanup_registered_handles = &registered_handles_before_cleanup;

  const auto previous_hook =
      starpu_server::testing::set_starpu_vector_register_hook_for_tests(
          &succeed_then_fail_starpu_vector_register);
  g_succeed_then_fail_previous_hook = previous_hook;
  const auto previous_observer =
      starpu_server::testing::set_starpu_register_failure_observer_for_tests(
          &capture_slot_state);

  int call_count_at_failure = 0;
  auto restore_hooks = [&]() {
    call_count_at_failure = g_succeed_then_fail_register_call_count;
    g_pre_cleanup_registered_handles = nullptr;
    g_succeed_then_fail_previous_hook = nullptr;
    g_succeed_then_fail_register_call_count = 0;
    starpu_server::testing::set_starpu_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_EQ(call_count_at_failure, 2);
  ASSERT_TRUE(g_failure_observer_called);
  ASSERT_EQ(g_observed_base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(g_observed_handles.size(), model.inputs.size());

  ASSERT_EQ(registered_handles_before_cleanup.size(), 1);
  EXPECT_NE(registered_handles_before_cleanup.front(), nullptr);

  ASSERT_FALSE(g_observed_base_ptrs.empty());
  ASSERT_FALSE(g_observed_handles.empty());
  EXPECT_EQ(g_observed_base_ptrs.front(), nullptr);
  EXPECT_EQ(g_observed_handles.front(), nullptr);

  for (std::byte* base_ptr : g_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }
}

TEST(OutputSlotPool_Unit, RegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "failing_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "failing_output_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  g_output_observed_base_ptrs.clear();
  g_output_observed_handles.clear();
  g_output_observed_host_buffer_infos.clear();
  g_output_failure_observer_called = false;

  const auto previous_hook =
      starpu_server::testing::set_output_starpu_vector_register_hook_for_tests(
          &failing_starpu_vector_register);
  const auto previous_observer =
      starpu_server::testing::set_output_register_failure_observer_for_tests(
          &capture_output_slot_state);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_output_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_TRUE(g_output_failure_observer_called);
  ASSERT_EQ(g_output_observed_base_ptrs.size(), model.outputs.size());
  ASSERT_EQ(g_output_observed_handles.size(), model.outputs.size());
  ASSERT_EQ(g_output_observed_host_buffer_infos.size(), model.outputs.size());

  for (std::byte* base_ptr : g_output_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_output_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }

  for (const auto& info : g_output_observed_host_buffer_infos) {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_FALSE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, 0);
    EXPECT_EQ(info.bytes, 0U);
  }
}

TEST(OutputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = std::numeric_limits<int>::max();
  opts.batching.pool_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "large_output";
  tensor.dims = {1, 65536, 65536};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "overflow_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::OutputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(OutputSlotPool_Unit, ConstructionWithoutModelsThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;

  EXPECT_THROW(
      { starpu_server::OutputSlotPool pool(opts, 1); }, std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonPositiveBatchDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "invalid_batch_output";
  tensor.dims = {0, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_batch_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonBatchDimensionNonPositiveThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;

  starpu_server::TensorConfig tensor;
  tensor.name = "non_positive_dims_output";
  tensor.dims = {1, 1, 0};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "non_positive_dims_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dimensions must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, BatchDimensionExceedsIntMaxThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "exceeds_int_max_output";
  tensor.dims = {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "exceeds_int_max_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) exceeds int max", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonBatchDimensionOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;

  starpu_server::TensorConfig tensor;
  tensor.name = "dimension_product_overflow_output";
  tensor.dims = {
      1, static_cast<int64_t>(1ULL << 32), static_cast<int64_t>(1ULL << 32)};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "dimension_product_overflow_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::overflow_error& ex) {
          EXPECT_STREQ("dimension product overflow", ex.what());
          throw;
        }
      },
      std::overflow_error);
}

TEST(OutputSlotPool_Unit, ReleaseReturnsSlotToPool)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "simple_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "simple_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  EXPECT_GE(slot_id, 0);

  EXPECT_FALSE(pool.try_acquire().has_value());

  pool.release(slot_id);

  auto maybe_slot = pool.try_acquire();
  ASSERT_TRUE(maybe_slot.has_value());
  EXPECT_EQ(*maybe_slot, slot_id);

  pool.release(*maybe_slot);
}

TEST(OutputSlotPool_Unit, SlotInfoProvidesConsistentReferences)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& info = pool.slot_info(slot_id);

  EXPECT_EQ(info.id, slot_id);
  ASSERT_EQ(info.base_ptrs.size(), model.outputs.size());
  ASSERT_EQ(info.handles.size(), model.outputs.size());

  const auto& base_ptrs_ref = pool.base_ptrs(slot_id);
  const auto& handles_ref = pool.handles(slot_id);

  EXPECT_EQ(base_ptrs_ref.size(), info.base_ptrs.size());
  EXPECT_EQ(handles_ref.size(), info.handles.size());

  EXPECT_EQ(
      static_cast<const void*>(&base_ptrs_ref),
      static_cast<const void*>(&info.base_ptrs));
  EXPECT_EQ(
      static_cast<const void*>(&handles_ref),
      static_cast<const void*>(&info.handles));

  EXPECT_THROW(
      static_cast<void>(pool.slot_info(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(pool.base_ptrs(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(pool.handles(slot_id + 1)), std::out_of_range);

  pool.release(slot_id);
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersReleasesResources)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  auto* raw_ptr = std::malloc(sizeof(int));
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}

TEST(OutputSlotPool_Unit, DefaultSlotCountUsesWorkerCount)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "single_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "single_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const int expected_slots =
      std::max(2, static_cast<int>(starpu_worker_get_count()));

  starpu_server::OutputSlotPool pool(opts, 0);

  std::vector<int> acquired_ids;
  acquired_ids.reserve(static_cast<size_t>(expected_slots));

  for (int i = 0; i < expected_slots; ++i) {
    auto maybe_slot = pool.try_acquire();
    ASSERT_TRUE(maybe_slot.has_value())
        << "Expected to acquire slot " << i << " of " << expected_slots;
    acquired_ids.push_back(*maybe_slot);
  }

  EXPECT_FALSE(pool.try_acquire().has_value());

  std::unordered_set<int> unique_ids(acquired_ids.begin(), acquired_ids.end());
  EXPECT_EQ(unique_ids.size(), acquired_ids.size());
  EXPECT_EQ(acquired_ids.size(), static_cast<size_t>(expected_slots));

  for (int slot_id : acquired_ids) {
    pool.release(slot_id);
  }
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersUnpinsStarpuMemory)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  auto* raw_ptr = std::malloc(sizeof(int));
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  ASSERT_EQ(starpu_memory_pin(raw_ptr, sizeof(int)), 0);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);
  buffer_infos[0].starpu_pinned = true;
  buffer_infos[0].starpu_pin_rc = 0;

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  // Failures in starpu_memory_unpin are reported via warnings; this test
  // exercises the successful cleanup path.
  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersFreesCudaPinnedMemory)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  void* raw_ptr = nullptr;
  cudaError_t alloc_rc =
      cudaHostAlloc(&raw_ptr, sizeof(int), cudaHostAllocPortable);
  if (alloc_rc != cudaSuccess) {
    GTEST_SKIP() << "cudaHostAlloc not supported: rc="
                 << static_cast<int>(alloc_rc);
  }
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);
  buffer_infos[0].cuda_pinned = true;

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  // Failures in cudaFreeHost are reported via warnings; this test exercises the
  // successful cleanup path.
  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}
