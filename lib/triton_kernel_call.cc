/* // Copyright 2023 The jax_triton Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>
#include <iostream>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include <hip/hip_runtime.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"  // IWYU pragma: keep
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

#define RETURN_IF_ERROR(expr)               \
  do {                                      \
    absl::Status status = (expr);           \
    if (ABSL_PREDICT_FALSE(!status.ok())) { \
      return status;                        \
    }                                       \
  } while (false)

#define ROCM_TO_STATUS(expr) \
  jax_triton::ToStatus(expr, __FILE__, __LINE__, #expr)

#define ROCM_RETURN_IF_ERROR(expr) RETURN_IF_ERROR(ROCM_TO_STATUS(expr))

namespace py = pybind11;

namespace jax_triton {
namespace {

absl::Status ToStatus(hipError_t result, const char* file, int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_TRUE(result == hipSuccess)) {
    return absl::OkStatus();
  }

  const char* str;
  CHECK_EQ(hipDrvGetErrorName(result, &str), hipSuccess);
  return absl::InternalError(absl::StrFormat("%s:%d: ROCM call `%s` failed: %s",
                                             file, line, expr, str));
}

constexpr uint32_t kNumThreadsPerWarp = 64;
// constexpr uint32_t kNumThreadsPerWarp = 32;

struct hipModuleDeleter {
  void operator()(hipModule_t module) { hipModuleUnload(module); }
};

using OwnedhipModule =
    std::unique_ptr<std::remove_pointer_t<hipModule_t>, hipModuleDeleter>;

class TritonKernel {
 public:
  TritonKernel(std::string module_image, std::string kernel_name,
               uint32_t num_warps, uint32_t shared_mem_bytes)
      : module_image_(std::move(module_image)),
        kernel_name_(std::move(kernel_name)),
        block_dim_x_(num_warps * kNumThreadsPerWarp),
        shared_mem_bytes_(shared_mem_bytes) {}

  absl::Status Launch(hipStream_t stream, uint32_t grid[3], void** params) {
    // TOD0: invetigate
    hipCtx_t context;
    hipDevice_t device;
    int device_id = hipGetStreamDeviceId(stream);
    ROCM_RETURN_IF_ERROR(hipDeviceGet(&device, device_id));
    ROCM_RETURN_IF_ERROR(hipDevicePrimaryCtxRetain(&context, device));
    absl::StatusOr<hipFunction_t> kernel = GetFunctionForContext(context);
    RETURN_IF_ERROR(kernel.status());
    return ROCM_TO_STATUS(hipModuleLaunchKernel(
        *kernel, grid[0], grid[1], grid[2], block_dim_x_,
        /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_, stream, params,
        /*extra=*/nullptr));
  }

 private:
  absl::StatusOr<hipFunction_t> GetFunctionForContext(hipCtx_t context) {
    absl::MutexLock lock(&mutex_);
    auto it = functions_.find(context);
    if (it != functions_.end()) {
      return it->second;
    }

    // set HIP options
    hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                          hipJitOptionErrorLogBuffer,
                          hipJitOptionInfoLogBufferSizeBytes,
                          hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
    const unsigned int errbufsize = 8192;
    const unsigned int logbufsize = 8192;
    char _err[errbufsize];
    char _log[logbufsize];
    void *optval[] = {(void *)(uintptr_t)errbufsize, (void *)_err,
                      (void *)(uintptr_t)logbufsize, (void *)_log, (void *)1};

    ROCM_RETURN_IF_ERROR(hipCtxPushCurrent(context));
    absl::Cleanup ctx_restorer = [] { hipCtxPopCurrent(nullptr); };

    hipModule_t module;
    void* hsaco_data = const_cast<char*>(module_image_.c_str());
    ROCM_RETURN_IF_ERROR(hipModuleLoadDataEx(&module, hsaco_data, 5, opt, optval));
    // ROCM_RETURN_IF_ERROR(hipModuleLoadData(&module, hsaco_data));
    modules_.push_back(OwnedhipModule(module, hipModuleDeleter()));

    hipFunction_t function;
    ROCM_RETURN_IF_ERROR(
        hipModuleGetFunction(&function, module, kernel_name_.c_str()));
    auto [_, success] = functions_.insert({context, function});
    CHECK(success);

    // The maximum permitted static shared memory allocation in CUDA is 48kB,
    // but we can expose more to the kernel using dynamic shared memory.
    constexpr int kMaxStaticSharedMemBytes = 49152;
    if (shared_mem_bytes_ <= kMaxStaticSharedMemBytes) {
      return function;
    }

    // Set up dynamic shared memory.
    hipDevice_t device;
    ROCM_RETURN_IF_ERROR(hipCtxGetDevice(&device));

    int shared_optin;
    ROCM_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &shared_optin, hipDeviceAttributeMaxSharedMemoryPerBlock ,
        device));

    if (shared_optin > kMaxStaticSharedMemBytes) {
      //ROCM_RETURN_IF_ERROR(
      //    cuFuncSetCacheConfig(function, hipFuncCachePreferShared));
      int shared_total;
      ROCM_RETURN_IF_ERROR(hipDeviceGetAttribute(
          &shared_total,
          hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
      int shared_static;
      ROCM_RETURN_IF_ERROR(hipFuncGetAttribute(
          &shared_static, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
      //ROCM_RETURN_IF_ERROR(cuFuncSetAttribute(
      //    function, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      //    shared_optin - shared_static));
    }
    return function;
  }

  std::string module_image_;
  std::string kernel_name_;
  uint32_t block_dim_x_;
  uint32_t shared_mem_bytes_;

  absl::Mutex mutex_;
  std::vector<OwnedhipModule> modules_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<hipCtx_t, hipFunction_t> functions_ ABSL_GUARDED_BY(mutex_);
};

struct TritonKernelCallBase {
  virtual ~TritonKernelCallBase() = default;
  virtual absl::Status Launch(hipStream_t stream, void** buffers) = 0;
};

class TritonKernelCall : public TritonKernelCallBase {
 public:
  struct ArrayParameter {
    size_t bytes_to_zero;
    bool ptr_must_be_divisible_by_16;
  };

  // Parameters can be either to either arrays or scalars (encoded as uint64).
  using Parameter = std::variant<ArrayParameter, uint64_t>;

  TritonKernelCall(TritonKernel& kernel, uint32_t grid_0, uint32_t grid_1,
                   uint32_t grid_2, std::vector<Parameter> parameters)
      : kernel_(kernel),
        grid_{grid_0, grid_1, grid_2},
        parameters_(std::move(parameters)) {}

  absl::Status Launch(hipStream_t stream, void** buffers) override final {
    std::vector<void*> params;
    params.reserve(parameters_.size());
    for (size_t i = 0; i < parameters_.size(); ++i) {
      const Parameter& param = parameters_[i];
      if (std::holds_alternative<ArrayParameter>(param)) {
        const ArrayParameter& array = std::get<ArrayParameter>(param);
        void*& ptr = *(buffers++);
        auto hip_ptr = reinterpret_cast<hipDeviceptr_t>(ptr);

        //if (array.ptr_must_be_divisible_by_16 && (hip_ptr % 16 != 0)) {
        //  return absl::InvalidArgumentError(absl::StrFormat(
        //      "Parameter %zu (%p) is not divisible by 16.", i, ptr));
        //}

        if (array.bytes_to_zero > 0) {
          ROCM_RETURN_IF_ERROR(
              hipMemsetD8Async(hip_ptr, 0, array.bytes_to_zero, stream));
        }
        params.push_back(&ptr);
      } else {
        params.push_back(const_cast<uint64_t*>(&std::get<uint64_t>(param)));
      }
    }

    return kernel_.Launch(stream, grid_, params.data());
  }

 private:
  TritonKernel& kernel_;
  uint32_t grid_[3];
  std::vector<Parameter> parameters_;
};

class TritonAutotunedKernelCall : public TritonKernelCallBase {
 public:
  struct Config {
    py::object kernel_call;
    std::string description;
  };

  TritonAutotunedKernelCall(
      std::string name, std::vector<Config> configs,
      std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases)
      : name_(std::move(name)),
        configs_(std::move(configs)),
        input_output_aliases_(std::move(input_output_aliases)) {}

  absl::Status Launch(hipStream_t stream, void** buffers) override {
    absl::call_once(autotune_once_, [=]() {
      if (configs_.size() > 1) {
        autotune_status_ = Autotune(stream, buffers);
      }
    });
    RETURN_IF_ERROR(autotune_status_);
    auto& kernel_call = py::cast<TritonKernelCall&>(configs_[0].kernel_call);
    return kernel_call.Launch(stream, buffers);
  }

 private:
  static constexpr float kBenchmarkTimeMillis = 10.;

  absl::Status Autotune(hipStream_t stream, void** buffers) {
    // Ensure a valid context for driver calls that don't take the stream.
    hipCtx_t context;
    hipDevice_t device;
    int device_id = hipDeviceGet(&device, device_id);
    ROCM_RETURN_IF_ERROR(hipDevicePrimaryCtxRetain(&context, device));
    ROCM_RETURN_IF_ERROR(hipCtxPushCurrent(context));
    absl::Cleanup ctx_restorer = [] { hipCtxPopCurrent(nullptr); };

    // If an input aliases with an output, it will get overwritten during the
    // kernel execution. If the kernel is called repeatedly, as we do during
    // auto-tuning, the final result will be junk, so we take a copy of the
    // input to restore after auto-tuning.
    std::unordered_map<size_t, std::vector<uint8_t>> input_copies;
    for (auto [input_idx, output_idx, size] : input_output_aliases_) {
      if (buffers[input_idx] == buffers[output_idx]) {
        std::vector<uint8_t> input_copy(size);
        ROCM_RETURN_IF_ERROR(hipMemcpyDtoHAsync(
            input_copy.data(),
            reinterpret_cast<hipDeviceptr_t>(buffers[input_idx]), size, stream));
        input_copies[input_idx] = std::move(input_copy);
      }
    }

    LOG(INFO) << "Autotuning function: " << name_;
    // First run a single iteration of each to config to determine how many
    // iterations to run for benchmarking.
    float best = std::numeric_limits<float>::infinity();
    for (Config& config : configs_) {
      auto& kernel_call = py::cast<TritonKernelCall&>(config.kernel_call);
      absl::StatusOr<float> t = Benchmark(stream, kernel_call, buffers, 1);
      RETURN_IF_ERROR(t.status());
      LOG(INFO) << config.description << ", ran 1 iter in " << *t << " ms";
      best = std::min(best, *t);
    }

    int timed_iters =
        std::max(static_cast<int>(kBenchmarkTimeMillis / best), 1);
    if (timed_iters > 100) {
      timed_iters = 100;
      LOG(INFO) << "Benchmarking with 100 iters (capped at 100)";
    } else {
      timed_iters = std::min(timed_iters, 100);
      LOG(INFO) << "Benchmarking with " << timed_iters
                << " iters (target time: " << kBenchmarkTimeMillis << " ms)";
    }

    best = std::numeric_limits<float>::infinity();
    for (Config& config : configs_) {
      auto& kernel_call = py::cast<TritonKernelCall&>(config.kernel_call);
      absl::StatusOr<float> t =
          Benchmark(stream, kernel_call, buffers, timed_iters);
      RETURN_IF_ERROR(t.status());
      LOG(INFO) << config.description << ", ran " << timed_iters << " iters in "
                << *t << " ms";

      if (*t < best) {
        LOG(INFO) << config.description << " is the new best config";
        best = *t;
        std::swap(config, configs_[0]);
      }
    }

    // Discard all but the best config.
    py::gil_scoped_acquire gil;
    configs_.erase(configs_.begin() + 1, configs_.end());

    // Restore aliased inputs to their original values.
    for (auto [input_idx, _, size] : input_output_aliases_) {
      ROCM_RETURN_IF_ERROR(
          hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(buffers[input_idx]),
                            input_copies[input_idx].data(), size, stream));
    }
    // Synchronize stream to ensure copies are complete before the host copy
    // is deleted.
    return ROCM_TO_STATUS(hipStreamSynchronize(stream));
  }

  absl::StatusOr<float> Benchmark(hipStream_t stream,
                                  TritonKernelCall& kernel_call, void** buffers,
                                  int num_iterations) {
    hipEvent_t start, stop;
    ROCM_RETURN_IF_ERROR(hipEventCreateWithFlags(&start, /*Flags=*/hipEventDefault));
    ROCM_RETURN_IF_ERROR(hipEventCreateWithFlags(&stop, /*Flags=*/hipEventDefault));
    RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));  // Warm-up iteration.
    ROCM_RETURN_IF_ERROR(hipEventRecord(start, stream));
    for (int i = 0; i < num_iterations; ++i) {
      RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));
    }
    ROCM_RETURN_IF_ERROR(hipEventRecord(stop, stream));
    ROCM_RETURN_IF_ERROR(hipEventSynchronize(stop));
    float elapsed_ms;
    ROCM_RETURN_IF_ERROR(hipEventElapsedTime(&elapsed_ms, start, stop));
    ROCM_RETURN_IF_ERROR(hipEventDestroy(start));
    ROCM_RETURN_IF_ERROR(hipEventDestroy(stop));
    return elapsed_ms;
  }

  std::string name_;
  // After auto-tuning, all configurations, except the best, will be discarded.
  std::vector<Config> configs_;
  // (input buffer idx, output buffer idx, size)
  std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases_;
  absl::once_flag autotune_once_;
  absl::Status autotune_status_;
};

template <typename CppT, typename PyT>
uint64_t EncodeKernelParameterAs(PyT value) {
  static_assert(sizeof(CppT) <= sizeof(uint64_t));
  union {
    CppT value;
    uint64_t bits;
  } encoded;
  encoded.bits = 0;
  encoded.value = CppT(value);
  return encoded.bits;
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::int_ value,
                                               std::string_view dtype) {
  if ((dtype == "i1") || (dtype == "i8")) {
    return EncodeKernelParameterAs<int8_t>(value);
  } else if (dtype == "u8") {
    return EncodeKernelParameterAs<uint8_t>(value);
  } else if (dtype == "i16") {
    return EncodeKernelParameterAs<int16_t>(value);
  } else if (dtype == "u16") {
    return EncodeKernelParameterAs<uint16_t>(value);
  } else if (dtype == "i32") {
    return EncodeKernelParameterAs<int32_t>(value);
  } else if (dtype == "u32") {
    return EncodeKernelParameterAs<uint32_t>(value);
  } else if (dtype == "i64") {
    return EncodeKernelParameterAs<int64_t>(value);
  } else if (dtype == "u64") {
    return EncodeKernelParameterAs<uint64_t>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::float_ value,
                                               std::string_view dtype) {
  if (dtype == "fp32") {
    return EncodeKernelParameterAs<float>(value);
  } else if (dtype == "fp64") {
    return EncodeKernelParameterAs<double>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::bool_ value,
                                               std::string_view dtype) {
  if ((dtype == "int1") || (dtype == "B")) {
    return EncodeKernelParameterAs<bool>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

}  // namespace

void LaunchTritonKernel(hipStream_t stream, void** buffers, char* opaque,
                        size_t opaque_len) {
CHECK_EQ(opaque_len, sizeof(TritonKernelCallBase *));
TritonKernelCallBase *kernel_call;
std::memcpy(&kernel_call, opaque, sizeof(TritonKernelCallBase *));
absl::Status status = kernel_call->Launch(stream, buffers);
LOG_IF(FATAL, !status.ok()) << status; // TODO(cjfj): Return the `Status`.
}

PYBIND11_MODULE(triton_kernel_call_lib, m) {
  py::class_<TritonKernel>(m, "TritonKernel")
      .def(py::init<std::string, std::string, uint32_t, uint32_t>());

  py::class_<TritonKernelCall>(m, "TritonKernelCall")
      .def(py::init<TritonKernel&, uint32_t, uint32_t, uint32_t,
                    std::vector<TritonKernelCall::Parameter>>(),
           py::keep_alive<1, 2>())  // Ensure that the kernel lives long enough.
      .def_property_readonly("descriptor", [](TritonKernelCall& kernel_call) {
        union {
          TritonKernelCall* ptr;
          char bytes[sizeof(TritonKernelCall*)];
        } descriptor;
        descriptor.ptr = &kernel_call;
        return py::bytes(descriptor.bytes, sizeof(TritonKernelCall*));
      });

  py::class_<TritonKernelCall::ArrayParameter>(m, "TritonArrayParameter");

  py::class_<TritonAutotunedKernelCall>(m, "TritonAutotunedKernelCall")
      .def(py::init<>([](std::string name,
                         std::vector<std::pair<py::object, std::string>>
                             calls_and_descriptions,
                         std::vector<std::tuple<size_t, size_t, size_t>>
                             input_output_aliases) {
        std::vector<TritonAutotunedKernelCall::Config> configs;
        configs.reserve(calls_and_descriptions.size());
        for (auto& [kernel_call, desc] : calls_and_descriptions) {
          configs.push_back({std::move(kernel_call), std::move(desc)});
        }
        return std::make_unique<TritonAutotunedKernelCall>(
            std::move(name), std::move(configs),
            std::move(input_output_aliases));
      }))
      .def_property_readonly(
          "descriptor", [](TritonAutotunedKernelCall& kernel_call) {
            union {
              TritonAutotunedKernelCall* ptr;
              char bytes[sizeof(TritonAutotunedKernelCall*)];
            } descriptor;
            descriptor.ptr = &kernel_call;
            return py::bytes(descriptor.bytes,
                             sizeof(TritonAutotunedKernelCall*));
          });

  m.def("get_custom_call", [] {
    return py::capsule(reinterpret_cast<void*>(&LaunchTritonKernel),
                       "xla._CUSTOM_CALL_TARGET");
  });

  m.def("create_array_parameter",
        [](size_t bytes_to_zero, bool ptr_must_be_divisible_by_16) {
          return TritonKernelCall::ArrayParameter{bytes_to_zero,
                                                  ptr_must_be_divisible_by_16};
        });
  m.def("create_scalar_parameter",
        py::overload_cast<py::int_, std::string_view>(&EncodeKernelParameter));
  m.def(
      "create_scalar_parameter",
      py::overload_cast<py::float_, std::string_view>(&EncodeKernelParameter));
  m.def("create_scalar_parameter",
        py::overload_cast<py::bool_, std::string_view>(&EncodeKernelParameter));
  m.def("get_compute_capability", [](int device) -> absl::StatusOr<int> {
    int major, minor;
    ROCM_RETURN_IF_ERROR(hipInit(device));
    ROCM_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, device));
    ROCM_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, device));
    return major * 10 + minor;
    //return 80;
  });
}

}  // namespace jax_triton
