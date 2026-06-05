// Copyright 2026 Open Source Robotics Foundation, Inc.
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
// limitations under the License.

#ifndef CUDA_BUFFER__CUDA_ERROR_HPP_
#define CUDA_BUFFER__CUDA_ERROR_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <rcutils/logging_macros.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda_buffer_backend
{

class CudaError : public std::runtime_error
{
public:
  explicit CudaError(const std::string & message)
  : std::runtime_error(message) {}

  CudaError(const char * file, int line, const char * expr, cudaError_t err)
  : std::runtime_error(
      format_error(file, line, expr, cudaGetErrorName(err), cudaGetErrorString(err))) {}

  CudaError(const char * file, int line, const char * expr, CUresult err)
  : std::runtime_error(format_driver_error(file, line, expr, err)) {}

private:
  static std::string format_error(
    const char * file, int line, const char * expr,
    const char * name, const char * desc)
  {
    std::ostringstream oss;
    oss << file << ":" << line << ": " << expr << " failed: "
        << name << " (" << desc << ")";
    return oss.str();
  }

  static std::string format_driver_error(
    const char * file, int line, const char * expr, CUresult err)
  {
    const char * name = nullptr;
    const char * desc = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    return format_error(
      file, line, expr,
      name ? name : "UNKNOWN",
      desc ? desc : "no description");
  }
};

inline bool cuda_error_is_safe(cudaError_t e)
{
  return e == cudaSuccess || e == cudaErrorCudartUnloading;
}

}  // namespace cuda_buffer_backend

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
      throw ::cuda_buffer_backend::CudaError(__FILE__, __LINE__, #expr, _err); \
    } \
  } while (0)

#define CUDA_CHECK_NOTHROW(expr, on_error) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend", \
        "%s:%d: %s failed: %s (%s)", \
        __FILE__, __LINE__, #expr, \
        cudaGetErrorName(_err), cudaGetErrorString(_err)); \
      (void)cudaGetLastError(); \
      on_error; \
    } \
  } while (0)

#endif  // CUDA_BUFFER__CUDA_ERROR_HPP_
