# torch_conversions (DLPack-aligned)

Header-only helper library that converts between a DLPack-shaped ROS 2
message (`tensor_msgs/ExperimentalTensor`) and an `at::Tensor`, riding on top of
whichever `rosidl::Buffer` storage backend is registered at runtime.

The message schema follows [DLPack](https://dmlc.github.io/dlpack/latest/)
tensor metadata, so any DLPack-compatible framework (PyTorch, TensorFlow,
JAX, CuPy, ONNX Runtime, MXNet, RAPIDS, ...) can plug in via a thin wrapper
and interoperate over the wire without re-encoding shape / dtype metadata.

> **Status: experimental.** The message is named `ExperimentalTensor` on
> purpose. The schema is used internally to validate the buffer-backend
> design and may change before it is renamed to `Tensor` and stabilized.

## Packages

| Package | Description |
|---|---|
| `tensor_msgs` | `ExperimentalTensor.msg` definition: DLPack-aligned `{dtype_code, dtype_bits, dtype_lanes}`, `shape[]`, `strides[]`, `byte_offset`, `data[]`. |
| `torch_conversions` | Header-only library: allocation, `at::Tensor` ↔ `ExperimentalTensor.msg` conversion, DLPack export, and CUDA stream helpers. |

The `uint8[] data` field maps to `rosidl::Buffer<uint8_t>`,
so storage and transport are delegated to whichever buffer backend is
registered for the connection.

## The `ExperimentalTensor.msg` schema

```
# DLDataType
uint8  dtype_code        # DLPack DLDataTypeCode: 0=Int, 1=UInt, 2=Float, 4=BFloat, 6=Bool, ...
uint8  dtype_bits        # 8, 16, 32, 64, ...
uint16 dtype_lanes       # SIMD lanes; 1 for plain scalar

# DLTensor
int64[] shape
int64[] strides          # empty = contiguous (DLPack nullptr convention)
uint64  byte_offset      # view offset into `data`

# Underlying storage (may be larger than numel * element_size for views)
uint8[] data
```

The message carries DLPack's dtype / shape / stride / offset metadata. The
`DLDevice` fields are derived from the underlying `msg.data` buffer backend.

## Build

```bash
# CUDA path (recommended): build cuda_buffer_backend first.
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh

colcon build --symlink-install --packages-up-to torch_conversions
source install/setup.sh
```

## Testing

```bash
colcon test --packages-select tensor_msgs torch_conversions
colcon test-result --verbose
```

## Examples

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

void timer_cb()
{
  // Uses a non-default CUDA stream when CUDA is available; no-op on CPU.
  auto guard = torch_conversions::set_stream();

  // Pre-sizes msg.data and fills DLPack shape / dtype metadata.
  // Uses the accelerated buffer backend when available, otherwise CPU.
  auto msg = torch_conversions::allocate_tensor_msg(
    {height, width, 3}, torch::kByte);

  {
    // Output path: writable tensor view that aliases msg.data.
    at::Tensor t = torch_conversions::from_output_tensor_msg(*msg);
    render_pipeline(t);
  }

  publisher_->publish(std::move(msg));
}
```

### Subscriber

```cpp
void cb(const tensor_msgs::msg::ExperimentalTensor::SharedPtr msg)
{
  // Uses the same stream discipline as the publisher side.
  auto guard = torch_conversions::set_stream();

  // Default clone=true: independent tensor, safe to mutate.
  at::Tensor in = torch_conversions::from_input_tensor_msg(*msg);
  auto out = model_(in);

  // Or clone=false for a zero-copy read-only view.
  at::Tensor view = torch_conversions::from_input_tensor_msg(*msg, /*clone=*/false);
}
```

### Publishing an existing tensor

```cpp
at::Tensor t = compute_something().contiguous();

// Allocates a Tensor message, copies tensor data, and fills metadata.
auto msg = torch_conversions::to_tensor_msg(t);
publisher_->publish(std::move(msg));
```

`to_tensor_msg(t)` allocates a message, copies tensor data into `msg.data`,
and updates shape / strides / dtype metadata to match `t`. Use
`to_tensor_msg(*msg, t)` when you want to reuse a pre-sized message buffer.

## License

Apache-2.0
