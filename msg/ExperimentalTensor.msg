# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DLPack-aligned tensor message.
#
# Field layout matches DLTensor from https://dmlc.github.io/dlpack/latest/
# so any DLPack-compatible framework can round-trip losslessly through a
# thin wrapper. Frameworks with first-party DLPack support include:
#
#   Deep-learning frameworks: PyTorch, TensorFlow, JAX, MXNet, PaddlePaddle
#   Array libraries:          NumPy (>= 1.23), CuPy
#   Compilers / runtimes:     TVM, ONNX Runtime
#
# See https://dmlc.github.io/dlpack/latest/ for the canonical, up-to-date
# list.
#
#   dtype  = {dtype_code, dtype_bits, dtype_lanes}   (DLDataType)
#   shape / strides / byte_offset / data             (DLTensor)

# DLDataType.
#
# `dtype_code` follows DLPack's DLDataTypeCode enum:
#    0 = Int                  signed integer
#    1 = UInt                 unsigned integer
#    2 = Float                IEEE-754 floating point (fp16, fp32, fp64, ...)
#    3 = OpaqueHandle         pointer-sized opaque handle (rarely transported)
#    4 = BFloat               brain floating point (bf16)
#    5 = Complex              complex number (real+imag pair)
#    6 = Bool                 boolean (1 byte)
#    7 = Float8_e3m4          FP8 sub-byte / sub-format families
#    8 = Float8_e4m3
#    9 = Float8_e4m3b11fnuz
#   10 = Float8_e4m3fn
#   11 = Float8_e4m3fnuz
#   12 = Float8_e5m2
#   13 = Float8_e5m2fnuz
#   14 = Float8_e8m0fnu
#   15 = Float6_e2m3fn
#   16 = Float6_e3m2fn
#   17 = Float4_e2m1fn
# Codes >= 7 are DLPack v1.x additions; consumers should treat unknown codes
# as opaque and surface them as an unsupported-dtype error.
#
# `dtype_bits` is the scalar bit width (8, 16, 32, 64, ...).
# `dtype_lanes` is the vector lane count (1 for a plain scalar).
uint8  dtype_code
uint8  dtype_bits
uint16 dtype_lanes

# DLTensor shape / stride / offset.
# `strides` may be empty, which (per DLPack convention) means the tensor is
# contiguous and the receiver should infer row-major strides from `shape`.
# `byte_offset` is the offset into `data` where the tensor's first element
# lives; nonzero values allow zero-copy transport of views / sliced tensors
# that share underlying storage.
int64[]  shape
int64[]  strides
uint64   byte_offset

# Underlying storage. May be larger than `prod(shape) * dtype_bits * dtype_lanes / 8`
# if the message carries a view into a larger allocation.
# Whichever buffer backend is registered owns the bytes.
uint8[] data
