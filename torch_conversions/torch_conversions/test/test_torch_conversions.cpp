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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "torch_conversions/torch_conversions.hpp"

using torch_conversions::TensorMsg;
using torch_conversions::allocate_tensor_msg;
using torch_conversions::from_input_tensor_msg;
using torch_conversions::from_output_tensor_msg;
using torch_conversions::to_tensor_msg;

// Note: kDLCPU, kDLUInt, kDLInt, kDLFloat are enumerators of DLDataTypeCode /
// DLDeviceType declared at global scope by <ATen/dlpack.h> (transitively
// included by torch_conversions.hpp), so they are already visible here.

TEST(TorchTensorBridge, AllocateCpuTensorPopulatesDlpackMetadata)
{
  auto msg = allocate_tensor_msg({2, 3, 4}, at::kFloat, c10::kCPU);

  ASSERT_EQ(msg->shape.size(), 3u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->shape[2], 4);

  ASSERT_EQ(msg->strides.size(), 3u);
  EXPECT_EQ(msg->strides[0], 12);
  EXPECT_EQ(msg->strides[1], 4);
  EXPECT_EQ(msg->strides[2], 1);

  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->dtype_lanes, 1u);

  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 2u * 3u * 4u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(TorchTensorBridge, ByteDtypeRoundTripsThroughDlpackTriple)
{
  auto msg = allocate_tensor_msg({5}, at::kByte, c10::kCPU);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLUInt));
  EXPECT_EQ(msg->dtype_bits, 8u);
  EXPECT_EQ(msg->dtype_lanes, 1u);
}

TEST(TorchTensorBridge, Int32DtypeRoundTripsThroughDlpackTriple)
{
  auto msg = allocate_tensor_msg({4}, at::kInt, c10::kCPU);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLInt));
  EXPECT_EQ(msg->dtype_bits, 32u);
}

TEST(TorchTensorBridge, WriteThenReadRoundTrip)
{
  auto msg = allocate_tensor_msg({4}, at::kInt, c10::kCPU);

  {
    at::Tensor t = from_output_tensor_msg(*msg);
    ASSERT_TRUE(t.defined());
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{4}));
    EXPECT_EQ(t.scalar_type(), at::kInt);
    t.copy_(torch::tensor({10, 20, 30, 40}, at::kInt));
  }

  at::Tensor v = from_input_tensor_msg(
    *msg, /*clone=*/false);
  ASSERT_TRUE(v.defined());
  ASSERT_EQ(v.numel(), 4);
  auto * p = v.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 10);
  EXPECT_EQ(p[1], 20);
  EXPECT_EQ(p[2], 30);
  EXPECT_EQ(p[3], 40);
}

TEST(TorchTensorBridge, ToTensorMsgCopiesAndUpdatesMetadata)
{
  auto msg = allocate_tensor_msg({16}, at::kFloat, c10::kCPU);

  at::Tensor src = torch::arange(0, 6, at::kFloat).reshape({2, 3});
  to_tensor_msg(*msg, src);

  ASSERT_EQ(msg->shape.size(), 2u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->byte_offset, 0u);

  at::Tensor round = from_input_tensor_msg(
    *msg, /*clone=*/true);
  ASSERT_EQ(round.numel(), 6);
  EXPECT_TRUE(torch::equal(round.flatten(), src.flatten()));
}

TEST(TorchTensorBridge, ToTensorMsgAllocatesAndCopies)
{
  at::Tensor src = torch::arange(0, 6, at::kFloat).reshape({2, 3});
  auto msg = to_tensor_msg(src);

  ASSERT_EQ(msg->shape.size(), 2u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");

  at::Tensor round = from_input_tensor_msg(*msg, /*clone=*/true);
  ASSERT_EQ(round.numel(), 6);
  EXPECT_TRUE(torch::equal(round.flatten(), src.flatten()));
}

TEST(TorchTensorBridge, ByteOffsetSelectsSubregionOfStorage)
{
  // Allocate 16 ints but publish only a 4-int view starting at index 4.
  auto msg = allocate_tensor_msg({16}, at::kInt, c10::kCPU);
  {
    at::Tensor full = from_output_tensor_msg(*msg);
    for (int i = 0; i < 16; ++i) {
      full.index_put_({i}, i * 100);
    }
  }

  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4 * sizeof(int32_t);

  at::Tensor view = from_input_tensor_msg(
    *msg, /*clone=*/false);
  ASSERT_EQ(view.numel(), 4);
  auto * p = view.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 400);
  EXPECT_EQ(p[1], 500);
  EXPECT_EQ(p[2], 600);
  EXPECT_EQ(p[3], 700);
}

TEST(TorchTensorBridge, ToTensorMsgRejectsOversizedTensor)
{
  auto msg = allocate_tensor_msg({4}, at::kByte, c10::kCPU);
  at::Tensor big = torch::zeros({128}, at::kByte);
  EXPECT_THROW(to_tensor_msg(*msg, big), std::runtime_error);
}

TEST(TorchTensorBridge, EmptyDataReturnsUndefinedTensor)
{
  TensorMsg msg;
  EXPECT_FALSE(from_input_tensor_msg(msg).defined());
  EXPECT_FALSE(from_output_tensor_msg(msg).defined());
}

TEST(TorchTensorBridge, DtypeConversionRejectsUnsupportedTriple)
{
  using torch_conversions::detail::DLDataType;
  using torch_conversions::detail::scalar_from_dl_dtype;
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 128, 1}), std::runtime_error);
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 32, 4}), std::runtime_error);
}

TEST(TorchTensorBridge, MakeInputDlpackPopulatesDlTensorFields)
{
  auto msg = allocate_tensor_msg({2, 3}, at::kFloat, c10::kCPU);
  {
    at::Tensor t = torch_conversions::from_output_tensor_msg(*msg);
    t.copy_(torch::arange(0, 6, at::kFloat).reshape({2, 3}));
  }

  auto * dlm = torch_conversions::detail::make_input_dlpack(*msg);
  ASSERT_NE(dlm, nullptr);
  ASSERT_NE(dlm->deleter, nullptr);

  EXPECT_EQ(dlm->dl_tensor.ndim, 2);
  ASSERT_NE(dlm->dl_tensor.shape, nullptr);
  EXPECT_EQ(dlm->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlm->dl_tensor.shape[1], 3);
  ASSERT_NE(dlm->dl_tensor.strides, nullptr);
  EXPECT_EQ(dlm->dl_tensor.strides[0], 3);
  EXPECT_EQ(dlm->dl_tensor.strides[1], 1);

  EXPECT_EQ(dlm->dl_tensor.dtype.code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(dlm->dl_tensor.dtype.bits, 32u);
  EXPECT_EQ(dlm->dl_tensor.dtype.lanes, 1u);

  EXPECT_EQ(static_cast<int32_t>(dlm->dl_tensor.device.device_type),
    static_cast<int32_t>(kDLCPU));
  EXPECT_EQ(dlm->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlm->dl_tensor.byte_offset, 0u);
  EXPECT_NE(dlm->dl_tensor.data, nullptr);

  dlm->deleter(dlm);
}

TEST(TorchTensorBridge, MakeInputDlpackWithByteOffset)
{
  auto msg = allocate_tensor_msg({16}, at::kInt, c10::kCPU);
  {
    at::Tensor full = torch_conversions::from_output_tensor_msg(*msg);
    for (int i = 0; i < 16; ++i) {
      full.index_put_({i}, i * 100);
    }
  }

  // Capture the base pointer of the allocation, then publish a 4-element
  // view starting at index 4 (16 bytes in).
  auto * base = static_cast<const uint8_t *>(msg->data.data());
  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4 * sizeof(int32_t);

  auto * dlm = torch_conversions::detail::make_input_dlpack(*msg);
  ASSERT_NE(dlm, nullptr);
  EXPECT_EQ(dlm->dl_tensor.ndim, 1);
  EXPECT_EQ(dlm->dl_tensor.shape[0], 4);

  // The bridge bakes msg->byte_offset into DLTensor::data and sets
  // DLTensor::byte_offset to 0 (portable across DLPack importers that
  // ignore the byte_offset field).
  EXPECT_EQ(dlm->dl_tensor.byte_offset, 0u);
  EXPECT_EQ(static_cast<const uint8_t *>(dlm->dl_tensor.data),
    base + 4 * sizeof(int32_t));

  dlm->deleter(dlm);
}

TEST(TorchTensorBridge, DlpackPtrFreesOnScopeExit)
{
  auto msg = allocate_tensor_msg({3}, at::kInt, c10::kCPU);

  {
    torch_conversions::detail::DlpackPtr holder{
      torch_conversions::detail::make_input_dlpack(*msg)};
    ASSERT_TRUE(holder);
    EXPECT_EQ(holder->dl_tensor.ndim, 1);
    EXPECT_EQ(holder->dl_tensor.shape[0], 3);
  }  // holder destructor invokes deleter; no leak.

  // A second one, this time handed off via release() (simulating a
  // framework's from_dlpack taking ownership).
  torch_conversions::detail::DlpackPtr holder2{
    torch_conversions::detail::make_input_dlpack(*msg)};
  DLManagedTensor * raw = holder2.release();
  ASSERT_NE(raw, nullptr);
  raw->deleter(raw);  // caller takes over ownership
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
