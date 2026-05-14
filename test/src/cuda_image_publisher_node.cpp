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

#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"

class CudaImagePublisher : public rclcpp::Node
{
public:
  explicit CudaImagePublisher(const rclcpp::NodeOptions & options)
  : Node("cuda_image_publisher", options), count_(0)
  {
    this->declare_parameter<int>("max_publish_count", 0);
    this->declare_parameter<int>("publish_rate_ms", 200);
    this->declare_parameter<int>("image_width", 8);
    this->declare_parameter<int>("image_height", 8);

    max_publish_count_ = this->get_parameter("max_publish_count").as_int();
    int publish_rate_ms = this->get_parameter("publish_rate_ms").as_int();
    image_width_ = this->get_parameter("image_width").as_int();
    image_height_ = this->get_parameter("image_height").as_int();

    cudaStreamCreate(&stream_);

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("test_cuda_image", 10);
    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("publisher_count", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_rate_ms),
      std::bind(&CudaImagePublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
      "CUDA image publisher started (max_publish_count=%d, rate=%dms)",
      max_publish_count_, publish_rate_ms);
  }

  ~CudaImagePublisher() override
  {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

private:
  void timer_callback()
  {
    if (max_publish_count_ > 0 && count_ >= static_cast<size_t>(max_publish_count_)) {
      timer_->cancel();
      RCLCPP_INFO(this->get_logger(),
        "Reached max_publish_count (%d), stopping publisher", max_publish_count_);
      return;
    }

    const size_t data_size = image_width_ * image_height_ * 3;

    sensor_msgs::msg::Image msg;
    msg.data = cuda_buffer_backend::allocate_buffer(data_size);
    msg.header.stamp = this->now();
    msg.header.frame_id = "cuda_test_frame";
    msg.height = image_height_;
    msg.width = image_width_;
    msg.encoding = "rgb8";
    msg.step = image_width_ * 3;
    msg.is_bigendian = 0;

    {
      cuda_buffer_backend::WriteHandle write_handle =
        cuda_buffer_backend::from_output_buffer(msg.data, stream_);
      cudaMemsetAsync(write_handle.get_ptr(), static_cast<int>(count_ % 256),
        data_size, stream_);
    }

    publisher_->publish(msg);

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = ++count_;
    count_publisher_->publish(count_msg);

    if (count_ % 10 == 0) {
      RCLCPP_INFO(this->get_logger(),
                  "Published %zu CUDA images (backend: %s)",
                  count_, msg.data.get_backend_type().c_str());
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cudaStream_t stream_{nullptr};
  size_t count_;
  int max_publish_count_;
  int image_width_;
  int image_height_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(CudaImagePublisher)
