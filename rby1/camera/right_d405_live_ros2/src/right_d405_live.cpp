#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <librealsense2/rs.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>

class RightD405Live : public rclcpp::Node
{
public:
  RightD405Live() : Node("right_d405_live")
  {
    const std::string serial = "230422272237";
    auto qos = rclcpp::SensorDataQoS();

    color_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/right_d405/color/image_raw", qos);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/right_d405/aligned_depth_to_color/image_raw", qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "/right_d405/color/camera_info", qos);

    cfg_.enable_device(serial);
    cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    cfg_.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    RCLCPP_INFO(get_logger(), "Opening right D405 serial %s", serial.c_str());
    profile_ = pipe_.start(cfg_);
    align_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);

    auto color_profile =
        profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    intr_ = color_profile.get_intrinsics();

    auto device = profile_.get_device();
    auto depth_sensor = device.first<rs2::depth_sensor>();
    depth_scale_ = depth_sensor.get_depth_scale();

    RCLCPP_INFO(
        get_logger(),
        "Color intrinsics: fx=%.6f fy=%.6f cx=%.6f cy=%.6f",
        intr_.fx, intr_.fy, intr_.ppx, intr_.ppy);
    RCLCPP_INFO(get_logger(), "Depth scale: %.9f m/unit", depth_scale_);
    RCLCPP_INFO(
        get_logger(),
        "Publishing:\n  /right_d405/color/image_raw"
        "\n  /right_d405/aligned_depth_to_color/image_raw"
        "\n  /right_d405/color/camera_info");

    for (int i = 0; i < 30; ++i)
      pipe_.wait_for_frames();

    timer_ = create_wall_timer(
        std::chrono::milliseconds(1),
        std::bind(&RightD405Live::capture, this));
  }

  ~RightD405Live()
  {
    try { pipe_.stop(); } catch (...) {}
  }

private:
  void capture()
  {
    rs2::frameset frames;
    if (!pipe_.poll_for_frames(&frames)) return;

    auto aligned = align_->process(frames);
    rs2::video_frame color = aligned.get_color_frame();
    rs2::depth_frame depth = aligned.get_depth_frame();
    if (!color || !depth) return;

    const auto stamp = now();
    publish_color(color, stamp);
    publish_depth(depth, stamp);
    publish_camera_info(stamp);
  }

  void publish_color(const rs2::video_frame &frame, const rclcpp::Time &stamp)
  {
    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "right_d405_optical_frame";
    msg.height = frame.get_height();
    msg.width = frame.get_width();
    msg.encoding = "rgb8";
    msg.is_bigendian = false;
    msg.step = msg.width * 3;
    msg.data.resize(msg.step * msg.height);

    const uint8_t *src = static_cast<const uint8_t *>(frame.get_data());
    const size_t src_stride = frame.get_stride_in_bytes();
    for (size_t y = 0; y < msg.height; ++y)
      std::memcpy(msg.data.data() + y * msg.step,
                  src + y * src_stride,
                  msg.step);

    color_pub_->publish(msg);
  }

  void publish_depth(const rs2::depth_frame &frame, const rclcpp::Time &stamp)
  {
    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "right_d405_optical_frame";
    msg.height = frame.get_height();
    msg.width = frame.get_width();
    msg.encoding = "16UC1";
    msg.is_bigendian = false;
    msg.step = msg.width * sizeof(uint16_t);
    msg.data.resize(msg.step * msg.height);

    const uint8_t *src = static_cast<const uint8_t *>(frame.get_data());
    const size_t src_stride = frame.get_stride_in_bytes();
    for (size_t y = 0; y < msg.height; ++y)
      std::memcpy(msg.data.data() + y * msg.step,
                  src + y * src_stride,
                  msg.step);

    depth_pub_->publish(msg);
  }

  void publish_camera_info(const rclcpp::Time &stamp)
  {
    sensor_msgs::msg::CameraInfo msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "right_d405_optical_frame";
    msg.width = 640;
    msg.height = 480;
    msg.k = {
        intr_.fx, 0.0, intr_.ppx,
        0.0, intr_.fy, intr_.ppy,
        0.0, 0.0, 1.0};
    msg.p = {
        intr_.fx, 0.0, intr_.ppx, 0.0,
        0.0, intr_.fy, intr_.ppy, 0.0,
        0.0, 0.0, 1.0, 0.0};
    msg.d = {
        intr_.coeffs[0], intr_.coeffs[1], intr_.coeffs[2],
        intr_.coeffs[3], intr_.coeffs[4]};
    msg.distortion_model = "plumb_bob";
    info_pub_->publish(msg);
  }

  rs2::pipeline pipe_;
  rs2::config cfg_;
  rs2::pipeline_profile profile_;
  std::unique_ptr<rs2::align> align_;
  rs2_intrinsics intr_{};
  float depth_scale_{0.001f};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<RightD405Live>();
    rclcpp::spin(node);
  } catch (const rs2::error &e) {
    RCLCPP_ERROR(
        rclcpp::get_logger("right_d405_live"),
        "RealSense error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  } catch (const std::exception &e) {
    RCLCPP_ERROR(
        rclcpp::get_logger("right_d405_live"),
        "Error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
