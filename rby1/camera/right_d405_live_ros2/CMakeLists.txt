cmake_minimum_required(VERSION 3.8)
project(right_d405_live)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(
  right_d405_live_node
  src/right_d405_live.cpp
)

ament_target_dependencies(
  right_d405_live_node
  rclcpp
  sensor_msgs
)

# Force the working local librealsense installation on the Nvidia U-PC.
target_include_directories(
  right_d405_live_node
  PRIVATE
  /usr/local/include
)

target_link_libraries(
  right_d405_live_node
  /usr/local/lib/librealsense2.so
)

set_target_properties(
  right_d405_live_node
  PROPERTIES
  BUILD_RPATH "/usr/local/lib"
  INSTALL_RPATH "/usr/local/lib"
)

install(
  TARGETS right_d405_live_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
