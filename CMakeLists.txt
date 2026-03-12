cmake_minimum_required(VERSION 3.8)
project(fr3wml_camera_gripper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(fairino_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(fr3wml_ik_test src/fr3wml_ik_test.cpp)
ament_target_dependencies(fr3wml_ik_test rclcpp moveit_ros_planning_interface std_msgs trajectory_msgs std_srvs fairino_msgs)

install(TARGETS 
  fr3wml_ik_test 
  DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME})


ament_package()
