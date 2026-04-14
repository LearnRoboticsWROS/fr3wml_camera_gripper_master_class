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
find_package(tf2_geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(fairino_bridge REQUIRED)

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

add_executable(scene_loader_bottle_capsule src/scene_loader_bottle_capsule.cpp)
ament_target_dependencies(scene_loader_bottle_capsule
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  geometry_msgs
  geometric_shapes
)

add_executable(pick_place_bottle_capsule_framework src/pick_place_bottle_capsule_framework.cpp)
ament_target_dependencies(pick_place_bottle_capsule_framework
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  geometry_msgs
  geometric_shapes
  fairino_bridge
  std_srvs
)

add_executable(scene_handling src/scene_handling.cpp)
ament_target_dependencies(scene_handling
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  geometry_msgs
  geometric_shapes
  std_srvs
)

add_executable(pick_place_framework src/pick_place_framework.cpp)
ament_target_dependencies(pick_place_framework
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
  fairino_bridge
  std_srvs
)

install(TARGETS
  fr3wml_ik_test
  scene_loader_bottle_capsule
  pick_place_bottle_capsule_framework
  scene_handling
  pick_place_framework
  DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME})


ament_package()
