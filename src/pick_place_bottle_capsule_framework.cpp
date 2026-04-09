#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mutex>

#include <Eigen/Geometry>
#include <boost/variant/get.hpp>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>

#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>

#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>
#include <sstream>

#include "fairino_bridge/srv/execute_pose_motion.hpp"

using MoveLClient = rclcpp::Client<fairino_bridge::srv::ExecutePoseMotion>;

using moveit::planning_interface::MoveGroupInterface;

// Scene object IDs
static const std::string BOTTLE_OBJECT_ID  = "bottle";
static const std::string CAPSULE_OBJECT_ID = "capsule";

// End-effector links that should be allowed to collide with scene objects
static const std::vector<std::string> END_EFFECTOR_LINKS = {
    "softgripper_link", "suctioncup_link", "gripper_body", "suction_cap"
};

// ============================================================================
// ============================= CONFIG STRUCT =================================
// ============================================================================

struct ExecutorConfig
{
  // Core MoveIt
  std::string planning_group;
  std::string home_named_target;
  std::string bottle_mesh_resource;

  // Optional perception integration
  bool use_perception_pipeline;
  std::string capsule_pick_pose_topic;
  double perception_pose_timeout_sec;

  // Joint targets in DEG (loaded from params, converted to rad at runtime)
  std::vector<double> pre_pick_bottle_joint_deg;
  std::vector<double> pre_pick_capsule_joint_deg;
  std::vector<double> pre_place_bottle_joint_deg;

  // Tool pick-frame wrt wrist3_link (full transform)
  std::vector<double> softgripper_pick_frame_wrt_wrist_xyz;
  std::vector<double> softgripper_pick_frame_wrt_wrist_rpy_deg;

  std::vector<double> suctioncup_pick_frame_wrt_wrist_xyz;
  std::vector<double> suctioncup_pick_frame_wrt_wrist_rpy_deg;

  // Object origin wrt tool pick-frame (full transform)
  std::vector<double> bottle_origin_wrt_softgripper_pick_frame_xyz;
  std::vector<double> bottle_origin_wrt_softgripper_pick_frame_rpy_deg;

  std::vector<double> capsule_origin_wrt_suctioncup_pick_frame_xyz;
  std::vector<double> capsule_origin_wrt_suctioncup_pick_frame_rpy_deg;

  // Runtime task points in base_link — wrist3_link positions (NOT tool-tip)
  // Bottle uses position only; orientation comes from validated joint poses.
  std::vector<double> bottle_pick_position_xyz;
  std::vector<double> bottle_place_position_xyz;

  // Blind-mode capsule pick pose in base_link (wrist3_link position)
  std::vector<double> capsule_pick_position_xyz;
  std::vector<double> capsule_pick_rpy_deg;

  // Capsule place pose in base_link (wrist3_link position)
  std::vector<double> capsule_place_position_xyz;
  std::vector<double> capsule_place_rpy_deg;

  // Capsule primitive dimensions
  double capsule_radius;
  double capsule_height;

  // --- PTP (MoveJ) tuning ---
  double planning_time_sec;
  int num_planning_attempts;
  int max_free_space_retries;
  double ptp_velocity_scaling;
  double ptp_acceleration_scaling;
  double goal_position_tolerance;
  double goal_orientation_tolerance;
  double joint_target_tolerance;

  // --- LIN (MoveL) tuning per phase ---
  // Each pick/place phase has its own descend speed, retreat speed, and retreat distance.
  // Speeds are in percent (1-100), distances in meters.
  double bottle_pick_speed_percent;
  double bottle_pick_retreat_speed_percent;
  double bottle_pick_retreat_distance_m;

  double bottle_place_speed_percent;
  double bottle_place_retreat_speed_percent;
  double bottle_place_retreat_distance_m;

  double capsule_pick_speed_percent;
  double capsule_pick_retreat_speed_percent;
  double capsule_pick_retreat_distance_m;

  double capsule_place_speed_percent;
  double capsule_place_retreat_speed_percent;
  double capsule_place_retreat_distance_m;

  // Wrist3_link-to-flange offset (along wrist3_link Z axis)
  // The Fairino controller MoveL operates in flange coordinates, but user
  // specifies wrist3_link positions. This offset converts between the two.
  double wrist3_to_flange_z;

  // Timing
  int scene_update_wait_ms;
  int retry_sleep_ms;

  // Small numerical safety offset when spawning objects
  double world_spawn_safety_z_m;
};

struct PerceptionState
{
  std::mutex mutex;
  bool has_capsule_pick_pose = false;
  geometry_msgs::msg::PoseStamped latest_capsule_pick_pose;
};

// ============================================================================
// ============================= BASIC HELPERS =================================
// ============================================================================

void validateVectorSize(const std::vector<double>& vec, std::size_t expected_size, const std::string& name)
{
  if (vec.size() != expected_size) {
    throw std::runtime_error(
      "Parameter '" + name + "' must contain exactly " + std::to_string(expected_size) + " values.");
  }
}

double deg2rad(double deg)
{
  return deg * M_PI / 180.0;
}

std::vector<double> degVectorToRad(const std::vector<double>& deg_values)
{
  std::vector<double> rad_values;
  rad_values.reserve(deg_values.size());
  for (double v : deg_values) {
    rad_values.push_back(deg2rad(v));
  }
  return rad_values;
}

Eigen::Vector3d vec3(const std::vector<double>& v, const std::string& name)
{
  validateVectorSize(v, 3, name);
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

geometry_msgs::msg::Pose makePose(
    double x, double y, double z,
    double qx, double qy, double qz, double qw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.x = qx;
  pose.orientation.y = qy;
  pose.orientation.z = qz;
  pose.orientation.w = qw;
  return pose;
}

geometry_msgs::msg::Pose makePose(
    const Eigen::Vector3d& p,
    const Eigen::Quaterniond& q)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = p.x();
  pose.position.y = p.y();
  pose.position.z = p.z();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

Eigen::Quaterniond quatFromMsg(const geometry_msgs::msg::Quaternion& q_msg)
{
  Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
  q.normalize();
  return q;
}

Eigen::Quaterniond quatFromRpyDeg(const std::vector<double>& rpy_deg, const std::string& name)
{
  validateVectorSize(rpy_deg, 3, name);

  const double roll  = deg2rad(rpy_deg[0]);
  const double pitch = deg2rad(rpy_deg[1]);
  const double yaw   = deg2rad(rpy_deg[2]);

  Eigen::AngleAxisd rx(roll,  Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rz(yaw,   Eigen::Vector3d::UnitZ());

  // ROS-style roll-pitch-yaw composition
  Eigen::Quaterniond q = rz * ry * rx;
  q.normalize();
  return q;
}

geometry_msgs::msg::Pose makePoseFromXyzRpyDeg(
    const std::vector<double>& xyz,
    const std::vector<double>& rpy_deg,
    const std::string& xyz_name,
    const std::string& rpy_name)
{
  const Eigen::Vector3d p = vec3(xyz, xyz_name);
  const Eigen::Quaterniond q = quatFromRpyDeg(rpy_deg, rpy_name);
  return makePose(p, q);
}

Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  T.linear() = quatFromMsg(pose.orientation).toRotationMatrix();
  return T;
}

Eigen::Isometry3d xyzRpyDegToIsometry(
    const std::vector<double>& xyz,
    const std::vector<double>& rpy_deg,
    const std::string& xyz_name,
    const std::string& rpy_name)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = vec3(xyz, xyz_name);
  T.linear() = quatFromRpyDeg(rpy_deg, rpy_name).toRotationMatrix();
  return T;
}

void waitMs(int ms)
{
  rclcpp::sleep_for(std::chrono::milliseconds(ms));
}

// ============================================================================
// ======================== PARAM DECLARE / LOAD ===============================
// ============================================================================

ExecutorConfig declareAndLoadConfig(const rclcpp::Node::SharedPtr& node)
{
  ExecutorConfig cfg;

  // Core
  cfg.planning_group = node->declare_parameter<std::string>("planning_group", "fr3wml");
  cfg.home_named_target = node->declare_parameter<std::string>("home_named_target", "pos1");
  cfg.bottle_mesh_resource = node->declare_parameter<std::string>(
      "bottle_mesh_resource",
      "package://fr3wml_camera_gripper/meshes/bottle/bottle_50cL_reframed.stl");

  // Perception
  cfg.use_perception_pipeline = node->declare_parameter<bool>("use_perception_pipeline", false);
  cfg.capsule_pick_pose_topic = node->declare_parameter<std::string>("capsule_pick_pose_topic", "/capsule_pick_pose");
  cfg.perception_pose_timeout_sec = node->declare_parameter<double>("perception_pose_timeout_sec", 5.0);

  // Joint targets in degrees
  cfg.pre_pick_bottle_joint_deg = node->declare_parameter<std::vector<double>>(
      "pre_pick_bottle_joint_deg", {61.0, -83.0, 41.0, -49.0, -94.0, -74.0});

  cfg.pre_pick_capsule_joint_deg = node->declare_parameter<std::vector<double>>(
      "pre_pick_capsule_joint_deg", {113.0, -131.0, 116.0, -75.0, -87.0, -74.0});

  cfg.pre_place_bottle_joint_deg = node->declare_parameter<std::vector<double>>(
      "pre_place_bottle_joint_deg", {99.0, -83.0, 47.0, -56.0, -89.0, -74.0});

  // Tool pick-frame wrt wrist3_link
  cfg.softgripper_pick_frame_wrt_wrist_xyz = node->declare_parameter<std::vector<double>>(
      "softgripper_pick_frame_wrt_wrist_xyz", {0.0, 0.0, 0.19});
  cfg.softgripper_pick_frame_wrt_wrist_rpy_deg = node->declare_parameter<std::vector<double>>(
      "softgripper_pick_frame_wrt_wrist_rpy_deg", {0.0, 0.0, 0.0});

  cfg.suctioncup_pick_frame_wrt_wrist_xyz = node->declare_parameter<std::vector<double>>(
      "suctioncup_pick_frame_wrt_wrist_xyz", {0.0, 0.0, 0.19});
  cfg.suctioncup_pick_frame_wrt_wrist_rpy_deg = node->declare_parameter<std::vector<double>>(
      "suctioncup_pick_frame_wrt_wrist_rpy_deg", {0.0, 0.0, 0.0});

  // Object origin wrt tool pick frame
  cfg.bottle_origin_wrt_softgripper_pick_frame_xyz = node->declare_parameter<std::vector<double>>(
      "bottle_origin_wrt_softgripper_pick_frame_xyz", {0.0, 0.0, 0.248});
  cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg = node->declare_parameter<std::vector<double>>(
      "bottle_origin_wrt_softgripper_pick_frame_rpy_deg", {0.0, 0.0, 0.0});

  cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz = node->declare_parameter<std::vector<double>>(
      "capsule_origin_wrt_suctioncup_pick_frame_xyz", {0.0, 0.0, 0.005});
  cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg = node->declare_parameter<std::vector<double>>(
      "capsule_origin_wrt_suctioncup_pick_frame_rpy_deg", {0.0, 0.0, 0.0});

  // Task points in base_link — wrist3_link positions
  cfg.bottle_pick_position_xyz = node->declare_parameter<std::vector<double>>(
      "bottle_pick_position_xyz", {-0.160, -0.420, 0.410});

  cfg.bottle_place_position_xyz = node->declare_parameter<std::vector<double>>(
      "bottle_place_position_xyz", {0.20, -0.420, 0.410});

  cfg.capsule_pick_position_xyz = node->declare_parameter<std::vector<double>>(
      "capsule_pick_position_xyz", {0.230, -0.14, 0.195});
  cfg.capsule_pick_rpy_deg = node->declare_parameter<std::vector<double>>(
      "capsule_pick_rpy_deg", {0.0, 0.0, 0.0});

  cfg.capsule_place_position_xyz = node->declare_parameter<std::vector<double>>(
      "capsule_place_position_xyz", {0.20, -0.420, 0.444});
  cfg.capsule_place_rpy_deg = node->declare_parameter<std::vector<double>>(
      "capsule_place_rpy_deg", {0.0, 0.0, 0.0});

  // Capsule primitive dimensions
  cfg.capsule_radius = node->declare_parameter<double>("capsule_radius", 0.0205);
  cfg.capsule_height = node->declare_parameter<double>("capsule_height", 0.004);

  // PTP (MoveJ) tuning
  cfg.planning_time_sec = node->declare_parameter<double>("planning_time_sec", 4.0);
  cfg.num_planning_attempts = node->declare_parameter<int>("num_planning_attempts", 30);
  cfg.max_free_space_retries = node->declare_parameter<int>("max_free_space_retries", 15);
  cfg.ptp_velocity_scaling = node->declare_parameter<double>("ptp_velocity_scaling", 0.20);
  cfg.ptp_acceleration_scaling = node->declare_parameter<double>("ptp_acceleration_scaling", 0.20);
  cfg.goal_position_tolerance = node->declare_parameter<double>("goal_position_tolerance", 0.003);
  cfg.goal_orientation_tolerance = node->declare_parameter<double>("goal_orientation_tolerance", 0.03);
  cfg.joint_target_tolerance = node->declare_parameter<double>("joint_target_tolerance", 0.02);

  // LIN (MoveL) tuning — per-phase speeds and retreat distances
  cfg.bottle_pick_speed_percent = node->declare_parameter<double>("bottle_pick_speed_percent", 15.0);
  cfg.bottle_pick_retreat_speed_percent = node->declare_parameter<double>("bottle_pick_retreat_speed_percent", 15.0);
  cfg.bottle_pick_retreat_distance_m = node->declare_parameter<double>("bottle_pick_retreat_distance_m", 0.4);

  cfg.bottle_place_speed_percent = node->declare_parameter<double>("bottle_place_speed_percent", 15.0);
  cfg.bottle_place_retreat_speed_percent = node->declare_parameter<double>("bottle_place_retreat_speed_percent", 20.0);
  cfg.bottle_place_retreat_distance_m = node->declare_parameter<double>("bottle_place_retreat_distance_m", 0.1);

  cfg.capsule_pick_speed_percent = node->declare_parameter<double>("capsule_pick_speed_percent", 15.0);
  cfg.capsule_pick_retreat_speed_percent = node->declare_parameter<double>("capsule_pick_retreat_speed_percent", 20.0);
  cfg.capsule_pick_retreat_distance_m = node->declare_parameter<double>("capsule_pick_retreat_distance_m", 0.05);

  cfg.capsule_place_speed_percent = node->declare_parameter<double>("capsule_place_speed_percent", 15.0);
  cfg.capsule_place_retreat_speed_percent = node->declare_parameter<double>("capsule_place_retreat_speed_percent", 20.0);
  cfg.capsule_place_retreat_distance_m = node->declare_parameter<double>("capsule_place_retreat_distance_m", 0.05);

  // Wrist3_link to flange offset (from URDF: world_to_gripper joint z=0.098)
  cfg.wrist3_to_flange_z = node->declare_parameter<double>("wrist3_to_flange_z", 0.098);

  // Timing
  cfg.scene_update_wait_ms = node->declare_parameter<int>("scene_update_wait_ms", 800);
  cfg.retry_sleep_ms = node->declare_parameter<int>("retry_sleep_ms", 150);
  cfg.world_spawn_safety_z_m = node->declare_parameter<double>("world_spawn_safety_z_m", 0.001);

  // Validate sizes
  validateVectorSize(cfg.pre_pick_bottle_joint_deg, 6, "pre_pick_bottle_joint_deg");
  validateVectorSize(cfg.pre_pick_capsule_joint_deg, 6, "pre_pick_capsule_joint_deg");
  validateVectorSize(cfg.pre_place_bottle_joint_deg, 6, "pre_place_bottle_joint_deg");

  validateVectorSize(cfg.softgripper_pick_frame_wrt_wrist_xyz, 3, "softgripper_pick_frame_wrt_wrist_xyz");
  validateVectorSize(cfg.softgripper_pick_frame_wrt_wrist_rpy_deg, 3, "softgripper_pick_frame_wrt_wrist_rpy_deg");
  validateVectorSize(cfg.suctioncup_pick_frame_wrt_wrist_xyz, 3, "suctioncup_pick_frame_wrt_wrist_xyz");
  validateVectorSize(cfg.suctioncup_pick_frame_wrt_wrist_rpy_deg, 3, "suctioncup_pick_frame_wrt_wrist_rpy_deg");

  validateVectorSize(cfg.bottle_origin_wrt_softgripper_pick_frame_xyz, 3, "bottle_origin_wrt_softgripper_pick_frame_xyz");
  validateVectorSize(cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg, 3, "bottle_origin_wrt_softgripper_pick_frame_rpy_deg");
  validateVectorSize(cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz, 3, "capsule_origin_wrt_suctioncup_pick_frame_xyz");
  validateVectorSize(cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg, 3, "capsule_origin_wrt_suctioncup_pick_frame_rpy_deg");

  validateVectorSize(cfg.bottle_pick_position_xyz, 3, "bottle_pick_position_xyz");
  validateVectorSize(cfg.bottle_place_position_xyz, 3, "bottle_place_position_xyz");
  validateVectorSize(cfg.capsule_pick_position_xyz, 3, "capsule_pick_position_xyz");
  validateVectorSize(cfg.capsule_pick_rpy_deg, 3, "capsule_pick_rpy_deg");
  validateVectorSize(cfg.capsule_place_position_xyz, 3, "capsule_place_position_xyz");
  validateVectorSize(cfg.capsule_place_rpy_deg, 3, "capsule_place_rpy_deg");

  return cfg;
}

// ============================================================================
// ========================== MOVE GROUP HELPERS ===============================
// ============================================================================

void configureMoveGroup(MoveGroupInterface& move_group, const ExecutorConfig& cfg)
{
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectkConfigDefault");
  move_group.setPlanningTime(cfg.planning_time_sec);
  move_group.setNumPlanningAttempts(cfg.num_planning_attempts);
  move_group.setMaxVelocityScalingFactor(cfg.ptp_velocity_scaling);
  move_group.setMaxAccelerationScalingFactor(cfg.ptp_acceleration_scaling);
  move_group.setGoalPositionTolerance(cfg.goal_position_tolerance);
  move_group.setGoalOrientationTolerance(cfg.goal_orientation_tolerance);
  move_group.clearPathConstraints();
}

void logCurrentJoints(
    MoveGroupInterface& move_group,
    const rclcpp::Logger& logger,
    const std::string& label)
{
  const auto joints = move_group.getCurrentJointValues();

  if (joints.empty()) {
    RCLCPP_WARN(logger, "[%s] Current joint vector is empty.", label.c_str());
    return;
  }

  std::ostringstream oss;
  oss << "[" << label << "] Current joints [deg]:";

  for (std::size_t i = 0; i < joints.size(); ++i) {
    oss << " j" << (i + 1) << "=" << joints[i] * 180.0 / M_PI;
  }

  RCLCPP_INFO(logger, "%s", oss.str().c_str());
}

bool verifyCurrentJointsNearTarget(
    MoveGroupInterface& move_group,
    const std::vector<double>& target_joints_rad,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const std::string& label)
{
  const auto current_joints = move_group.getCurrentJointValues();
  if (current_joints.size() != target_joints_rad.size()) {
    RCLCPP_WARN(
        logger,
        "[%s] Cannot verify joint target. Current size=%zu target size=%zu",
        label.c_str(),
        current_joints.size(),
        target_joints_rad.size());
    return false;
  }

  double max_abs_err = 0.0;
  for (std::size_t i = 0; i < current_joints.size(); ++i) {
    max_abs_err = std::max(max_abs_err, std::abs(current_joints[i] - target_joints_rad[i]));
  }

  RCLCPP_INFO(
      logger,
      "[%s] Max joint absolute error = %.4f rad (%.2f deg)",
      label.c_str(),
      max_abs_err,
      max_abs_err * 180.0 / M_PI);

  return max_abs_err <= cfg.joint_target_tolerance;
}

geometry_msgs::msg::Pose computePoseFromJointTarget(
    MoveGroupInterface& move_group,
    const std::string& planning_group,
    const std::vector<double>& joint_values_rad)
{
  auto current_state = move_group.getCurrentState(5.0);
  if (!current_state) {
    throw std::runtime_error("Failed to get current robot state from MoveGroupInterface.");
  }

  const moveit::core::JointModelGroup* jmg =
      current_state->getJointModelGroup(planning_group);
  if (!jmg) {
    throw std::runtime_error("Failed to get JointModelGroup for " + planning_group);
  }

  moveit::core::RobotState target_state(*current_state);
  target_state.setJointGroupPositions(jmg, joint_values_rad);
  target_state.update();

  const Eigen::Isometry3d& tf =
      target_state.getGlobalLinkTransform(move_group.getEndEffectorLink());

  return makePose(Eigen::Vector3d(tf.translation()), Eigen::Quaterniond(tf.rotation()));
}

bool planAndExecuteWithRetries(
    MoveGroupInterface& move_group,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const std::string& description)
{
  for (int attempt = 1; attempt <= cfg.max_free_space_retries; ++attempt) {
    move_group.setStartStateToCurrentState();

    MoveGroupInterface::Plan plan;
    auto result = move_group.plan(plan);

    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(
          logger,
          "[%s] Planning succeeded on attempt %d/%d. Executing...",
          description.c_str(), attempt, cfg.max_free_space_retries);

      auto exec_result = move_group.execute(plan);
      if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(
            logger,
            "[%s] Execution succeeded on attempt %d/%d.",
            description.c_str(), attempt, cfg.max_free_space_retries);
        return true;
      }

      RCLCPP_WARN(
          logger,
          "[%s] Execution failed on attempt %d/%d. Retrying...",
          description.c_str(), attempt, cfg.max_free_space_retries);
    } else {
      RCLCPP_WARN(
          logger,
          "[%s] Planning failed on attempt %d/%d. Retrying...",
          description.c_str(), attempt, cfg.max_free_space_retries);
    }

    waitMs(cfg.retry_sleep_ms);
  }

  RCLCPP_ERROR(
      logger,
      "[%s] Exhausted all %d free-space planning attempts.",
      description.c_str(),
      cfg.max_free_space_retries);
  return false;
}

bool moveToNamedTargetRobust(
    MoveGroupInterface& move_group,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const std::string& named_target,
    const std::string& description)
{
  configureMoveGroup(move_group, cfg);
  move_group.clearPoseTargets();
  move_group.setNamedTarget(named_target);

  return planAndExecuteWithRetries(move_group, cfg, logger, description);
}

bool moveToExactJointTargetRobust(
    MoveGroupInterface& move_group,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const std::vector<double>& joint_target_rad,
    const std::string& description)
{
  configureMoveGroup(move_group, cfg);
  move_group.clearPoseTargets();
  move_group.setJointValueTarget(joint_target_rad);

  const bool ok = planAndExecuteWithRetries(move_group, cfg, logger, description);
  if (!ok) {
    return false;
  }

  logCurrentJoints(move_group, logger, description + " after execution");
  verifyCurrentJointsNearTarget(move_group, joint_target_rad, cfg, logger, description + " verification");
  return true;
}

// ============================================================================
// ============================= WORLD OBJECTS =================================
// ============================================================================

moveit_msgs::msg::CollisionObject makeBottleMeshObject(
    const std::string& frame_id,
    const std::string& object_id,
    const std::string& mesh_resource,
    const geometry_msgs::msg::Pose& pose)
{
  shapes::Mesh* mesh = shapes::createMeshFromResource(
      mesh_resource,
      Eigen::Vector3d(1.0, 1.0, 1.0));

  if (!mesh) {
    throw std::runtime_error("Failed to load bottle mesh resource: " + mesh_resource);
  }

  shapes::ShapeMsg mesh_msg;
  shapes::constructMsgFromShape(mesh, mesh_msg);
  shape_msgs::msg::Mesh mesh_msg_concrete = boost::get<shape_msgs::msg::Mesh>(mesh_msg);

  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = frame_id;
  obj.id = object_id;
  obj.meshes.push_back(mesh_msg_concrete);
  obj.mesh_poses.push_back(pose);
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;

  delete mesh;
  return obj;
}

moveit_msgs::msg::CollisionObject makeCapsulePrimitiveObject(
    const std::string& frame_id,
    const std::string& object_id,
    const geometry_msgs::msg::Pose& pose,
    double capsule_height,
    double capsule_radius)
{
  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = frame_id;
  obj.id = object_id;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  primitive.dimensions = {capsule_height, capsule_radius}; // [height, radius]

  obj.primitives.push_back(primitive);
  obj.primitive_poses.push_back(pose);
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;

  return obj;
}

void allowEndEffectorCollisions(
    rclcpp::Node::SharedPtr node,
    const rclcpp::Logger& logger)
{
  auto planning_scene_pub = node->create_publisher<moveit_msgs::msg::PlanningScene>(
      "planning_scene", 1);

  moveit_msgs::msg::PlanningScene ps_msg;
  ps_msg.is_diff = true;

  ps_msg.allowed_collision_matrix.default_entry_names = END_EFFECTOR_LINKS;
  ps_msg.allowed_collision_matrix.default_entry_values.assign(
      END_EFFECTOR_LINKS.size(), true);

  rclcpp::sleep_for(std::chrono::milliseconds(500));
  planning_scene_pub->publish(ps_msg);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  RCLCPP_INFO(logger,
      "ACM updated: end-effector links allowed to collide with everything.");
}

void removeObjectFromWorld(
    MoveGroupInterface& move_group,
    moveit::planning_interface::PlanningSceneInterface& planning_scene,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const std::string& object_id)
{
  planning_scene.removeCollisionObjects({object_id});
  waitMs(cfg.scene_update_wait_ms);

  auto known = planning_scene.getKnownObjectNames();
  if (std::find(known.begin(), known.end(), object_id) != known.end()) {
    RCLCPP_WARN(logger, "Object '%s' still in scene after first remove, retrying...", object_id.c_str());
    moveit_msgs::msg::CollisionObject remove_obj;
    remove_obj.header.frame_id = move_group.getPlanningFrame();
    remove_obj.id = object_id;
    remove_obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    planning_scene.applyCollisionObjects({remove_obj});
    waitMs(cfg.scene_update_wait_ms);

    known = planning_scene.getKnownObjectNames();
    if (std::find(known.begin(), known.end(), object_id) != known.end()) {
      RCLCPP_ERROR(logger, "Object '%s' STILL in scene after retry!", object_id.c_str());
    } else {
      RCLCPP_INFO(logger, "Object '%s' removed on retry.", object_id.c_str());
    }
  } else {
    RCLCPP_INFO(logger, "Object '%s' removed from world collision scene.", object_id.c_str());
  }
}

void addBottleToWorld(
    MoveGroupInterface& move_group,
    moveit::planning_interface::PlanningSceneInterface& planning_scene,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const geometry_msgs::msg::Pose& bottle_pose)
{
  const auto bottle_obj = makeBottleMeshObject(
    move_group.getPlanningFrame(),
    BOTTLE_OBJECT_ID,
    cfg.bottle_mesh_resource,
    bottle_pose);

  planning_scene.applyCollisionObjects({bottle_obj});
  waitMs(cfg.scene_update_wait_ms);

  RCLCPP_INFO(
      logger,
      "Bottle added to world at pose x=%.3f y=%.3f z=%.3f",
      bottle_pose.position.x,
      bottle_pose.position.y,
      bottle_pose.position.z);
}

void addCapsuleToWorld(
    MoveGroupInterface& move_group,
    moveit::planning_interface::PlanningSceneInterface& planning_scene,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const geometry_msgs::msg::Pose& capsule_pose)
{
  const auto capsule_obj = makeCapsulePrimitiveObject(
    move_group.getPlanningFrame(),
    CAPSULE_OBJECT_ID,
    capsule_pose,
    cfg.capsule_height,
    cfg.capsule_radius);

  planning_scene.applyCollisionObjects({capsule_obj});
  waitMs(cfg.scene_update_wait_ms);

  RCLCPP_INFO(
      logger,
      "Capsule added to world at pose x=%.3f y=%.3f z=%.3f",
      capsule_pose.position.x,
      capsule_pose.position.y,
      capsule_pose.position.z);
}

// ============================================================================
// ========================== GEOMETRY TRANSFORMS ==============================
// ============================================================================

geometry_msgs::msg::Pose computeObjectOriginPoseFromToolPickPose(
    const geometry_msgs::msg::Pose& pick_pose_in_base,
    const std::vector<double>& object_origin_wrt_pick_frame_xyz,
    const std::vector<double>& object_origin_wrt_pick_frame_rpy_deg)
{
  const Eigen::Isometry3d T_base_pick = poseMsgToIsometry(pick_pose_in_base);
  const Eigen::Isometry3d T_pick_object_origin = xyzRpyDegToIsometry(
      object_origin_wrt_pick_frame_xyz,
      object_origin_wrt_pick_frame_rpy_deg,
      "object_origin_wrt_pick_frame_xyz",
      "object_origin_wrt_pick_frame_rpy_deg");

  const Eigen::Isometry3d T_base_object = T_base_pick * T_pick_object_origin;
  return makePose(Eigen::Vector3d(T_base_object.translation()), Eigen::Quaterniond(T_base_object.rotation()));
}

// ============================================================================
// ========================= PERCEPTION RUNTIME HELPERS ========================
// ============================================================================

bool waitForRuntimeCapsulePickPose(
    const std::shared_ptr<PerceptionState>& perception_state,
    double timeout_sec,
    const std::string& planning_frame,
    const rclcpp::Logger& logger,
    geometry_msgs::msg::Pose& out_pose)
{
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(timeout_sec));

  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(perception_state->mutex);
      if (perception_state->has_capsule_pick_pose) {
        const auto& msg = perception_state->latest_capsule_pick_pose;

        if (!msg.header.frame_id.empty() && msg.header.frame_id != planning_frame) {
          RCLCPP_ERROR(
              logger,
              "Runtime capsule pick pose received in frame '%s', but planning frame is '%s'. "
              "Add TF conversion if you want to support different frames.",
              msg.header.frame_id.c_str(),
              planning_frame.c_str());
          return false;
        }

        out_pose = msg.pose;
        RCLCPP_INFO(
            logger,
            "Using runtime capsule pick pose: x=%.3f y=%.3f z=%.3f quat=(%.4f, %.4f, %.4f, %.4f)",
            out_pose.position.x,
            out_pose.position.y,
            out_pose.position.z,
            out_pose.orientation.x,
            out_pose.orientation.y,
            out_pose.orientation.z,
            out_pose.orientation.w);
        return true;
      }
    }

    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  RCLCPP_ERROR(logger, "Timed out waiting for runtime capsule pick pose.");
  return false;
}

geometry_msgs::msg::Pose getBlindCapsulePickPose(const ExecutorConfig& cfg)
{
  return makePoseFromXyzRpyDeg(
      cfg.capsule_pick_position_xyz,
      cfg.capsule_pick_rpy_deg,
      "capsule_pick_position_xyz",
      "capsule_pick_rpy_deg");
}

geometry_msgs::msg::Pose getCapsulePlacePose(const ExecutorConfig& cfg)
{
  return makePoseFromXyzRpyDeg(
      cfg.capsule_place_position_xyz,
      cfg.capsule_place_rpy_deg,
      "capsule_place_position_xyz",
      "capsule_place_rpy_deg");
}

// ============================================================================
// ====================== LIN (MoveL) — Fairino SDK ===========================
// ============================================================================
//
// Sends a linear Cartesian motion command to the Fairino controller via the
// bridge service /fairino/movel_pose.  The target is specified as a wrist3_link
// pose in base_link; the function converts it to a flange pose before sending.
//
// With MoveL_blendR=-1 in the controller, MoveL is BLOCKING: the SDK call
// returns only after the robot has reached the target position.

bool executeMoveL(
    MoveGroupInterface& move_group,
    MoveLClient::SharedPtr movel_client,
    const ExecutorConfig& cfg,
    const rclcpp::Logger& logger,
    const geometry_msgs::msg::Pose& wrist3_target_pose,
    double speed_percent,
    const std::string& description)
{
  // Convert wrist3_link position to flange position (controller operates in flange frame)
  // flange = wrist3_link_origin + R * (0, 0, wrist3_to_flange_z)
  const Eigen::Quaterniond q = quatFromMsg(wrist3_target_pose.orientation);
  const Eigen::Vector3d flange_offset = q * Eigen::Vector3d(0, 0, cfg.wrist3_to_flange_z);

  geometry_msgs::msg::Pose flange_target = wrist3_target_pose;
  flange_target.position.x += flange_offset.x();
  flange_target.position.y += flange_offset.y();
  flange_target.position.z += flange_offset.z();

  RCLCPP_INFO(
      logger,
      "[%s] MoveL wrist3 target: x=%.3f y=%.3f z=%.3f -> flange target: x=%.3f y=%.3f z=%.3f  speed=%.0f%%",
      description.c_str(),
      wrist3_target_pose.position.x,
      wrist3_target_pose.position.y,
      wrist3_target_pose.position.z,
      flange_target.position.x,
      flange_target.position.y,
      flange_target.position.z,
      speed_percent);

  auto req = std::make_shared<fairino_bridge::srv::ExecutePoseMotion::Request>();
  req->target_pose = flange_target;
  req->motion_type = "LINEAR";
  req->speed_percent = static_cast<float>(speed_percent);
  req->tool_id = 0;
  req->user_id = 0;
  req->load_frames_before_motion = false;

  if (!movel_client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(logger, "[%s] /fairino/movel_pose service not available.", description.c_str());
    return false;
  }

  auto future = movel_client->async_send_request(req);
  // MoveL_blendR=-1 makes the controller blocking, so we need a generous timeout
  const auto timeout = std::chrono::seconds(60);

  if (future.wait_for(timeout) != std::future_status::ready) {
    RCLCPP_ERROR(logger, "[%s] MoveL service call timed out.", description.c_str());
    return false;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(logger, "[%s] MoveL failed: %s", description.c_str(), response->message.c_str());
    return false;
  }

  RCLCPP_INFO(logger, "[%s] MoveL complete: %s", description.c_str(), response->message.c_str());

  // Log actual robot position after MoveL for verification
  const auto actual_pose = move_group.getCurrentPose().pose;
  RCLCPP_INFO(logger,
      "[%s] Actual wrist3_link after MoveL: x=%.3f y=%.3f z=%.3f (target was x=%.3f y=%.3f z=%.3f)",
      description.c_str(),
      actual_pose.position.x, actual_pose.position.y, actual_pose.position.z,
      wrist3_target_pose.position.x, wrist3_target_pose.position.y, wrist3_target_pose.position.z);

  return true;
}

// ============================================================================
// ============================= TASK SEQUENCES ================================
// ============================================================================

bool executeBottlePickPlace(
    MoveGroupInterface& move_group,
    moveit::planning_interface::PlanningSceneInterface& planning_scene,
    const ExecutorConfig& cfg,
    MoveLClient::SharedPtr movel_client,
    const rclcpp::Logger& logger)
{
  const auto pre_pick_bottle_rad  = degVectorToRad(cfg.pre_pick_bottle_joint_deg);
  const auto pre_place_bottle_rad = degVectorToRad(cfg.pre_place_bottle_joint_deg);

  // Remove bottle from scene so it doesn't block PTP or MoveL motions
  removeObjectFromWorld(move_group, planning_scene, cfg, logger, BOTTLE_OBJECT_ID);

  // 1) PTP to pre-pick bottle
  if (!moveToExactJointTargetRobust(
          move_group, cfg, logger, pre_pick_bottle_rad, "Bottle PTP to pre-pick")) {
    return false;
  }

  // Build bottle pick pose: position from params, orientation from actual robot pose at pre-pick
  geometry_msgs::msg::Pose bottle_pick_pose = move_group.getCurrentPose().pose;
  bottle_pick_pose.position.x = cfg.bottle_pick_position_xyz[0];
  bottle_pick_pose.position.y = cfg.bottle_pick_position_xyz[1];
  bottle_pick_pose.position.z = cfg.bottle_pick_position_xyz[2];

  // 2) LIN descend to bottle pick
  if (!executeMoveL(move_group, movel_client, cfg, logger, bottle_pick_pose,
          cfg.bottle_pick_speed_percent, "Bottle LIN descend to pick")) {
    return false;
  }

  // 3) Simulated pick (gripper close)

  // 4) LIN retreat: go up by retreat_distance_m from pick position
  geometry_msgs::msg::Pose bottle_pick_retreat_pose = bottle_pick_pose;
  bottle_pick_retreat_pose.position.z += cfg.bottle_pick_retreat_distance_m;
  RCLCPP_INFO(logger, "[Bottle LIN retreat from pick] retreat %.3f m -> target z=%.3f",
      cfg.bottle_pick_retreat_distance_m, bottle_pick_retreat_pose.position.z);
  if (!executeMoveL(move_group, movel_client, cfg, logger, bottle_pick_retreat_pose,
          cfg.bottle_pick_retreat_speed_percent, "Bottle LIN retreat from pick")) {
    return false;
  }

  // 5) PTP transfer to pre-place bottle joint config
  if (!moveToExactJointTargetRobust(
          move_group, cfg, logger, pre_place_bottle_rad, "Bottle PTP transfer to pre-place")) {
    return false;
  }

  // Build bottle place pose: position from params, orientation from actual robot pose at pre-place
  geometry_msgs::msg::Pose bottle_place_pose = move_group.getCurrentPose().pose;
  bottle_place_pose.position.x = cfg.bottle_place_position_xyz[0];
  bottle_place_pose.position.y = cfg.bottle_place_position_xyz[1];
  bottle_place_pose.position.z = cfg.bottle_place_position_xyz[2];

  // 6) LIN descend to bottle place
  if (!executeMoveL(move_group, movel_client, cfg, logger, bottle_place_pose,
          cfg.bottle_place_speed_percent, "Bottle LIN descend to place")) {
    return false;
  }

  // 7) Simulated release (gripper open)

  // 8) LIN retreat: go up by retreat_distance_m from place position
  geometry_msgs::msg::Pose bottle_place_retreat_pose = bottle_place_pose;
  bottle_place_retreat_pose.position.z += cfg.bottle_place_retreat_distance_m;
  RCLCPP_INFO(logger, "[Bottle LIN retreat from place] retreat %.3f m -> target z=%.3f",
      cfg.bottle_place_retreat_distance_m, bottle_place_retreat_pose.position.z);
  if (!executeMoveL(move_group, movel_client, cfg, logger, bottle_place_retreat_pose,
          cfg.bottle_place_retreat_speed_percent, "Bottle LIN retreat from place")) {
    return false;
  }

  // 9) Spawn bottle in final world pose (after retreat, to avoid collision)
  geometry_msgs::msg::Pose bottle_final_world_pose =
      computeObjectOriginPoseFromToolPickPose(
          bottle_place_pose,
          cfg.bottle_origin_wrt_softgripper_pick_frame_xyz,
          cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg);

  bottle_final_world_pose.position.z += cfg.world_spawn_safety_z_m;
  addBottleToWorld(move_group, planning_scene, cfg, logger, bottle_final_world_pose);

  return true;
}

bool executeCapsulePickPlace(
    MoveGroupInterface& move_group,
    moveit::planning_interface::PlanningSceneInterface& planning_scene,
    const ExecutorConfig& cfg,
    MoveLClient::SharedPtr movel_client,
    const std::shared_ptr<PerceptionState>& perception_state,
    const rclcpp::Logger& logger)
{
  const auto pre_pick_capsule_rad = degVectorToRad(cfg.pre_pick_capsule_joint_deg);
  const auto pre_place_bottle_rad = degVectorToRad(cfg.pre_place_bottle_joint_deg);

  // Pick pose of capsule in base_link
  geometry_msgs::msg::Pose capsule_pick_frame_pose;
  if (cfg.use_perception_pipeline) {
    if (!waitForRuntimeCapsulePickPose(
            perception_state,
            cfg.perception_pose_timeout_sec,
            move_group.getPlanningFrame(),
            logger,
            capsule_pick_frame_pose)) {
      return false;
    }
  } else {
    capsule_pick_frame_pose = getBlindCapsulePickPose(cfg);
    RCLCPP_INFO(
        logger,
        "Using blind capsule pick pose: x=%.3f y=%.3f z=%.3f",
        capsule_pick_frame_pose.position.x,
        capsule_pick_frame_pose.position.y,
        capsule_pick_frame_pose.position.z);
  }

  // Remove capsule from scene so it doesn't block PTP or MoveL motions
  removeObjectFromWorld(move_group, planning_scene, cfg, logger, CAPSULE_OBJECT_ID);

  // 1) PTP to validated capsule pre-pick joint configuration
  if (!moveToExactJointTargetRobust(
          move_group, cfg, logger, pre_pick_capsule_rad, "Capsule PTP to pre-pick")) {
    return false;
  }

  // Use actual robot orientation at pre-pick (tool pointing down)
  capsule_pick_frame_pose.orientation = move_group.getCurrentPose().pose.orientation;

  // 2) LIN descend to capsule pick
  if (!executeMoveL(move_group, movel_client, cfg, logger, capsule_pick_frame_pose,
          cfg.capsule_pick_speed_percent, "Capsule LIN descend to pick")) {
    return false;
  }

  // 3) Simulated capsule pick (suction on)

  // 4) LIN retreat: go up by retreat_distance_m from pick position
  geometry_msgs::msg::Pose capsule_pick_retreat_pose = capsule_pick_frame_pose;
  capsule_pick_retreat_pose.position.z += cfg.capsule_pick_retreat_distance_m;
  RCLCPP_INFO(logger, "[Capsule LIN retreat from pick] retreat %.3f m -> target z=%.3f",
      cfg.capsule_pick_retreat_distance_m, capsule_pick_retreat_pose.position.z);
  if (!executeMoveL(move_group, movel_client, cfg, logger, capsule_pick_retreat_pose,
          cfg.capsule_pick_retreat_speed_percent, "Capsule LIN retreat from pick")) {
    return false;
  }

  // 5) PTP to pre-place bottle pose (transfer)
  if (!moveToExactJointTargetRobust(
          move_group, cfg, logger, pre_place_bottle_rad, "Capsule PTP to pre-place")) {
    return false;
  }

  // Capsule place pose from params, with actual robot orientation at pre-place
  const geometry_msgs::msg::Pose capsule_place_frame_pose = getCapsulePlacePose(cfg);
  geometry_msgs::msg::Pose capsule_place_oriented = capsule_place_frame_pose;
  capsule_place_oriented.orientation = move_group.getCurrentPose().pose.orientation;

  // 6) LIN descend to capsule place
  if (!executeMoveL(move_group, movel_client, cfg, logger, capsule_place_oriented,
          cfg.capsule_place_speed_percent, "Capsule LIN descend to place")) {
    return false;
  }

  // 7) Simulated release (suction off)

  // 8) LIN retreat: go up by retreat_distance_m from place position
  geometry_msgs::msg::Pose capsule_place_retreat_pose = capsule_place_oriented;
  capsule_place_retreat_pose.position.z += cfg.capsule_place_retreat_distance_m;
  RCLCPP_INFO(logger, "[Capsule LIN retreat from place] retreat %.3f m -> target z=%.3f",
      cfg.capsule_place_retreat_distance_m, capsule_place_retreat_pose.position.z);
  if (!executeMoveL(move_group, movel_client, cfg, logger, capsule_place_retreat_pose,
          cfg.capsule_place_retreat_speed_percent, "Capsule LIN retreat from place")) {
    return false;
  }

  // 9) Spawn capsule in final world pose (after retreat)
  geometry_msgs::msg::Pose capsule_final_world_pose =
      computeObjectOriginPoseFromToolPickPose(
          capsule_place_oriented,
          cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz,
          cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg);

  capsule_final_world_pose.position.z += cfg.world_spawn_safety_z_m;
  addCapsuleToWorld(move_group, planning_scene, cfg, logger, capsule_final_world_pose);

  return true;
}

// ============================================================================
// ================================= MAIN =====================================
// ============================================================================

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("pick_place_bottle_capsule_framework");
  auto logger = node->get_logger();

  ExecutorConfig cfg;
  try {
    cfg = declareAndLoadConfig(node);
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Parameter loading failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  auto perception_state = std::make_shared<PerceptionState>();

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr capsule_pick_pose_sub;
  if (cfg.use_perception_pipeline) {
    capsule_pick_pose_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        cfg.capsule_pick_pose_topic,
        10,
        [perception_state, logger](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
          {
            std::lock_guard<std::mutex> lock(perception_state->mutex);
            perception_state->latest_capsule_pick_pose = *msg;
            perception_state->has_capsule_pick_pose = true;
          }

          RCLCPP_INFO(
              logger,
              "Received capsule pick pose: frame='%s' x=%.3f y=%.3f z=%.3f",
              msg->header.frame_id.c_str(),
              msg->pose.position.x,
              msg->pose.position.y,
              msg->pose.position.z);
        });
  }

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  MoveGroupInterface move_group(node, cfg.planning_group);

  RCLCPP_INFO(logger, "Planning group    : %s", cfg.planning_group.c_str());
  RCLCPP_INFO(logger, "Planning frame    : %s", move_group.getPlanningFrame().c_str());
  RCLCPP_INFO(logger, "End effector link : %s", move_group.getEndEffectorLink().c_str());
  RCLCPP_INFO(logger, "Mode              : %s", cfg.use_perception_pipeline ? "PERCEPTION" : "BLIND");
  RCLCPP_INFO(logger, "PTP velocity      : %.0f%%", cfg.ptp_velocity_scaling * 100.0);
  RCLCPP_INFO(logger, "Bottle pick  LIN  : descend=%.0f%% retreat=%.0f%% dist=%.3fm",
      cfg.bottle_pick_speed_percent, cfg.bottle_pick_retreat_speed_percent, cfg.bottle_pick_retreat_distance_m);
  RCLCPP_INFO(logger, "Bottle place LIN  : descend=%.0f%% retreat=%.0f%% dist=%.3fm",
      cfg.bottle_place_speed_percent, cfg.bottle_place_retreat_speed_percent, cfg.bottle_place_retreat_distance_m);
  RCLCPP_INFO(logger, "Capsule pick LIN  : descend=%.0f%% retreat=%.0f%% dist=%.3fm",
      cfg.capsule_pick_speed_percent, cfg.capsule_pick_retreat_speed_percent, cfg.capsule_pick_retreat_distance_m);
  RCLCPP_INFO(logger, "Capsule place LIN : descend=%.0f%% retreat=%.0f%% dist=%.3fm",
      cfg.capsule_place_speed_percent, cfg.capsule_place_retreat_speed_percent, cfg.capsule_place_retreat_distance_m);

  moveit::planning_interface::PlanningSceneInterface planning_scene;

  // Allow end-effector links to collide with everything
  allowEndEffectorCollisions(node, logger);

  // Service client for bridge-level linear Cartesian motion
  auto movel_client = node->create_client<fairino_bridge::srv::ExecutePoseMotion>(
      "/fairino/movel_pose");
  RCLCPP_INFO(logger, "Waiting for /fairino/movel_pose service...");
  if (!movel_client->wait_for_service(std::chrono::seconds(10))) {
    RCLCPP_ERROR(logger, "/fairino/movel_pose service not available. Is the bridge running?");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  RCLCPP_INFO(logger, "/fairino/movel_pose service connected.");

  try {
    // Clean scene: remove bottle and capsule added by scene_loader so they don't
    // interfere with pick motions.  They will be re-added after place.
    RCLCPP_INFO(logger, "=== Cleanup: removing bottle & capsule from planning scene ===");
    removeObjectFromWorld(move_group, planning_scene, cfg, logger, BOTTLE_OBJECT_ID);
    removeObjectFromWorld(move_group, planning_scene, cfg, logger, CAPSULE_OBJECT_ID);

    // 0) Move home
    RCLCPP_INFO(logger, "=== Global Step 0: Move home ===");
    if (!moveToNamedTargetRobust(move_group, cfg, logger, cfg.home_named_target, "Move home")) {
      throw std::runtime_error("Failed to move home.");
    }

    // 1) Bottle pick and place
    RCLCPP_INFO(logger, "=== Global Step 1: Bottle pick and place ===");
    if (!executeBottlePickPlace(move_group, planning_scene, cfg, movel_client, logger)) {
      throw std::runtime_error("Bottle pick and place failed.");
    }

    // 2) Capsule pick and place
    RCLCPP_INFO(logger, "=== Global Step 2: Capsule pick and place ===");
    if (!executeCapsulePickPlace(move_group, planning_scene, cfg, movel_client, perception_state, logger)) {
      throw std::runtime_error("Capsule pick and place failed.");
    }

    // 3) Remove scene objects before Return home — they can block MoveIt planning
    RCLCPP_INFO(logger, "=== Cleanup: removing objects before Return home ===");
    removeObjectFromWorld(move_group, planning_scene, cfg, logger, BOTTLE_OBJECT_ID);
    removeObjectFromWorld(move_group, planning_scene, cfg, logger, CAPSULE_OBJECT_ID);

    // 4) Return home
    RCLCPP_INFO(logger, "=== Global Step 3: Return home ===");
    if (!moveToNamedTargetRobust(move_group, cfg, logger, cfg.home_named_target, "Return home")) {
      throw std::runtime_error("Failed to return home.");
    }

    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Bottle + capsule framework COMPLETED.");
    RCLCPP_INFO(logger, "========================================");
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Task aborted: %s", e.what());
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
