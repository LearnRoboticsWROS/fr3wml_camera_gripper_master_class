// pick_place_framework.cpp
//
// Motion-only pick-and-place for bottle + capsule. This node does NOT touch
// the planning scene; all scene state (initial load, dynamic add/remove of
// bottle and capsule) is owned by the scene_handling node, which exposes:
//
//   /scene/remove_bottle    (std_srvs/Trigger)
//   /scene/remove_capsule   (std_srvs/Trigger)
//   /scene/set_bottle_pose  (geometry_msgs/PoseStamped)
//   /scene/set_capsule_pose (geometry_msgs/PoseStamped)
//
// Static collision objects (table, walls, crate) are loaded once by
// scene_handling and persist through the whole task — this node plans
// against them via the standard MoveIt planning_scene_monitor.

#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <sstream>

#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>

#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include "fairino_bridge/srv/execute_pose_motion.hpp"
#include <std_srvs/srv/trigger.hpp>

using MoveLClient    = rclcpp::Client<fairino_bridge::srv::ExecutePoseMotion>;
using TriggerClient  = rclcpp::Client<std_srvs::srv::Trigger>;
using moveit::planning_interface::MoveGroupInterface;

// ============================================================================
// =============================== CONFIG =====================================
// ============================================================================

struct ExecutorConfig
{
  std::string planning_group;
  std::string home_named_target;

  bool use_perception_pipeline;
  std::string capsule_pick_pose_topic;
  double perception_pose_timeout_sec;

  std::vector<double> pre_pick_bottle_joint_deg;
  std::vector<double> pre_pick_capsule_joint_deg;
  std::vector<double> pre_place_bottle_joint_deg;

  // Object origin wrt tool pick-frame (used to compute final spawn pose for
  // the visual scene update post-task).
  std::vector<double> bottle_origin_wrt_softgripper_pick_frame_xyz;
  std::vector<double> bottle_origin_wrt_softgripper_pick_frame_rpy_deg;
  std::vector<double> capsule_origin_wrt_suctioncup_pick_frame_xyz;
  std::vector<double> capsule_origin_wrt_suctioncup_pick_frame_rpy_deg;

  std::vector<double> bottle_pick_position_xyz;
  std::vector<double> bottle_place_position_xyz;
  std::vector<double> capsule_pick_position_xyz;
  std::vector<double> capsule_pick_rpy_deg;
  std::vector<double> capsule_place_position_xyz;
  std::vector<double> capsule_place_rpy_deg;

  double planning_time_sec;
  int    num_planning_attempts;
  int    max_free_space_retries;
  double ptp_velocity_scaling;
  double ptp_acceleration_scaling;
  double goal_position_tolerance;
  double goal_orientation_tolerance;
  double joint_target_tolerance;

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

  double wrist3_to_flange_z;

  int scene_update_wait_ms;
  int retry_sleep_ms;

  double world_spawn_safety_z_m;

  std::string scene_frame_id;
};

struct PerceptionState
{
  std::mutex mutex;
  bool has_capsule_pick_pose = false;
  geometry_msgs::msg::PoseStamped latest_capsule_pick_pose;
};

// ============================================================================
// ============================== HELPERS =====================================
// ============================================================================

void validateVectorSize(const std::vector<double>& v, std::size_t n, const std::string& name)
{
  if (v.size() != n) {
    throw std::runtime_error(
        "Parameter '" + name + "' must contain exactly " + std::to_string(n) + " values.");
  }
}

double deg2rad(double d) { return d * M_PI / 180.0; }

std::vector<double> degVectorToRad(const std::vector<double>& deg)
{
  std::vector<double> r;
  r.reserve(deg.size());
  for (double v : deg) r.push_back(deg2rad(v));
  return r;
}

Eigen::Vector3d vec3(const std::vector<double>& v, const std::string& name)
{
  validateVectorSize(v, 3, name);
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

geometry_msgs::msg::Pose makePose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = p.x(); pose.position.y = p.y(); pose.position.z = p.z();
  pose.orientation.x = q.x(); pose.orientation.y = q.y();
  pose.orientation.z = q.z(); pose.orientation.w = q.w();
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
  Eigen::AngleAxisd rx(deg2rad(rpy_deg[0]), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd ry(deg2rad(rpy_deg[1]), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rz(deg2rad(rpy_deg[2]), Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond q = rz * ry * rx;
  q.normalize();
  return q;
}

geometry_msgs::msg::Pose makePoseFromXyzRpyDeg(const std::vector<double>& xyz,
                                               const std::vector<double>& rpy_deg,
                                               const std::string& xn,
                                               const std::string& rn)
{
  return makePose(vec3(xyz, xn), quatFromRpyDeg(rpy_deg, rn));
}

Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::msg::Pose& p)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  T.linear() = quatFromMsg(p.orientation).toRotationMatrix();
  return T;
}

Eigen::Isometry3d xyzRpyDegToIsometry(const std::vector<double>& xyz,
                                      const std::vector<double>& rpy_deg,
                                      const std::string& xn,
                                      const std::string& rn)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = vec3(xyz, xn);
  T.linear() = quatFromRpyDeg(rpy_deg, rn).toRotationMatrix();
  return T;
}

void waitMs(int ms) { rclcpp::sleep_for(std::chrono::milliseconds(ms)); }

// ============================================================================
// =========================== PARAM LOADING ==================================
// ============================================================================

ExecutorConfig declareAndLoadConfig(const rclcpp::Node::SharedPtr& node)
{
  ExecutorConfig cfg;

  cfg.planning_group        = node->declare_parameter<std::string>("planning_group", "fr3wml");
  cfg.home_named_target     = node->declare_parameter<std::string>("home_named_target", "pos1");

  cfg.use_perception_pipeline    = node->declare_parameter<bool>("use_perception_pipeline", false);
  cfg.capsule_pick_pose_topic    = node->declare_parameter<std::string>("capsule_pick_pose_topic", "/capsule_pick_pose");
  cfg.perception_pose_timeout_sec = node->declare_parameter<double>("perception_pose_timeout_sec", 5.0);

  cfg.pre_pick_bottle_joint_deg  = node->declare_parameter<std::vector<double>>(
      "pre_pick_bottle_joint_deg",  {61.0, -83.0, 41.0, -49.0, -94.0, -74.0});
  cfg.pre_pick_capsule_joint_deg = node->declare_parameter<std::vector<double>>(
      "pre_pick_capsule_joint_deg", {113.0, -131.0, 116.0, -75.0, -87.0, -74.0});
  cfg.pre_place_bottle_joint_deg = node->declare_parameter<std::vector<double>>(
      "pre_place_bottle_joint_deg", {99.0, -83.0, 47.0, -56.0, -89.0, -74.0});

  cfg.bottle_origin_wrt_softgripper_pick_frame_xyz = node->declare_parameter<std::vector<double>>(
      "bottle_origin_wrt_softgripper_pick_frame_xyz", {0.0, 0.0, 0.248});
  cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg = node->declare_parameter<std::vector<double>>(
      "bottle_origin_wrt_softgripper_pick_frame_rpy_deg", {0.0, 0.0, 0.0});
  cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz = node->declare_parameter<std::vector<double>>(
      "capsule_origin_wrt_suctioncup_pick_frame_xyz", {0.0, 0.0, 0.005});
  cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg = node->declare_parameter<std::vector<double>>(
      "capsule_origin_wrt_suctioncup_pick_frame_rpy_deg", {0.0, 0.0, 0.0});

  cfg.bottle_pick_position_xyz   = node->declare_parameter<std::vector<double>>(
      "bottle_pick_position_xyz",  {-0.160, -0.420, 0.420});
  cfg.bottle_place_position_xyz  = node->declare_parameter<std::vector<double>>(
      "bottle_place_position_xyz", {0.20, -0.420, 0.420});
  cfg.capsule_pick_position_xyz  = node->declare_parameter<std::vector<double>>(
      "capsule_pick_position_xyz", {0.230, -0.14, 0.195});
  cfg.capsule_pick_rpy_deg       = node->declare_parameter<std::vector<double>>(
      "capsule_pick_rpy_deg",      {0.0, 0.0, 0.0});
  cfg.capsule_place_position_xyz = node->declare_parameter<std::vector<double>>(
      "capsule_place_position_xyz", {0.20, -0.420, 0.444});
  cfg.capsule_place_rpy_deg      = node->declare_parameter<std::vector<double>>(
      "capsule_place_rpy_deg",     {0.0, 0.0, 0.0});

  cfg.planning_time_sec          = node->declare_parameter<double>("planning_time_sec", 4.0);
  cfg.num_planning_attempts      = node->declare_parameter<int>("num_planning_attempts", 30);
  cfg.max_free_space_retries     = node->declare_parameter<int>("max_free_space_retries", 15);
  cfg.ptp_velocity_scaling       = node->declare_parameter<double>("ptp_velocity_scaling", 0.20);
  cfg.ptp_acceleration_scaling   = node->declare_parameter<double>("ptp_acceleration_scaling", 0.20);
  cfg.goal_position_tolerance    = node->declare_parameter<double>("goal_position_tolerance", 0.003);
  cfg.goal_orientation_tolerance = node->declare_parameter<double>("goal_orientation_tolerance", 0.03);
  cfg.joint_target_tolerance     = node->declare_parameter<double>("joint_target_tolerance", 0.02);

  cfg.bottle_pick_speed_percent          = node->declare_parameter<double>("bottle_pick_speed_percent", 15.0);
  cfg.bottle_pick_retreat_speed_percent  = node->declare_parameter<double>("bottle_pick_retreat_speed_percent", 15.0);
  cfg.bottle_pick_retreat_distance_m     = node->declare_parameter<double>("bottle_pick_retreat_distance_m", 0.4);
  cfg.bottle_place_speed_percent         = node->declare_parameter<double>("bottle_place_speed_percent", 15.0);
  cfg.bottle_place_retreat_speed_percent = node->declare_parameter<double>("bottle_place_retreat_speed_percent", 20.0);
  cfg.bottle_place_retreat_distance_m    = node->declare_parameter<double>("bottle_place_retreat_distance_m", 0.1);
  cfg.capsule_pick_speed_percent         = node->declare_parameter<double>("capsule_pick_speed_percent", 15.0);
  cfg.capsule_pick_retreat_speed_percent = node->declare_parameter<double>("capsule_pick_retreat_speed_percent", 20.0);
  cfg.capsule_pick_retreat_distance_m    = node->declare_parameter<double>("capsule_pick_retreat_distance_m", 0.05);
  cfg.capsule_place_speed_percent        = node->declare_parameter<double>("capsule_place_speed_percent", 15.0);
  cfg.capsule_place_retreat_speed_percent = node->declare_parameter<double>("capsule_place_retreat_speed_percent", 20.0);
  cfg.capsule_place_retreat_distance_m   = node->declare_parameter<double>("capsule_place_retreat_distance_m", 0.05);

  cfg.wrist3_to_flange_z   = node->declare_parameter<double>("wrist3_to_flange_z", 0.098);
  cfg.scene_update_wait_ms = node->declare_parameter<int>("scene_update_wait_ms", 800);
  cfg.retry_sleep_ms       = node->declare_parameter<int>("retry_sleep_ms", 150);
  cfg.world_spawn_safety_z_m = node->declare_parameter<double>("world_spawn_safety_z_m", 0.001);

  cfg.scene_frame_id       = node->declare_parameter<std::string>("scene_frame_id", "base_link");

  validateVectorSize(cfg.pre_pick_bottle_joint_deg,  6, "pre_pick_bottle_joint_deg");
  validateVectorSize(cfg.pre_pick_capsule_joint_deg, 6, "pre_pick_capsule_joint_deg");
  validateVectorSize(cfg.pre_place_bottle_joint_deg, 6, "pre_place_bottle_joint_deg");

  validateVectorSize(cfg.bottle_origin_wrt_softgripper_pick_frame_xyz,    3, "bottle_origin_wrt_softgripper_pick_frame_xyz");
  validateVectorSize(cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg, 3, "bottle_origin_wrt_softgripper_pick_frame_rpy_deg");
  validateVectorSize(cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz,    3, "capsule_origin_wrt_suctioncup_pick_frame_xyz");
  validateVectorSize(cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg, 3, "capsule_origin_wrt_suctioncup_pick_frame_rpy_deg");

  validateVectorSize(cfg.bottle_pick_position_xyz,   3, "bottle_pick_position_xyz");
  validateVectorSize(cfg.bottle_place_position_xyz,  3, "bottle_place_position_xyz");
  validateVectorSize(cfg.capsule_pick_position_xyz,  3, "capsule_pick_position_xyz");
  validateVectorSize(cfg.capsule_pick_rpy_deg,       3, "capsule_pick_rpy_deg");
  validateVectorSize(cfg.capsule_place_position_xyz, 3, "capsule_place_position_xyz");
  validateVectorSize(cfg.capsule_place_rpy_deg,      3, "capsule_place_rpy_deg");

  return cfg;
}

// ============================================================================
// ========================= MOVE GROUP HELPERS ===============================
// ============================================================================

void configureMoveGroup(MoveGroupInterface& mg, const ExecutorConfig& cfg)
{
  mg.setPlanningPipelineId("ompl");
  mg.setPlannerId("RRTConnectkConfigDefault");
  mg.setPlanningTime(cfg.planning_time_sec);
  mg.setNumPlanningAttempts(cfg.num_planning_attempts);
  mg.setMaxVelocityScalingFactor(cfg.ptp_velocity_scaling);
  mg.setMaxAccelerationScalingFactor(cfg.ptp_acceleration_scaling);
  mg.setGoalPositionTolerance(cfg.goal_position_tolerance);
  mg.setGoalOrientationTolerance(cfg.goal_orientation_tolerance);
  mg.clearPathConstraints();
}

bool planAndExecuteWithRetries(MoveGroupInterface& mg,
                               const ExecutorConfig& cfg,
                               const rclcpp::Logger& logger,
                               const std::string& description)
{
  for (int attempt = 1; attempt <= cfg.max_free_space_retries; ++attempt) {
    mg.setStartStateToCurrentState();
    MoveGroupInterface::Plan plan;
    auto result = mg.plan(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(logger, "[%s] Plan OK on %d/%d. Executing...",
                  description.c_str(), attempt, cfg.max_free_space_retries);
      auto exec = mg.execute(plan);
      if (exec == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(logger, "[%s] Execute OK on %d/%d.",
                    description.c_str(), attempt, cfg.max_free_space_retries);
        return true;
      }
      RCLCPP_WARN(logger, "[%s] Execute failed on %d/%d. Retrying.",
                  description.c_str(), attempt, cfg.max_free_space_retries);
    } else {
      RCLCPP_WARN(logger, "[%s] Plan failed on %d/%d. Retrying.",
                  description.c_str(), attempt, cfg.max_free_space_retries);
    }
    waitMs(cfg.retry_sleep_ms);
  }
  RCLCPP_ERROR(logger, "[%s] Exhausted %d planning attempts.",
               description.c_str(), cfg.max_free_space_retries);
  return false;
}

bool moveToNamedTargetRobust(MoveGroupInterface& mg, const ExecutorConfig& cfg,
                             const rclcpp::Logger& logger,
                             const std::string& named, const std::string& desc)
{
  configureMoveGroup(mg, cfg);
  mg.clearPoseTargets();
  mg.setNamedTarget(named);
  return planAndExecuteWithRetries(mg, cfg, logger, desc);
}

bool moveToExactJointTargetRobust(MoveGroupInterface& mg, const ExecutorConfig& cfg,
                                  const rclcpp::Logger& logger,
                                  const std::vector<double>& target_rad,
                                  const std::string& desc)
{
  configureMoveGroup(mg, cfg);
  mg.clearPoseTargets();
  mg.setJointValueTarget(target_rad);
  return planAndExecuteWithRetries(mg, cfg, logger, desc);
}

// ============================================================================
// ============================ MoveL (LIN) ===================================
// ============================================================================

bool executeMoveL(MoveGroupInterface& mg,
                  MoveLClient::SharedPtr client,
                  const ExecutorConfig& cfg,
                  const rclcpp::Logger& logger,
                  const geometry_msgs::msg::Pose& wrist3_target,
                  double speed_percent,
                  const std::string& desc)
{
  const Eigen::Quaterniond q = quatFromMsg(wrist3_target.orientation);
  const Eigen::Vector3d off = q * Eigen::Vector3d(0, 0, cfg.wrist3_to_flange_z);

  geometry_msgs::msg::Pose flange = wrist3_target;
  flange.position.x += off.x();
  flange.position.y += off.y();
  flange.position.z += off.z();

  RCLCPP_INFO(logger,
      "[%s] MoveL wrist3 (%.3f,%.3f,%.3f) -> flange (%.3f,%.3f,%.3f)  %.0f%%",
      desc.c_str(),
      wrist3_target.position.x, wrist3_target.position.y, wrist3_target.position.z,
      flange.position.x, flange.position.y, flange.position.z, speed_percent);

  auto req = std::make_shared<fairino_bridge::srv::ExecutePoseMotion::Request>();
  req->target_pose = flange;
  req->motion_type = "LINEAR";
  req->speed_percent = static_cast<float>(speed_percent);
  req->tool_id = 0;
  req->user_id = 0;
  req->load_frames_before_motion = false;

  if (!client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(logger, "[%s] /fairino/movel_pose unavailable.", desc.c_str());
    return false;
  }

  auto fut = client->async_send_request(req);
  if (fut.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
    RCLCPP_ERROR(logger, "[%s] MoveL timed out.", desc.c_str());
    return false;
  }

  auto res = fut.get();
  if (!res->success) {
    RCLCPP_ERROR(logger, "[%s] MoveL failed: %s", desc.c_str(), res->message.c_str());
    return false;
  }

  RCLCPP_INFO(logger, "[%s] MoveL complete: %s", desc.c_str(), res->message.c_str());
  const auto actual = mg.getCurrentPose().pose;
  RCLCPP_INFO(logger, "[%s] actual wrist3: (%.3f,%.3f,%.3f)",
      desc.c_str(), actual.position.x, actual.position.y, actual.position.z);
  return true;
}

// ============================================================================
// ============================ TRIGGER HELPER ================================
// ============================================================================

bool callTrigger(const TriggerClient::SharedPtr& client,
                 const rclcpp::Logger& logger,
                 const std::string& name)
{
  if (!client->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_ERROR(logger, "Service '%s' not available", name.c_str());
    return false;
  }
  auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto fut = client->async_send_request(req);
  if (fut.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    RCLCPP_ERROR(logger, "Service '%s' timed out", name.c_str());
    return false;
  }
  auto res = fut.get();
  if (res->success) {
    RCLCPP_INFO(logger, "Service %s -> OK (%s)", name.c_str(), res->message.c_str());
  } else {
    RCLCPP_ERROR(logger, "Service %s -> FAILED (%s)", name.c_str(), res->message.c_str());
  }
  return res->success;
}

// ============================================================================
// ========================= GEOMETRY (final spawn pose) ======================
// ============================================================================

geometry_msgs::msg::Pose computeObjectOriginPoseFromToolPickPose(
    const geometry_msgs::msg::Pose& pick_in_base,
    const std::vector<double>& origin_xyz,
    const std::vector<double>& origin_rpy_deg)
{
  const Eigen::Isometry3d T_base_pick = poseMsgToIsometry(pick_in_base);
  const Eigen::Isometry3d T_pick_obj = xyzRpyDegToIsometry(
      origin_xyz, origin_rpy_deg,
      "object_origin_xyz", "object_origin_rpy_deg");
  const Eigen::Isometry3d T = T_base_pick * T_pick_obj;
  return makePose(Eigen::Vector3d(T.translation()), Eigen::Quaterniond(T.rotation()));
}

// ============================================================================
// ============================ PERCEPTION HELPER =============================
// ============================================================================

bool waitForRuntimeCapsulePickPose(
    const std::shared_ptr<PerceptionState>& state,
    double timeout_sec,
    const std::string& planning_frame,
    const rclcpp::Logger& logger,
    geometry_msgs::msg::Pose& out)
{
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_sec));
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->has_capsule_pick_pose) {
        const auto& msg = state->latest_capsule_pick_pose;
        if (!msg.header.frame_id.empty() && msg.header.frame_id != planning_frame) {
          RCLCPP_ERROR(logger,
              "Runtime capsule pick pose frame '%s' != planning frame '%s'.",
              msg.header.frame_id.c_str(), planning_frame.c_str());
          return false;
        }
        out = msg.pose;
        return true;
      }
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
  RCLCPP_ERROR(logger, "Timed out waiting for runtime capsule pick pose.");
  return false;
}

// ============================================================================
// ============================ TASK SEQUENCES ================================
// ============================================================================

bool executeBottlePickPlace(
    MoveGroupInterface& mg,
    const ExecutorConfig& cfg,
    MoveLClient::SharedPtr movel_client,
    const TriggerClient::SharedPtr& gripper_close_client,
    const TriggerClient::SharedPtr& gripper_idle_client,
    const TriggerClient::SharedPtr& scene_remove_bottle_client,
    geometry_msgs::msg::Pose& out_bottle_final_world_pose,
    const rclcpp::Logger& logger)
{
  const auto pre_pick  = degVectorToRad(cfg.pre_pick_bottle_joint_deg);
  const auto pre_place = degVectorToRad(cfg.pre_place_bottle_joint_deg);

  // 1) PTP to pre-pick (bottle still in scene)
  if (!moveToExactJointTargetRobust(mg, cfg, logger, pre_pick, "Bottle PTP to pre-pick"))
    return false;

  // Remove bottle via scene_handling (LIN bypasses MoveIt collisions anyway)
  if (!callTrigger(scene_remove_bottle_client, logger, "/scene/remove_bottle")) {
    RCLCPP_WARN(logger, "remove_bottle failed — continuing.");
  }
  waitMs(cfg.scene_update_wait_ms);

  geometry_msgs::msg::Pose bottle_pick = mg.getCurrentPose().pose;
  bottle_pick.position.x = cfg.bottle_pick_position_xyz[0];
  bottle_pick.position.y = cfg.bottle_pick_position_xyz[1];
  bottle_pick.position.z = cfg.bottle_pick_position_xyz[2];

  if (!executeMoveL(mg, movel_client, cfg, logger, bottle_pick,
                    cfg.bottle_pick_speed_percent, "Bottle LIN descend to pick"))
    return false;

  if (!callTrigger(gripper_close_client, logger, "gripper/close"))
    RCLCPP_WARN(logger, "gripper close failed — continuing.");
  waitMs(500);

  geometry_msgs::msg::Pose bottle_pick_retreat = bottle_pick;
  bottle_pick_retreat.position.z += cfg.bottle_pick_retreat_distance_m;
  if (!executeMoveL(mg, movel_client, cfg, logger, bottle_pick_retreat,
                    cfg.bottle_pick_retreat_speed_percent, "Bottle LIN retreat from pick"))
    return false;

  if (!moveToExactJointTargetRobust(mg, cfg, logger, pre_place, "Bottle PTP transfer to pre-place"))
    return false;

  geometry_msgs::msg::Pose bottle_place = mg.getCurrentPose().pose;
  bottle_place.position.x = cfg.bottle_place_position_xyz[0];
  bottle_place.position.y = cfg.bottle_place_position_xyz[1];
  bottle_place.position.z = cfg.bottle_place_position_xyz[2];

  if (!executeMoveL(mg, movel_client, cfg, logger, bottle_place,
                    cfg.bottle_place_speed_percent, "Bottle LIN descend to place"))
    return false;

  if (!callTrigger(gripper_idle_client, logger, "gripper/idle"))
    RCLCPP_WARN(logger, "gripper idle failed — continuing.");
  waitMs(500);

  geometry_msgs::msg::Pose bottle_place_retreat = bottle_place;
  bottle_place_retreat.position.z += cfg.bottle_place_retreat_distance_m;
  if (!executeMoveL(mg, movel_client, cfg, logger, bottle_place_retreat,
                    cfg.bottle_place_retreat_speed_percent, "Bottle LIN retreat from place"))
    return false;

  // Final upright bottle pose at place — handed back for post-home spawn
  geometry_msgs::msg::Pose final_pose = computeObjectOriginPoseFromToolPickPose(
      bottle_place,
      cfg.bottle_origin_wrt_softgripper_pick_frame_xyz,
      cfg.bottle_origin_wrt_softgripper_pick_frame_rpy_deg);
  final_pose.position.z += cfg.world_spawn_safety_z_m;
  final_pose.orientation.x = 0.0;
  final_pose.orientation.y = 0.0;
  final_pose.orientation.z = 0.0;
  final_pose.orientation.w = 1.0;
  out_bottle_final_world_pose = final_pose;

  return true;
}

bool executeCapsulePickPlace(
    MoveGroupInterface& mg,
    const ExecutorConfig& cfg,
    MoveLClient::SharedPtr movel_client,
    const TriggerClient::SharedPtr& suction_on_client,
    const TriggerClient::SharedPtr& suction_off_client,
    const TriggerClient::SharedPtr& scene_remove_capsule_client,
    const std::shared_ptr<PerceptionState>& perception_state,
    geometry_msgs::msg::Pose& out_capsule_final_world_pose,
    const rclcpp::Logger& logger)
{
  const auto pre_pick  = degVectorToRad(cfg.pre_pick_capsule_joint_deg);
  const auto pre_place = degVectorToRad(cfg.pre_place_bottle_joint_deg);

  geometry_msgs::msg::Pose capsule_pick;
  if (cfg.use_perception_pipeline) {
    if (!waitForRuntimeCapsulePickPose(perception_state, cfg.perception_pose_timeout_sec,
                                       mg.getPlanningFrame(), logger, capsule_pick))
      return false;
  } else {
    capsule_pick = makePoseFromXyzRpyDeg(
        cfg.capsule_pick_position_xyz, cfg.capsule_pick_rpy_deg,
        "capsule_pick_position_xyz", "capsule_pick_rpy_deg");
  }

  if (!moveToExactJointTargetRobust(mg, cfg, logger, pre_pick, "Capsule PTP to pre-pick"))
    return false;

  if (!callTrigger(scene_remove_capsule_client, logger, "/scene/remove_capsule"))
    RCLCPP_WARN(logger, "remove_capsule failed — continuing.");
  waitMs(cfg.scene_update_wait_ms);

  capsule_pick.orientation = mg.getCurrentPose().pose.orientation;

  if (!executeMoveL(mg, movel_client, cfg, logger, capsule_pick,
                    cfg.capsule_pick_speed_percent, "Capsule LIN descend to pick"))
    return false;

  if (!callTrigger(suction_on_client, logger, "suction/on"))
    RCLCPP_WARN(logger, "suction on failed — continuing.");
  waitMs(500);

  geometry_msgs::msg::Pose capsule_pick_retreat = capsule_pick;
  capsule_pick_retreat.position.z += cfg.capsule_pick_retreat_distance_m;
  if (!executeMoveL(mg, movel_client, cfg, logger, capsule_pick_retreat,
                    cfg.capsule_pick_retreat_speed_percent, "Capsule LIN retreat from pick"))
    return false;

  if (!moveToExactJointTargetRobust(mg, cfg, logger, pre_place, "Capsule PTP to pre-place"))
    return false;

  geometry_msgs::msg::Pose capsule_place = makePoseFromXyzRpyDeg(
      cfg.capsule_place_position_xyz, cfg.capsule_place_rpy_deg,
      "capsule_place_position_xyz", "capsule_place_rpy_deg");
  capsule_place.orientation = mg.getCurrentPose().pose.orientation;

  if (!executeMoveL(mg, movel_client, cfg, logger, capsule_place,
                    cfg.capsule_place_speed_percent, "Capsule LIN descend to place"))
    return false;

  if (!callTrigger(suction_off_client, logger, "suction/off"))
    RCLCPP_WARN(logger, "suction off failed — continuing.");
  waitMs(500);

  geometry_msgs::msg::Pose capsule_place_retreat = capsule_place;
  capsule_place_retreat.position.z += cfg.capsule_place_retreat_distance_m;
  if (!executeMoveL(mg, movel_client, cfg, logger, capsule_place_retreat,
                    cfg.capsule_place_retreat_speed_percent, "Capsule LIN retreat from place"))
    return false;

  geometry_msgs::msg::Pose final_pose = computeObjectOriginPoseFromToolPickPose(
      capsule_place,
      cfg.capsule_origin_wrt_suctioncup_pick_frame_xyz,
      cfg.capsule_origin_wrt_suctioncup_pick_frame_rpy_deg);
  final_pose.position.z += cfg.world_spawn_safety_z_m;
  out_capsule_final_world_pose = final_pose;

  return true;
}

// ============================================================================
// ================================= MAIN =====================================
// ============================================================================

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("pick_place_framework");
  auto logger = node->get_logger();

  ExecutorConfig cfg;
  try {
    cfg = declareAndLoadConfig(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Param load failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  auto perception_state = std::make_shared<PerceptionState>();

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr capsule_pick_pose_sub;
  if (cfg.use_perception_pipeline) {
    capsule_pick_pose_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        cfg.capsule_pick_pose_topic, 10,
        [perception_state, logger](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(perception_state->mutex);
          perception_state->latest_capsule_pick_pose = *msg;
          perception_state->has_capsule_pick_pose = true;
          RCLCPP_INFO(logger, "Capsule pick pose: frame='%s' (%.3f,%.3f,%.3f)",
                      msg->header.frame_id.c_str(),
                      msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        });
  }

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  MoveGroupInterface move_group(node, cfg.planning_group);

  RCLCPP_INFO(logger, "Planning group   : %s", cfg.planning_group.c_str());
  RCLCPP_INFO(logger, "Planning frame   : %s", move_group.getPlanningFrame().c_str());
  RCLCPP_INFO(logger, "End effector link: %s", move_group.getEndEffectorLink().c_str());
  RCLCPP_INFO(logger, "Mode             : %s", cfg.use_perception_pipeline ? "PERCEPTION" : "BLIND");

  // Bridge MoveL service
  auto movel_client = node->create_client<fairino_bridge::srv::ExecutePoseMotion>(
      "/fairino/movel_pose");
  RCLCPP_INFO(logger, "Waiting for /fairino/movel_pose...");
  if (!movel_client->wait_for_service(std::chrono::seconds(10))) {
    RCLCPP_ERROR(logger, "/fairino/movel_pose unavailable. Bridge running?");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  // Gripper service clients
  auto gripper_close_client = node->create_client<std_srvs::srv::Trigger>("gripper/close");
  auto gripper_idle_client  = node->create_client<std_srvs::srv::Trigger>("gripper/idle");
  auto suction_on_client    = node->create_client<std_srvs::srv::Trigger>("suction/on");
  auto suction_off_client   = node->create_client<std_srvs::srv::Trigger>("suction/off");

  // Scene-handling clients (REMOVE) and publishers (set pose / re-add)
  auto scene_remove_bottle_client  = node->create_client<std_srvs::srv::Trigger>("/scene/remove_bottle");
  auto scene_remove_capsule_client = node->create_client<std_srvs::srv::Trigger>("/scene/remove_capsule");

  RCLCPP_INFO(logger, "Waiting for /scene/remove_bottle...");
  if (!scene_remove_bottle_client->wait_for_service(std::chrono::seconds(15))) {
    RCLCPP_ERROR(logger, "/scene/remove_bottle unavailable. Is scene_handling_node running?");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  RCLCPP_INFO(logger, "Waiting for /scene/remove_capsule...");
  if (!scene_remove_capsule_client->wait_for_service(std::chrono::seconds(15))) {
    RCLCPP_ERROR(logger, "/scene/remove_capsule unavailable. Is scene_handling_node running?");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  auto set_bottle_pose_pub  = node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/scene/set_bottle_pose", 10);
  auto set_capsule_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/scene/set_capsule_pose", 10);

  try {
    // 0) Move home
    RCLCPP_INFO(logger, "=== Step 0: Move home ===");
    if (!moveToNamedTargetRobust(move_group, cfg, logger, cfg.home_named_target, "Move home"))
      throw std::runtime_error("Failed to move home.");

    geometry_msgs::msg::Pose bottle_final_world_pose;
    geometry_msgs::msg::Pose capsule_final_world_pose;

    // 1) Bottle
    RCLCPP_INFO(logger, "=== Step 1: Bottle pick and place ===");
    if (!executeBottlePickPlace(move_group, cfg, movel_client,
                                gripper_close_client, gripper_idle_client,
                                scene_remove_bottle_client,
                                bottle_final_world_pose, logger))
      throw std::runtime_error("Bottle pick and place failed.");

    // 2) Capsule
    RCLCPP_INFO(logger, "=== Step 2: Capsule pick and place ===");
    if (!executeCapsulePickPlace(move_group, cfg, movel_client,
                                 suction_on_client, suction_off_client,
                                 scene_remove_capsule_client,
                                 perception_state,
                                 capsule_final_world_pose, logger))
      throw std::runtime_error("Capsule pick and place failed.");

    // 3) Return home (scene has neither bottle nor capsule — nothing blocks PTP)
    RCLCPP_INFO(logger, "=== Step 3: Return home ===");
    if (!moveToNamedTargetRobust(move_group, cfg, logger, cfg.home_named_target, "Return home"))
      throw std::runtime_error("Failed to return home.");

    // 4) Tell scene_handling to re-add bottle and capsule at place positions
    RCLCPP_INFO(logger, "=== Step 4: publish final scene poses to scene_handling ===");

    geometry_msgs::msg::PoseStamped bottle_msg;
    bottle_msg.header.frame_id = cfg.scene_frame_id;
    bottle_msg.header.stamp = node->now();
    bottle_msg.pose = bottle_final_world_pose;
    set_bottle_pose_pub->publish(bottle_msg);
    RCLCPP_INFO(logger, "Published /scene/set_bottle_pose at (%.3f,%.3f,%.3f)",
                bottle_msg.pose.position.x, bottle_msg.pose.position.y, bottle_msg.pose.position.z);

    waitMs(cfg.scene_update_wait_ms);

    geometry_msgs::msg::PoseStamped capsule_msg;
    capsule_msg.header.frame_id = cfg.scene_frame_id;
    capsule_msg.header.stamp = node->now();
    capsule_msg.pose = capsule_final_world_pose;
    set_capsule_pose_pub->publish(capsule_msg);
    RCLCPP_INFO(logger, "Published /scene/set_capsule_pose at (%.3f,%.3f,%.3f)",
                capsule_msg.pose.position.x, capsule_msg.pose.position.y, capsule_msg.pose.position.z);

    waitMs(cfg.scene_update_wait_ms);

    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "pick_place_framework COMPLETED.");
    RCLCPP_INFO(logger, "Node staying alive — Ctrl+C to exit.");
    RCLCPP_INFO(logger, "========================================");
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Task aborted: %s", e.what());
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  while (rclcpp::ok()) {
    rclcpp::sleep_for(std::chrono::milliseconds(200));
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}