// scene_handling.cpp
//
// Single owner of the MoveIt planning scene for the bottle+capsule task.
//
// Responsibilities:
//   1) Load the initial scene at startup: table, wall1, wall2, crate (mesh),
//      bottle (mesh), capsule (cylinder primitive).
//   2) Provide services to REMOVE bottle / capsule from the scene.
//   3) Provide topics to (re)ADD bottle / capsule at a runtime-given pose.
//
// The pick_place_framework node never touches the planning scene directly; it
// only calls these services / publishes on these topics. This keeps scene
// management isolated in one long-lived node.
//
// Services (std_srvs/Trigger):
//   /scene/remove_bottle
//   /scene/remove_capsule
//
// Topics (geometry_msgs/PoseStamped, frame must equal the loaded frame_id):
//   /scene/set_bottle_pose
//   /scene/set_capsule_pose

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>
#include <boost/variant/get.hpp>

using ApplySceneClient = rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>;

class SceneHandlingNode : public rclcpp::Node
{
public:
  SceneHandlingNode() : Node("scene_handling_node")
  {
    declareParameters();

    // Reentrant group for the /apply_planning_scene client so service
    // responses can be processed on a different thread while a subscription
    // or service handler is blocked in future.wait_for().
    client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    apply_scene_client_ = create_client<moveit_msgs::srv::ApplyPlanningScene>(
        "/apply_planning_scene",
        rmw_qos_profile_services_default,
        client_cb_group_);

    planning_scene_pub_ = create_publisher<moveit_msgs::msg::PlanningScene>(
        "planning_scene", 1);

    remove_bottle_srv_ = create_service<std_srvs::srv::Trigger>(
        "/scene/remove_bottle",
        std::bind(&SceneHandlingNode::handleRemoveBottle, this,
                  std::placeholders::_1, std::placeholders::_2));

    remove_capsule_srv_ = create_service<std_srvs::srv::Trigger>(
        "/scene/remove_capsule",
        std::bind(&SceneHandlingNode::handleRemoveCapsule, this,
                  std::placeholders::_1, std::placeholders::_2));

    set_bottle_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/scene/set_bottle_pose", 10,
        std::bind(&SceneHandlingNode::handleSetBottlePose, this, std::placeholders::_1));

    set_capsule_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/scene/set_capsule_pose", 10,
        std::bind(&SceneHandlingNode::handleSetCapsulePose, this, std::placeholders::_1));

    // Defer initial scene load slightly so move_group is ready.
    init_timer_ = create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SceneHandlingNode::loadInitialScene, this));

    // Periodically reassert the latest "forced" ADD for bottle/capsule on
    // /planning_scene so any source republishing a stale scene gets overridden.
    force_timer_ = create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SceneHandlingNode::reassertForcedState, this));
  }

private:
  // -------- parameters --------
  void declareParameters()
  {
    declare_parameter<std::string>("frame_id", "base_link");

    declare_parameter<std::vector<double>>("table.size", {1.2, 0.8, 0.02});
    declare_parameter<std::vector<double>>("table.position", {0.3, 0.0, -0.01});
    declare_parameter<std::vector<double>>("table.orientation", {0.0, 0.0, 0.0, 1.0});

    declare_parameter<std::vector<double>>("wall1.size", {0.02, 1.6, 1.2});
    declare_parameter<std::vector<double>>("wall1.position", {-0.5, 0.0, 0.6});
    declare_parameter<std::vector<double>>("wall1.orientation", {0.0, 0.0, 0.0, 1.0});

    declare_parameter<std::vector<double>>("wall2.size", {0.9, 0.02, 1.2});
    declare_parameter<std::vector<double>>("wall2.position", {0.0, 0.4, 0.6});
    declare_parameter<std::vector<double>>("wall2.orientation", {0.0, 0.0, 0.0, 1.0});

    declare_parameter<std::string>("crate.mesh_path", "");
    declare_parameter<std::vector<double>>("crate.position", {0.35, -0.10, 0.0});
    declare_parameter<std::vector<double>>("crate.orientation", {0.0, 0.0, 0.0, 1.0});
    declare_parameter<std::vector<double>>("crate.scale", {1.0, 1.0, 1.0});

    declare_parameter<std::string>("bottle.mesh_path", "");
    declare_parameter<std::vector<double>>("bottle.position", {0.32, -0.10, 0.0});
    declare_parameter<std::vector<double>>("bottle.orientation", {0.0, 0.0, 0.0, 1.0});
    declare_parameter<std::vector<double>>("bottle.scale", {1.0, 1.0, 1.0});

    declare_parameter<double>("capsule.radius", 0.0205);
    declare_parameter<double>("capsule.height", 0.004);
    declare_parameter<std::vector<double>>("capsule.position", {-0.2, 0.2, 0.0});
    declare_parameter<std::vector<double>>("capsule.orientation", {0.0, 0.0, 0.0, 1.0});

    declare_parameter<int>("scene_update_wait_ms", 500);

    // End-effector links that should be allowed to collide with everything
    declare_parameter<std::vector<std::string>>("end_effector_links",
        {"softgripper_link", "suctioncup_link", "gripper_body", "suction_cap"});
  }

  void validateVectorSize(const std::vector<double>& v, std::size_t n, const std::string& name)
  {
    if (v.size() != n) {
      throw std::runtime_error(
        "Parameter '" + name + "' must contain exactly " + std::to_string(n) + " values.");
    }
  }

  geometry_msgs::msg::Pose makePose(const std::vector<double>& p,
                                    const std::vector<double>& q,
                                    const std::string& pn,
                                    const std::string& qn)
  {
    validateVectorSize(p, 3, pn);
    validateVectorSize(q, 4, qn);
    geometry_msgs::msg::Pose pose;
    pose.position.x = p[0]; pose.position.y = p[1]; pose.position.z = p[2];
    pose.orientation.x = q[0]; pose.orientation.y = q[1];
    pose.orientation.z = q[2]; pose.orientation.w = q[3];
    return pose;
  }

  // -------- collision-object builders --------
  moveit_msgs::msg::CollisionObject makeBox(const std::string& id,
                                            const std::vector<double>& size,
                                            const std::vector<double>& pos,
                                            const std::vector<double>& quat)
  {
    validateVectorSize(size, 3, id + ".size");
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id_;
    obj.id = id;
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    prim.dimensions = {size[0], size[1], size[2]};
    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(makePose(pos, quat, id + ".position", id + ".orientation"));
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    return obj;
  }

  bool makeMesh(const std::string& id,
                const std::string& mesh_resource,
                const std::vector<double>& pos,
                const std::vector<double>& quat,
                const std::vector<double>& scale,
                moveit_msgs::msg::CollisionObject& out)
  {
    validateVectorSize(scale, 3, id + ".scale");
    shapes::Mesh* mesh = shapes::createMeshFromResource(
        mesh_resource, Eigen::Vector3d(scale[0], scale[1], scale[2]));
    if (!mesh) {
      RCLCPP_ERROR(get_logger(), "Failed to load mesh '%s' from %s",
                   id.c_str(), mesh_resource.c_str());
      return false;
    }
    shapes::ShapeMsg mesh_msg;
    shapes::constructMsgFromShape(mesh, mesh_msg);
    auto concrete = boost::get<shape_msgs::msg::Mesh>(mesh_msg);
    delete mesh;

    out.header.frame_id = frame_id_;
    out.id = id;
    out.meshes.push_back(concrete);
    out.mesh_poses.push_back(makePose(pos, quat, id + ".position", id + ".orientation"));
    out.operation = moveit_msgs::msg::CollisionObject::ADD;
    return true;
  }

  moveit_msgs::msg::CollisionObject makeCapsule(double radius, double height,
                                                const std::vector<double>& pos,
                                                const std::vector<double>& quat)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id_;
    obj.id = "capsule";
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    prim.dimensions = {height, radius};
    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(makePose(pos, quat, "capsule.position", "capsule.orientation"));
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    return obj;
  }

  // Publish the same diff on /planning_scene topic so RViz's Planning Scene
  // display (and any other planning_scene_monitor) gets the update even if
  // move_group's republication is missed / filtered.
  void publishDiffOnTopic(const std::vector<moveit_msgs::msg::CollisionObject>& objs)
  {
    moveit_msgs::msg::PlanningScene ps;
    ps.is_diff = true;
    ps.robot_state.is_diff = true;
    ps.world.collision_objects = objs;
    planning_scene_pub_->publish(ps);
  }

  // -------- direct service apply (single object diff) --------
  bool applyDiff(const moveit_msgs::msg::CollisionObject& obj, const std::string& label)
  {
    if (!apply_scene_client_->wait_for_service(std::chrono::seconds(3))) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene unavailable", label.c_str());
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    req->scene.is_diff = true;
    req->scene.robot_state.is_diff = true;
    req->scene.world.collision_objects.push_back(obj);

    auto fut = apply_scene_client_->async_send_request(req);
    if (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene timed out", label.c_str());
      return false;
    }
    auto res = fut.get();
    if (!res->success) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene returned FAILURE", label.c_str());
      return false;
    }
    // Also push the diff on the topic — covers RViz displays that subscribe
    // to /planning_scene rather than relying on move_group's republication.
    publishDiffOnTopic({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(scene_update_wait_ms_));
    RCLCPP_INFO(get_logger(), "[%s] OK", label.c_str());
    return true;
  }

  bool applyDiffMany(const std::vector<moveit_msgs::msg::CollisionObject>& objs,
                     const std::string& label)
  {
    if (!apply_scene_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene unavailable", label.c_str());
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    req->scene.is_diff = true;
    req->scene.robot_state.is_diff = true;
    req->scene.world.collision_objects = objs;

    auto fut = apply_scene_client_->async_send_request(req);
    if (fut.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene timed out", label.c_str());
      return false;
    }
    auto res = fut.get();
    if (!res->success) {
      RCLCPP_ERROR(get_logger(), "[%s] /apply_planning_scene returned FAILURE", label.c_str());
      return false;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(scene_update_wait_ms_));
    RCLCPP_INFO(get_logger(), "[%s] OK (%zu objects)", label.c_str(), objs.size());
    return true;
  }

  // -------- initial load --------
  void loadInitialScene()
  {
    if (initial_loaded_) return;
    initial_loaded_ = true;
    init_timer_->cancel();

    try {
      frame_id_              = get_parameter("frame_id").as_string();
      scene_update_wait_ms_  = get_parameter("scene_update_wait_ms").as_int();

      const auto table_size  = get_parameter("table.size").as_double_array();
      const auto table_pos   = get_parameter("table.position").as_double_array();
      const auto table_quat  = get_parameter("table.orientation").as_double_array();

      const auto wall1_size  = get_parameter("wall1.size").as_double_array();
      const auto wall1_pos   = get_parameter("wall1.position").as_double_array();
      const auto wall1_quat  = get_parameter("wall1.orientation").as_double_array();

      const auto wall2_size  = get_parameter("wall2.size").as_double_array();
      const auto wall2_pos   = get_parameter("wall2.position").as_double_array();
      const auto wall2_quat  = get_parameter("wall2.orientation").as_double_array();

      const auto crate_mesh  = get_parameter("crate.mesh_path").as_string();
      const auto crate_pos   = get_parameter("crate.position").as_double_array();
      const auto crate_quat  = get_parameter("crate.orientation").as_double_array();
      const auto crate_scale = get_parameter("crate.scale").as_double_array();

      bottle_mesh_path_      = get_parameter("bottle.mesh_path").as_string();
      bottle_scale_          = get_parameter("bottle.scale").as_double_array();
      const auto bottle_pos  = get_parameter("bottle.position").as_double_array();
      const auto bottle_quat = get_parameter("bottle.orientation").as_double_array();

      capsule_radius_        = get_parameter("capsule.radius").as_double();
      capsule_height_        = get_parameter("capsule.height").as_double();
      const auto cap_pos     = get_parameter("capsule.position").as_double_array();
      const auto cap_quat    = get_parameter("capsule.orientation").as_double_array();

      RCLCPP_INFO(get_logger(), "scene_handling: loading initial scene (frame_id=%s)",
                  frame_id_.c_str());

      // Wipe any previous instances first.
      moveit::planning_interface::PlanningSceneInterface psi;
      psi.removeCollisionObjects({"table", "wall1", "wall2", "crate", "bottle", "capsule"});
      rclcpp::sleep_for(std::chrono::milliseconds(scene_update_wait_ms_));

      std::vector<moveit_msgs::msg::CollisionObject> objs;
      objs.push_back(makeBox("table", table_size, table_pos, table_quat));
      objs.push_back(makeBox("wall1", wall1_size, wall1_pos, wall1_quat));
      objs.push_back(makeBox("wall2", wall2_size, wall2_pos, wall2_quat));

      if (!crate_mesh.empty()) {
        moveit_msgs::msg::CollisionObject crate;
        if (makeMesh("crate", crate_mesh, crate_pos, crate_quat, crate_scale, crate)) {
          objs.push_back(crate);
        }
      } else {
        RCLCPP_WARN(get_logger(), "crate.mesh_path empty; skipping crate.");
      }

      if (!bottle_mesh_path_.empty()) {
        moveit_msgs::msg::CollisionObject bottle;
        if (makeMesh("bottle", bottle_mesh_path_, bottle_pos, bottle_quat, bottle_scale_, bottle)) {
          objs.push_back(bottle);
        }
      } else {
        RCLCPP_WARN(get_logger(), "bottle.mesh_path empty; skipping bottle.");
      }

      objs.push_back(makeCapsule(capsule_radius_, capsule_height_, cap_pos, cap_quat));

      applyDiffMany(objs, "INITIAL load");

      // Update ACM so end-effector links can touch scene objects.
      const auto ee_links = get_parameter("end_effector_links").as_string_array();
      moveit_msgs::msg::PlanningScene ps_msg;
      ps_msg.is_diff = true;
      ps_msg.allowed_collision_matrix.default_entry_names = ee_links;
      ps_msg.allowed_collision_matrix.default_entry_values.assign(ee_links.size(), true);
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      planning_scene_pub_->publish(ps_msg);
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      RCLCPP_INFO(get_logger(),
                  "ACM updated: %zu end-effector link(s) allowed to collide with everything.",
                  ee_links.size());

      RCLCPP_INFO(get_logger(),
        "scene_handling: ready. Services: /scene/remove_bottle, /scene/remove_capsule. "
        "Topics: /scene/set_bottle_pose, /scene/set_capsule_pose");
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "scene_handling initial load failed: %s", e.what());
    }
  }

  // -------- service handlers --------
  void handleRemoveBottle(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    // Clear forced-state so the periodic timer stops re-asserting bottle.
    bottle_forced_valid_ = false;
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id_;
    obj.id = "bottle";
    obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    res->success = applyDiff(obj, "REMOVE bottle");
    res->message = res->success ? "bottle removed" : "remove failed";
  }

  void handleRemoveCapsule(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    capsule_forced_valid_ = false;
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id_;
    obj.id = "capsule";
    obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    res->success = applyDiff(obj, "REMOVE capsule");
    res->message = res->success ? "capsule removed" : "remove failed";
  }

  // -------- periodic force re-publish --------
  void reassertForcedState()
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    std::vector<moveit_msgs::msg::CollisionObject> objs;
    if (bottle_forced_valid_)  objs.push_back(bottle_forced_);
    if (capsule_forced_valid_) objs.push_back(capsule_forced_);
    if (objs.empty()) return;
    // Topic-only — cheap and idempotent; move_group's scene monitor applies
    // the diff on /planning_scene without blocking on service calls.
    publishDiffOnTopic(objs);
  }

  // -------- topic handlers (re-add at given pose) --------
  void handleSetBottlePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    if (bottle_mesh_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "set_bottle_pose: bottle.mesh_path is empty.");
      return;
    }
    if (!msg->header.frame_id.empty() && msg->header.frame_id != frame_id_) {
      RCLCPP_ERROR(get_logger(),
                   "set_bottle_pose: frame '%s' does not match scene frame '%s'.",
                   msg->header.frame_id.c_str(), frame_id_.c_str());
      return;
    }

    // ADD with existing id implicitly replaces in MoveIt; no pre-REMOVE needed.
    shapes::Mesh* mesh = shapes::createMeshFromResource(
        bottle_mesh_path_,
        Eigen::Vector3d(bottle_scale_[0], bottle_scale_[1], bottle_scale_[2]));
    if (!mesh) {
      RCLCPP_ERROR(get_logger(), "set_bottle_pose: failed to load mesh.");
      return;
    }
    shapes::ShapeMsg mesh_msg;
    shapes::constructMsgFromShape(mesh, mesh_msg);
    auto concrete = boost::get<shape_msgs::msg::Mesh>(mesh_msg);
    delete mesh;

    moveit_msgs::msg::CollisionObject add;
    add.header.frame_id = frame_id_;
    add.id = "bottle";
    add.meshes.push_back(concrete);
    add.mesh_poses.push_back(msg->pose);
    add.operation = moveit_msgs::msg::CollisionObject::ADD;

    RCLCPP_INFO(get_logger(),
        "set_bottle_pose -> ADD bottle at x=%.3f y=%.3f z=%.3f",
        msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    applyDiff(add, "ADD bottle");

    // Remember and republish periodically to defeat any source that pushes
    // back an outdated scene (e.g. stale planning_scene_monitor snapshots).
    bottle_forced_ = add;
    bottle_forced_valid_ = true;
  }

  void handleSetCapsulePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(scene_mutex_);
    if (!msg->header.frame_id.empty() && msg->header.frame_id != frame_id_) {
      RCLCPP_ERROR(get_logger(),
                   "set_capsule_pose: frame '%s' does not match scene frame '%s'.",
                   msg->header.frame_id.c_str(), frame_id_.c_str());
      return;
    }

    moveit_msgs::msg::CollisionObject add;
    add.header.frame_id = frame_id_;
    add.id = "capsule";
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    prim.dimensions = {capsule_height_, capsule_radius_};
    add.primitives.push_back(prim);
    add.primitive_poses.push_back(msg->pose);
    add.operation = moveit_msgs::msg::CollisionObject::ADD;

    RCLCPP_INFO(get_logger(),
        "set_capsule_pose -> ADD capsule at x=%.3f y=%.3f z=%.3f",
        msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    applyDiff(add, "ADD capsule");

    capsule_forced_ = add;
    capsule_forced_valid_ = true;
  }

  // -------- members --------
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;
  ApplySceneClient::SharedPtr apply_scene_client_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr remove_bottle_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr remove_capsule_srv_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr set_bottle_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr set_capsule_pose_sub_;

  std::mutex scene_mutex_;
  bool initial_loaded_ = false;

  std::string frame_id_ = "base_link";
  int scene_update_wait_ms_ = 500;

  std::string bottle_mesh_path_;
  std::vector<double> bottle_scale_ = {1.0, 1.0, 1.0};
  double capsule_radius_ = 0.0205;
  double capsule_height_ = 0.004;

  // Latest "forced" state — the last ADD we wanted to persist. The force-
  // republish timer uses these to keep RViz/move_group in sync against any
  // source that might publish a stale scene back.
  moveit_msgs::msg::CollisionObject bottle_forced_;
  moveit_msgs::msg::CollisionObject capsule_forced_;
  bool bottle_forced_valid_ = false;
  bool capsule_forced_valid_ = false;
  rclcpp::TimerBase::SharedPtr force_timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SceneHandlingNode>();

  // MultiThreadedExecutor is required: the reentrant callback group used by
  // the /apply_planning_scene client needs a second thread to dispatch the
  // service response while a subscription/service handler is blocked in
  // future.wait_for().
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}