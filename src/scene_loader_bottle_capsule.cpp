#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>
#include <boost/variant/get.hpp>

class SceneLoaderBottleCapsuleNode : public rclcpp::Node
{
public:
  SceneLoaderBottleCapsuleNode() : Node("scene_loader_node")
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

    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&SceneLoaderBottleCapsuleNode::loadScene, this));
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  bool loaded_ = false;

  void validateVectorSize(
      const std::vector<double>& vec,
      std::size_t expected_size,
      const std::string& param_name)
  {
    if (vec.size() != expected_size) {
      throw std::runtime_error(
        "Parameter '" + param_name + "' must contain exactly " +
        std::to_string(expected_size) + " values.");
    }
  }

  geometry_msgs::msg::Pose makePose(
      const std::vector<double>& pos,
      const std::vector<double>& quat,
      const std::string& pos_name,
      const std::string& quat_name)
  {
    validateVectorSize(pos, 3, pos_name);
    validateVectorSize(quat, 4, quat_name);

    geometry_msgs::msg::Pose pose;
    pose.position.x = pos.at(0);
    pose.position.y = pos.at(1);
    pose.position.z = pos.at(2);
    pose.orientation.x = quat.at(0);
    pose.orientation.y = quat.at(1);
    pose.orientation.z = quat.at(2);
    pose.orientation.w = quat.at(3);
    return pose;
  }

  moveit_msgs::msg::CollisionObject makeBoxObject(
      const std::string& object_id,
      const std::string& frame_id,
      const std::vector<double>& size,
      const std::vector<double>& pos,
      const std::vector<double>& quat)
  {
    validateVectorSize(size, 3, object_id + ".size");
    validateVectorSize(pos, 3, object_id + ".position");
    validateVectorSize(quat, 4, object_id + ".orientation");

    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id;
    obj.id = object_id;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {size.at(0), size.at(1), size.at(2)};

    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(
      makePose(pos, quat, object_id + ".position", object_id + ".orientation"));
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;

    return obj;
  }

  bool loadMeshObject(
      const std::string& object_id,
      const std::string& frame_id,
      const std::string& mesh_resource,
      const std::vector<double>& pos,
      const std::vector<double>& quat,
      const std::vector<double>& scale,
      moveit_msgs::msg::CollisionObject& obj)
  {
    validateVectorSize(pos, 3, object_id + ".position");
    validateVectorSize(quat, 4, object_id + ".orientation");
    validateVectorSize(scale, 3, object_id + ".scale");

    RCLCPP_INFO(
      get_logger(),
      "Loading mesh object '%s' from resource: %s",
      object_id.c_str(),
      mesh_resource.c_str());

    shapes::Mesh* mesh = shapes::createMeshFromResource(
      mesh_resource,
      Eigen::Vector3d(scale.at(0), scale.at(1), scale.at(2)));

    if (!mesh) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to load mesh resource for '%s': %s",
        object_id.c_str(),
        mesh_resource.c_str());
      return false;
    }

    shapes::ShapeMsg mesh_msg;
    shapes::constructMsgFromShape(mesh, mesh_msg);
    shape_msgs::msg::Mesh mesh_msg_concrete = boost::get<shape_msgs::msg::Mesh>(mesh_msg);

    obj.header.frame_id = frame_id;
    obj.id = object_id;
    obj.meshes.push_back(mesh_msg_concrete);
    obj.mesh_poses.push_back(
      makePose(pos, quat, object_id + ".position", object_id + ".orientation"));
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;

    delete mesh;

    RCLCPP_INFO(
      get_logger(),
      "Loaded mesh object '%s' successfully.",
      object_id.c_str());

    return true;
  }

  moveit_msgs::msg::CollisionObject makeCapsuleObject(
      const std::string& frame_id,
      double radius,
      double height,
      const std::vector<double>& pos,
      const std::vector<double>& quat)
  {
    validateVectorSize(pos, 3, "capsule.position");
    validateVectorSize(quat, 4, "capsule.orientation");

    moveit_msgs::msg::CollisionObject capsule;
    capsule.header.frame_id = frame_id;
    capsule.id = "capsule";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    primitive.dimensions = {height, radius};  // [height, radius]

    capsule.primitives.push_back(primitive);
    capsule.primitive_poses.push_back(
      makePose(pos, quat, "capsule.position", "capsule.orientation"));
    capsule.operation = moveit_msgs::msg::CollisionObject::ADD;

    return capsule;
  }

  void removeExistingObjects(
      moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
      const std::vector<std::string>& object_ids)
  {
    planning_scene_interface.removeCollisionObjects(object_ids);
  }

  void loadScene()
  {
    if (loaded_) {
      return;
    }
    loaded_ = true;
    timer_->cancel();

    try {
      const auto frame_id = get_parameter("frame_id").as_string();

      const auto table_size = get_parameter("table.size").as_double_array();
      const auto table_pos = get_parameter("table.position").as_double_array();
      const auto table_quat = get_parameter("table.orientation").as_double_array();

      const auto wall1_size = get_parameter("wall1.size").as_double_array();
      const auto wall1_pos = get_parameter("wall1.position").as_double_array();
      const auto wall1_quat = get_parameter("wall1.orientation").as_double_array();

      const auto wall2_size = get_parameter("wall2.size").as_double_array();
      const auto wall2_pos = get_parameter("wall2.position").as_double_array();
      const auto wall2_quat = get_parameter("wall2.orientation").as_double_array();

      const auto crate_mesh = get_parameter("crate.mesh_path").as_string();
      const auto crate_pos = get_parameter("crate.position").as_double_array();
      const auto crate_quat = get_parameter("crate.orientation").as_double_array();
      const auto crate_scale = get_parameter("crate.scale").as_double_array();

      const auto bottle_mesh = get_parameter("bottle.mesh_path").as_string();
      const auto bottle_pos = get_parameter("bottle.position").as_double_array();
      const auto bottle_quat = get_parameter("bottle.orientation").as_double_array();
      const auto bottle_scale = get_parameter("bottle.scale").as_double_array();

      const double capsule_radius = get_parameter("capsule.radius").as_double();
      const double capsule_height = get_parameter("capsule.height").as_double();
      const auto capsule_pos = get_parameter("capsule.position").as_double_array();
      const auto capsule_quat = get_parameter("capsule.orientation").as_double_array();

      moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
      std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

      RCLCPP_INFO(get_logger(), "Removing previous objects from planning scene...");
      removeExistingObjects(planning_scene_interface, {"table", "wall1", "wall2", "crate", "bottle", "capsule"});
      rclcpp::sleep_for(std::chrono::milliseconds(500));

      // Table
      collision_objects.push_back(
        makeBoxObject("table", frame_id, table_size, table_pos, table_quat));
      RCLCPP_INFO(get_logger(), "Prepared object: table");

      // Wall1
      collision_objects.push_back(
        makeBoxObject("wall1", frame_id, wall1_size, wall1_pos, wall1_quat));
      RCLCPP_INFO(get_logger(), "Prepared object: wall1");

      // Wall2
      collision_objects.push_back(
        makeBoxObject("wall2", frame_id, wall2_size, wall2_pos, wall2_quat));
      RCLCPP_INFO(get_logger(), "Prepared object: wall2");

      // Crate mesh
      if (!crate_mesh.empty()) {
        moveit_msgs::msg::CollisionObject crate;
        if (loadMeshObject("crate", frame_id, crate_mesh, crate_pos, crate_quat, crate_scale, crate)) {
          collision_objects.push_back(crate);
          RCLCPP_INFO(get_logger(), "Prepared object: crate");
        } else {
          RCLCPP_WARN(get_logger(), "Crate mesh could not be loaded, skipping crate.");
        }
      } else {
        RCLCPP_WARN(get_logger(), "crate.mesh_path is empty, crate will not be added.");
      }

      // Bottle mesh
      if (!bottle_mesh.empty()) {
        moveit_msgs::msg::CollisionObject bottle;
        if (loadMeshObject("bottle", frame_id, bottle_mesh, bottle_pos, bottle_quat, bottle_scale, bottle)) {
          collision_objects.push_back(bottle);
          RCLCPP_INFO(get_logger(), "Prepared object: bottle");
        } else {
          RCLCPP_WARN(get_logger(), "Bottle mesh could not be loaded, skipping bottle.");
        }
      } else {
        RCLCPP_WARN(get_logger(), "bottle.mesh_path is empty, bottle will not be added.");
      }

      // Capsule primitive
      collision_objects.push_back(
        makeCapsuleObject(frame_id, capsule_radius, capsule_height, capsule_pos, capsule_quat));
      RCLCPP_INFO(get_logger(), "Prepared object: capsule");

      planning_scene_interface.applyCollisionObjects(collision_objects);
      rclcpp::sleep_for(std::chrono::milliseconds(500));

      RCLCPP_INFO(get_logger(), "Scene loaded successfully.");
      RCLCPP_INFO(get_logger(), "  frame_id: %s", frame_id.c_str());
      RCLCPP_INFO(get_logger(), "  collision objects added: %zu", collision_objects.size());
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to load scene: %s", e.what());
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SceneLoaderBottleCapsuleNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}