#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

const double tau = 2 * M_PI;

int main(int argc, char **argv)
{
    // ROS2 Initialization
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("move_group_interface");

    // Logger
    auto logger = rclcpp::get_logger("move_group_interface");

    // Spinner with more thread for avoiding blocks
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner_thread([&executor]() { executor.spin(); });

    // Wait initialization
    rclcpp::sleep_for(std::chrono::seconds(2));

    // MoveIt2 interface
    using moveit::planning_interface::MoveGroupInterface;
    MoveGroupInterface move_group(node, "fr3wml");
    move_group.setPoseReferenceFrame("base_link");
    move_group.setPlanningTime(10.0);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    RCLCPP_INFO(logger, "Pose reference frame set to: %s", move_group.getPoseReferenceFrame().c_str());

    geometry_msgs::msg::Pose target_pose;
    tf2::Quaternion orientation;
    orientation.setRPY(0, 0, 0);
    target_pose.orientation = tf2::toMsg(orientation);
    target_pose.position.x = -0.3;
    target_pose.position.y = -0.3;
    target_pose.position.z = 0.5;

    move_group.setPoseTarget(target_pose, "wrist3_link");

    RCLCPP_INFO(logger, "Planning frame: %s", move_group.getPlanningFrame().c_str());
    RCLCPP_INFO(logger, "End effector link: %s", move_group.getEndEffectorLink().c_str());


    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.resize(2);

    collision_objects[0].id = "table";
    collision_objects[0].header.frame_id = "base_link";
    collision_objects[0].primitives.resize(1);
    collision_objects[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[0].primitives[0].dimensions = {0.6, 2.0, 1.0};
    collision_objects[0].primitive_poses.resize(1);
    collision_objects[0].primitive_poses[0].position.x = 0.1;
    collision_objects[0].primitive_poses[0].position.y = 0.0;
    collision_objects[0].primitive_poses[0].position.z = -0.5;
    // ORIENTATION (relative to base_link)
    {
    const double deg2rad = M_PI / 180.0;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, -45.0 * deg2rad);  // yaw 45°
    q.normalize();
    collision_objects[0].primitive_poses[0].orientation = tf2::toMsg(q);
    }
    collision_objects[0].operation = moveit_msgs::msg::CollisionObject::ADD;


    collision_objects[1].id = "wall";
    collision_objects[1].header.frame_id = "base_link";
    collision_objects[1].primitives.resize(1);
    collision_objects[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[1].primitives[0].dimensions = {0.1, 2.0, 2.0};
    collision_objects[1].primitive_poses.resize(1);
    collision_objects[1].primitive_poses[0].position.x = 0.4;
    collision_objects[1].primitive_poses[0].position.y = -0.3;
    collision_objects[1].primitive_poses[0].position.z = 0.0;
    // ORIENTATION (relative to base_link)
    {
    const double deg2rad = M_PI / 180.0;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, -45.0 * deg2rad);  // yaw 45°
    q.normalize();
    collision_objects[1].primitive_poses[0].orientation = tf2::toMsg(q);
    }
    collision_objects[1].operation = moveit_msgs::msg::CollisionObject::ADD;




    // Add objects to the scene
    planning_scene_interface.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(logger, "Collision objects added to the planning scene.");

    // Planning
    MoveGroupInterface::Plan my_plan;
    bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(logger, "Visualizing plan: %s", success ? "SUCCESS" : "FAILED");

    // Execution
    if (success)
    {
        move_group.move();
        RCLCPP_INFO(logger, "Motion execution completed.");
    }
    else
    {
        RCLCPP_ERROR(logger, "Motion planning failed!");
    }

    // stop the spinner
    rclcpp::shutdown();
    spinner_thread.join();
    return 0;
}