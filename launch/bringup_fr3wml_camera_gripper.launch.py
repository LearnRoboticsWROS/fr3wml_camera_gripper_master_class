from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
import os
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_share = get_package_share_directory('fr3wml_fr5_camera_gripper_moveit_config')

    moveit_config = (
        MoveItConfigsBuilder("fairino3mt_v6_robot", package_name="fr3wml_fr5_camera_gripper_moveit_config")
        .robot_description(file_path="config/fairino3mt_v6_robot.urdf.xacro")
        .robot_description_semantic(file_path="config/fairino3mt_v6_robot.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True
        )
        .to_moveit_configs()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"controllers_file": os.path.join(pkg_share, "config", "moveit_controllers.yaml")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_config_path = os.path.join(pkg_share, "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[moveit_config.to_dict()],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    moveit_joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["moveit_joint_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    delay_moveit_joint_controller = RegisterEventHandler(
        OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[moveit_joint_controller_spawner],
        )
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
        rviz_node,
        joint_state_broadcaster_spawner,
        delay_moveit_joint_controller,
    ])
