import os
from launch import LaunchDescription
from launch.actions import TimerAction, RegisterEventHandler, DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_share = get_package_share_directory('fr3wml_fr5_camera_gripper_moveit_config')
    this_pkg = get_package_share_directory("fr3wml_camera_gripper")

    default_scene_file = os.path.join(this_pkg, "config", "scene_bottle_capsule.yaml")
    default_params_file = os.path.join(this_pkg, "config", "pick_place_bottle_capsule_framework.yaml")

    scene_file_arg = DeclareLaunchArgument(
        "scene_file",
        default_value=default_scene_file,
        description="Path to the scene_handling YAML file"
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to the pick_place_framework YAML file"
    )

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

    # Long-lived owner of the planning scene (initial load + dynamic add/remove)
    scene_handling_node = Node(
        package="fr3wml_camera_gripper",
        executable="scene_handling",
        name="scene_loader_node",  # matches root key in scene_bottle_capsule.yaml
        output="screen",
        parameters=[LaunchConfiguration("scene_file")],
    )

    # Motion-only pick and place
    pick_place_node = Node(
        package="fr3wml_camera_gripper",
        executable="pick_place_framework",
        name="pick_place_framework",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    delay_moveit_joint_controller = RegisterEventHandler(
        OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[moveit_joint_controller_spawner],
        )
    )

    # scene_handling first (it loads the initial scene and exposes services)
    delayed_scene_handling = TimerAction(
        period=5.0,
        actions=[scene_handling_node]
    )

    # pick_place after scene_handling has loaded and is serving
    delayed_pick_place = TimerAction(
        period=10.0,
        actions=[pick_place_node]
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
        rviz_node,
        joint_state_broadcaster_spawner,
        scene_file_arg,
        params_file_arg,
        delay_moveit_joint_controller,
        delayed_scene_handling,
        delayed_pick_place,
    ])