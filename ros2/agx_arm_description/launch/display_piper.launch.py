from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    xacro_file = PathJoinSubstitution(
        [
            FindPackageShare("agx_arm_description"),
            "agx_arm_urdf",
            "piper",
            "urdf",
            "piper_with_gripper_description.xacro",
        ]
    )
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("agx_arm_description"), "rviz", "piper.rviz"]
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
