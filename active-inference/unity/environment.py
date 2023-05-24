import os

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .enums import *


class UnityContainer:
    """
    Class encapsulating the ML-Agents connection with Unity allowing easy interaction with the simulated environment
    """
    VISUAL_OBSERVATION_INDEX = 0
    VECTOR_OBSERVATIONS_INDEX = 1

    # Path of the pre-build environment
    BUILD_PATH = os.path.join(
        os.path.dirname(__file__), "../../RHI-Agent/Builds/env.x86_64")

    def __init__(self, use_editor, time_scale=1):
        """
        Set up the Unity environment
        :param use_editor: Set to true to connect directly to the Unity editor, set to false to use the pre-build
        environment at BUILD_PATH
        :param time_scale: time_scale of the environment (1 is normal time)
        """
        if use_editor:
            self.env_path = None
        else:
            self.env_path = self.BUILD_PATH
        self.time_scale = time_scale
        self.env = None
        self.env_params_channel = None
        self.behavior_name = None
        self.spec = None

        self.env_condition = None
        self.env_visible_arm = None
        self.env_stimulation = None
        self.env_mode = None
        self.env_ball_range = None

    def initialise_environment(self):
        """Initialise and reset unity environment"""
        engine_configuration_channel = EngineConfigurationChannel()
        self.env_params_channel = EnvironmentParametersChannel()
        self.env = UnityEnvironment(file_name=self.env_path, base_port=5004,
                                    side_channels=[engine_configuration_channel, self.env_params_channel])

        # Reset the environment
        self.env.reset()

        # Set the default brain to work with
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        # Set the time scale of the engine
        engine_configuration_channel.set_configuration_parameters(
            time_scale=self.time_scale)

    def set_condition(self, condition: Condition):
        """Sets the experimental condition setting"""
        self.env_condition = condition
        self.env_params_channel.set_float_parameter(
            "condition", condition.value)

    def set_visible_arm(self, visible_arm: VisibleArm):
        """Sets the visible arm setting"""
        self.env_visible_arm = visible_arm
        self.env_params_channel.set_float_parameter(
            "visiblearm", visible_arm.value)

    def set_stimulation(self, stimulation: Stimulation):
        """Sets the stimulation setting"""
        self.env_stimulation = stimulation
        self.env_params_channel.set_float_parameter(
            "stimulation", stimulation.value)

    def set_mode(self, mode: Mode):
        """Sets the mode setting"""
        self.env_mode = mode
        self.env_params_channel.set_float_parameter(
            "mode", mode.value)

    def set_ball_range(self, ball_range: BallRange):
        """Sets the ball range setting"""
        self.env_ball_range = ball_range
        self.env_params_channel.set_float_parameter(
            "ball_range_min", ball_range.b_min)
        self.env_params_channel.set_float_parameter(
            "ball_range_max", ball_range.b_max)

    def get_joint_observation(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns joint angles of the agent"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][:, :2]

    def get_touch_observation(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns last visual and tactile touch events"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][:, 2:4]

    def get_current_env_time(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns current env time"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][:, 4]

    def get_cartesian_distance(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns cartesian euclidean (absolute) distance between real hand and rubber hand"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][0, 5]

    def get_horizontal_distance(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns horizontal euclidean distance between real hand and rubber hand"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][0, 6]

    def get_rubber_joint_observation(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns joint angles of the rubber arm"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][:, 7:9]

    def get_active_ball_distance(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns distance between the active ball and real arm"""
        return decision_steps.obs[self.VECTOR_OBSERVATIONS_INDEX][:, 9]

    def get_visual_observation(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        """:returns visual perception of the agent"""
        return decision_steps.obs[self.VISUAL_OBSERVATION_INDEX][0]

    def act(self, action):
        """Make the agent perform an action (velocity) in the environment"""
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.append([[0]], action, axis=1))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    def set_rotation(self, rotation):
        """Manually set the joint angles to a particular rotation"""
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.append([[1]], rotation,  axis=1))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    def set_rubber_arm_rotation(self, rotation):
        """Manually set the joint angles of the rubber arm to a particular rotation"""
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.append([[2]], rotation,  axis=1))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    def set_active_ball_yAxis(self, yAxis):
        """Manually set the yAxis of the active ball"""
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.append([[3]], yAxis,  axis=1))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    def reset(self):
        """Reset the environment"""
        self.env.reset()

    def close(self):
        """Gracefully close the environment"""
        self.env.close()
