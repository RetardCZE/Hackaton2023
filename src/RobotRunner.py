import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from scipy.spatial.transform import Rotation

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.mission
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry as geometry
import bosdyn.mission.client
import bosdyn.util
from bosdyn.api import geometry_pb2, image_pb2, world_object_pb2
from bosdyn.api.autowalk import walks_pb2
from bosdyn.api.data_acquisition_pb2 import AcquireDataRequest, DataCapture, ImageSourceCapture
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2
from bosdyn.api.mission import nodes_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.docking import DockingClient, docking_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.util import now_sec, seconds_to_timestamp
from bosdyn.client.robot import Robot
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
import bosdyn.client as BC
from PIL import Image
import io
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         get_se2_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME,
                                         GROUND_PLANE_FRAME_NAME)
import time
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
import io
import numpy as np
from bosdyn.api import geometry_pb2
import struct
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from DataReader import DataReader, World
from bosdyn.client import robot_command
import threading

class Robot:
    def __init__(self):
        self.x = None
        self.y = None
        self.Pathing = False
        sdk = bosdyn.client.create_standard_sdk('our-spot')
        self.robot = sdk.create_robot('192.168.80.3')
        bosdyn.client.util.authenticate(self.robot)
        self.reader = DataReader(self.robot)
        self.threads = []
        # Scatter plot
        self.reader.reset()
        self.reader.get_terrain_frames()

        position = self.reader.position()
        self.world = World(position)
        self._quit = True
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.robot_command_client = self.robot.ensure_client(robot_command.RobotCommandClient.default_service_name)

    def relative_move(self, dx, dy, angle, frame_name, robot_command_client, robot_state_client, stairs=False):
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Build the transform for where we want the robot to be relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=angle)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
        end_time = 10.0
        cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                    end_time_secs=time.time() + end_time)
        # Wait until the robot has reached the goal.
        while True:
            feedback = robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                return True
            time.sleep(1)

        return True

    def set_path(self):
        current_yaw = self.get_world_yaw()
        x, y = self.world.position
        cx = self.x
        cy = self.y
        angle = np.arctan2(cy - y, cx - x)

        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        # Setup clients for the robot state and robot command services.
        robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        robot_command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        ep = EstopEndpoint(estop_client, 'name', 30)
        ep.force_simple_setup()

        # Begin periodic check-in between keep-alive and robot
        estop_keep_alive = EstopKeepAlive(ep)
        estop_keep_alive.allow()

        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Power on the robot and stand it up.
            self.robot.time_sync.wait_for_sync()
            self.robot.power_on()
            blocking_stand(robot_command_client)

            try:
                self.relative_move(0, 0, angle, ODOM_FRAME_NAME, robot_command_client, robot_state_client, stairs=False)
            finally:
                # Send a Stop at the end, regardless of what happened.
                robot_command_client.robot_command(RobotCommandBuilder.stop_command())

    def run(self):
        mapper = threading.Thread(target=self.mapping)
        self.threads.append(mapper)
        mapper.start()

        while not self._quit:
            time.sleep(1)

        for t in self.threads:
            t.join()

    @staticmethod
    def rotate_coordinates(x, y, angle_radians):
        # angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                    [np.sin(angle_radians), np.cos(angle_radians)]])

        original_coordinates = np.array([x, y])
        transformed_coordinates = np.dot(rotation_matrix, original_coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    def get_world_yaw(self):
        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        transform = transforms.child_to_parent_edge_map.get('odom')
        yaw = (
            math_helpers.SE2Pose.flatten(math_helpers.SE3Pose.from_proto(transform.parent_tform_child).inverse())).angle
        return yaw

    def mapping(self):
        q = -1
        cv2.namedWindow('Terrain')
        cv2.setMouseCallback('Terrain', self.mouse_callback)
        while q != 113:
            grid = self.reader.get_terrain(0)

            grid_mask = self.reader.get_terrain(1)
            mask_off = np.mean(grid_mask[:, :, 2])
            mask_free = grid_mask[:, :, 2] > mask_off

            # mask_free = grid_mask[:, :, 2] == 1
            mean = np.mean(grid[mask_free, 2])

            mask_obstacles = grid[:, :, 2] > (mean + 0.15)
            mask_hidden = grid_mask[:, :, 2] <= mask_off

            grid[:, :, 2] = 255
            grid[mask_hidden, 2] = 122
            grid[mask_obstacles, 2] = 0

            self.world.write_grid(grid)
            # image = reader.show_terran_2d(grid)
            # image = reader.show_terran_opencv(grid)
            # stepper += 1
            # if stepper == 6:
            #    stepper = 0
            self.world.eval_weights()
            image = self.world.show_world()
            p = self.world.position
            x = p[0] - self.world.min_x
            y = p[1] - self.world.min_y

            vx, vy = self.rotate_coordinates(0, -1, self.get_world_yaw())

            color_mapped_image = cv2.applyColorMap((image).astype(np.uint8), cv2.COLORMAP_BONE)


            image = cv2.resize(color_mapped_image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

            image = cv2.circle(image, (y * 4, x * 4), 10, (0, 255, 0), -1)  # Green circle, -1 for filled
            image = cv2.line(image, (y * 4, x * 4), ((y+int(vy/self.world.resolution)) * 4, (x+int(vx/self.world.resolution)) * 4), (255, 0, 0), 2)

            cv2.imshow('Terrain', image)
            q = cv2.waitKey(100)
            # reader.reset()
        cv2.destroyAllWindows()
        self._quit = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.y = int(x/4) + self.world.min_x
            self.x = int(y/4) + self.world.min_y
            if self.world.show_world()[self.x-self.world.min_x, self.y-self.world.min_y] == 255:
                self.set_path()
                print(f"Clicked at FREE: ({self.x}, {self.y})")
            else:
                print(f"Clicked at OBSCURED: ({self.x}, {self.y})")

if __name__ == "__main__":
    os.environ['BOSDYN_CLIENT_USERNAME'] = 'user'
    os.environ['BOSDYN_CLIENT_PASSWORD'] = 'rik36otsjn73'

    r = Robot()
    r.run()

