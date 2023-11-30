import argparse
import logging
import os
import sys
import time

import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

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
import io
import numpy as np
import struct

class DataReader:
    def __init__(self, robot: Robot):
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

    def reset(self):
        self._graph_nav_client.clear_graph()

    def decode(self, data, size):
        floats = []
        print(size)
        for i in range(size):
            x = struct.unpack('f', data[i:i+4])
            y = struct.unpack('f', data[i+4:i+8])
            z = struct.unpack('f', data[i+8:i+12])
            floats.append([x, y, z])

        array = np.array(floats)
        return array

    def get_pc(self):
        response = self._graph_nav_client.get_localization_state(request_live_point_cloud=True)

        raw_data = response.live_data.point_cloud.data
        return self.decode(raw_data, response.live_data.point_cloud.num_points)


if __name__ == "__main__":
    os.environ['BOSDYN_CLIENT_USERNAME'] = 'user'
    os.environ['BOSDYN_CLIENT_PASSWORD'] = 'rik36otsjn73'

    sdk = bosdyn.client.create_standard_sdk('our-spot')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    reader = DataReader(robot)
    reader.get_pc()