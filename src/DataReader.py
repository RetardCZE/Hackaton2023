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
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

class DataReader:
    def __init__(self, robot: Robot):
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

    def reset(self):
        self._graph_nav_client.clear_graph()

    def decode(self, data, size):
        floats = []
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

    def decode_rle(self, encoded_data, rle_counts):
        decoded_data = []
        data_index = 0

        for count in rle_counts:
            value = struct.unpack('<' + 'h', encoded_data[data_index:data_index + 2])
            for i in range(count):
                decoded_data.append([value])
            data_index += 2

        return decoded_data

    def show_terran(self, grid):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract x, y, and amplitude values from the grid
        x_values = grid[:, :, 0]
        y_values = grid[:, :, 1]
        amplitude_values = grid[:, :, 2]

        # Create a 3D surface plot
        surface = ax.plot_surface(x_values, y_values, amplitude_values, cmap='viridis')

        # Add a colorbar to the plot
        fig.colorbar(surface, ax=ax, label='Amplitude')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Amplitude')

        # Show the plot
        plt.show()
    def get_terrain(self):
        response = self._graph_nav_client.get_localization_state(request_live_terrain_maps=True)
        #print(response.live_data.robot_local_grids[0])
        decoded_terrain_data = self.decode_rle(response.live_data.robot_local_grids[0].data, response.live_data.robot_local_grids[0].rle_counts)
        decoded_terrain_data = np.array(decoded_terrain_data).astype('float')
        scale = float(response.live_data.robot_local_grids[0].cell_value_scale)
        offset = float(response.live_data.robot_local_grids[0].cell_value_offset)
        decoded_terrain_data = scale * decoded_terrain_data + offset
        x_resolution = response.live_data.robot_local_grids[0].extent.num_cells_x
        y_resolution = response.live_data.robot_local_grids[0].extent.num_cells_y
        increment = response.live_data.robot_local_grids[0].extent.cell_size

        grid_2d = np.reshape(decoded_terrain_data, (x_resolution, y_resolution))

        # Create the x and y values using NumPy's arange function
        x_values = np.arange(0, x_resolution) * increment
        y_values = np.arange(0, y_resolution) * increment

        # Create a meshgrid from x and y values
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        # Stack the x, y, and amplitude values to create the 3D array
        grid_3d = np.stack((x_mesh, y_mesh, grid_2d), axis=-1)
        return grid_3d


if __name__ == "__main__":
    os.environ['BOSDYN_CLIENT_USERNAME'] = 'user'
    os.environ['BOSDYN_CLIENT_PASSWORD'] = 'rik36otsjn73'

    sdk = bosdyn.client.create_standard_sdk('our-spot')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    reader = DataReader(robot)

    # Scatter plot
    reader.reset()
    reader.get_terrain()
    reader.show_terran()