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

    def decode_rle(self, encoded_data, rle_counts, type):
        decoded_data = []
        data_index = 0

        #print(type)
        if type == 4:
            #print(len(encoded_data))
            #print(sum(rle_counts))
            for count in rle_counts:
                value = encoded_data[data_index]
                for i in range(count):
                    decoded_data.append([value])
                data_index += 1

        else:
            for count in rle_counts:
                value = struct.unpack('<' + 'h', encoded_data[data_index:data_index + 2])
                for i in range(count):
                    decoded_data.append([value])
                data_index += 2

        return decoded_data

    def show_terran_2d(self, grid):
        x_values = grid[:, :, 0]
        y_values = grid[:, :, 1]
        amplitude_values = grid[:, :, 2]

        # Create an image with white background
        image = np.ones((128, 128, 3), dtype=np.uint8) * 255
        # print(image.shape)
        # Set pixel colors based on amplitude values
        for x, y, amplitude in zip(*np.where(amplitude_values != 0), amplitude_values[amplitude_values != 0]):
            color = [255, 255, 255]  # Default to white
            if amplitude == -1:
                color = [128, 128, 128]  # Grey
            elif amplitude == 1:
                color = [0, 0, 0]  # Black

            image[y, x] = color

        # Show the image using cv2.imshow
        return image

    def show_terran_opencv(self, grid):
        # Extract x, y, and amplitude values from the grid
        x_values = grid[:, :, 0].flatten()
        y_values = grid[:, :, 1].flatten()
        amplitude_values = grid[:, :, 2].flatten()

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_values, y_values, amplitude_values, c=amplitude_values, cmap='viridis')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Amplitude')

        # Convert the plot to an image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer)

        # Close the plot
        plt.close(fig)
        return image

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

    def position(self):
        response = self._graph_nav_client.get_localization_state(request_live_terrain_maps=True)
        #print(response.live_data.robot_local_grids[0].transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child)
        pos = response.live_data.robot_local_grids[0].transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child.position
        return np.array([pos.x, pos.y, pos.z])

    def transform_point(self, position, rotation):
        # Convert the quaternion rotation to a rotation matrix
        rotation_matrix = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()

        # Append a row of ones to make it homogeneous coordinates
        homogeneous_coords = np.hstack([position.x, position.y, position.z])

        # Apply the transformation matrix
        transformed_homogeneous_coords = np.dot(rotation_matrix, homogeneous_coords)

        # Convert back to 3D coordinates
        transformed_point = transformed_homogeneous_coords[:3]

        return transformed_point

    def get_terrain(self, id):

        response = self._graph_nav_client.get_localization_state(request_live_terrain_maps=True)
        # print(response.live_data.robot_local_grids[id].transforms_snapshot.child_to_parent_edge_map) #['odom'].parent_tform_child)

        position = response.live_data.robot_local_grids[id].transforms_snapshot.child_to_parent_edge_map['terrain_local_grid_corner'].parent_tform_child.position
        # position = response.live_data.robot_local_grids[id].transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child.position
        #rotation = response.live_data.robot_local_grids[id].transforms_snapshot.child_to_parent_edge_map['odom'].parent_tform_child.rotation

        # [rotation.x, rotation.y, rotation.z, rotation.w]
        #rotation_matrix = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
        # Combine translation and rotation into a transformation matrix
        transformation_matrix = np.eye(4)
        #transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [position.x, position.y, position.z]

        #print(response)
        #exit()
        decoded_terrain_data = self.decode_rle(response.live_data.robot_local_grids[id].data,
                                               response.live_data.robot_local_grids[id].rle_counts,
                                               response.live_data.robot_local_grids[id].cell_format)
        decoded_terrain_data = np.array(decoded_terrain_data).astype('float')
        scale = float(response.live_data.robot_local_grids[id].cell_value_scale)
        offset = float(response.live_data.robot_local_grids[id].cell_value_offset)
        if scale == 0:
            scale = 1
            offset = 0
        decoded_terrain_data = scale * decoded_terrain_data + offset
        x_resolution = response.live_data.robot_local_grids[id].extent.num_cells_x
        y_resolution = response.live_data.robot_local_grids[id].extent.num_cells_y
        increment = response.live_data.robot_local_grids[id].extent.cell_size
        #print(increment)
        grid_2d = np.reshape(decoded_terrain_data, (x_resolution, y_resolution))

        # Create the x and y values using NumPy's arange function
        x_values = np.arange(0, x_resolution) * increment
        y_values = np.arange(0, y_resolution) * increment

        # Create a meshgrid from x and y values
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        # Stack the x, y, and amplitude values to create the 3D array
        grid_3d = np.stack((x_mesh, y_mesh, grid_2d), axis=-1)
        #print(transformation_matrix, grid_3d.shape)

        #transformation_matrix = np.linalg.inv(transformation_matrix)

        flattened_grid = grid_3d.reshape(-1, 3).T

        # Append a row of ones to make it homogeneous coordinates
        homogeneous_coords = np.vstack([flattened_grid, np.ones(flattened_grid.shape[1])])

        # Apply the transformation matrix
        transformed_homogeneous_coords = np.dot(transformation_matrix, homogeneous_coords)

        # Convert back to 3D coordinates
        transformed_grid_3d = transformed_homogeneous_coords[:3, :].T.reshape(grid_3d.shape)

        #grid_3d[:, :, 0] -= position.x
        #grid_3d[:, :, 1] -= position.y
        #grid_3d[:, :, 2] -= position.z
        return transformed_grid_3d

    def get_terrain_frames(self):
        response = self._graph_nav_client.get_localization_state(request_live_terrain_maps=True)
        for lg in response.live_data.robot_local_grids:
            print(lg.local_grid_type_name)

class World:
    def __init__(self):
        self.resolution = 0.1
        self.size = 50
        self.world = 100 * np.ones((int(self.size/self.resolution), int(self.size/self.resolution)))
        self.weights = np.ones((int(self.size / self.resolution), int(self.size / self.resolution)))
        self.origin_coors = [int((self.size/self.resolution)/2), int((self.size/self.resolution)/2)]
        self.min_x = 10000
        self.max_x = -10000
        self.min_y = 10000
        self.max_y = -10000

    def xy2coords(self, x, y):
        x_coords = self.origin_coors[0] + np.divide(x, self.resolution).astype('int')
        y_coords = self.origin_coors[1] + np.divide(y, self.resolution).astype('int')
        return x_coords, y_coords

    def write_grid(self, grid):
        x = grid[:, :, 0].reshape((128*128, 1))
        y = grid[:, :, 1].reshape((128*128, 1))
        z = grid[:, :, 2].reshape((128*128, 1))
        xCoords, yCoords = self.xy2coords(x, y)
        self.min_y = min(self.min_y, np.min(yCoords))
        self.min_x = min(self.min_x, np.min(xCoords))
        self.max_y = max(self.max_y, np.max(yCoords))
        self.max_x = max(self.max_x, np.max(xCoords))
        print(self.min_x, self.max_x, self.min_y, self.max_y)
        for p in np.hstack([xCoords, yCoords, z]).astype('int'):
            self.world[p[0], p[1]] += p[2]
            self.weights [p[0], p[1]] += 1

    def eval_weights(self):
        self.world = np.divide(self.world, self.weights)
        obst = self.world[:, :] > 200
        free = self.world[:, :] < 70
        rest = np.logical_not(np.logical_or(obst, free))
        self.world[self.world[:, :] > 200] = 255
        self.world[self.world[:, :] < 70] = 0
        self.world[rest] = 122
        self.weights = self.weights * 0 + 1

    def show_world(self):
        return self.world[self.min_x:self.max_x, self.min_y:self.max_y]

if __name__ == "__main__":
    os.environ['BOSDYN_CLIENT_USERNAME'] = 'user'
    os.environ['BOSDYN_CLIENT_PASSWORD'] = 'rik36otsjn73'

    sdk = bosdyn.client.create_standard_sdk('our-spot')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    reader = DataReader(robot)

    # Scatter plot
    reader.reset()
    reader.get_terrain_frames()
    q = -1
    grid = reader.get_terrain(0)
    world = World()
    stepper = 0
    sumPos = reader.position()
    pos = reader.position()
    while q != 113:

        grid = reader.get_terrain(0)
        grid_mask = reader.get_terrain(1)
        mask_off = np.mean(grid_mask[:, :, 2])
        mask_free = grid_mask[:, :, 2] > mask_off

        # mask_free = grid_mask[:, :, 2] == 1
        mean = np.mean(grid[mask_free, 2])

        mask_obstacles = grid[:, :, 2] > (mean + 0.1)
        mask_hidden = grid_mask[:, :, 2] <= mask_off

        grid[:, :, 2] = 255
        grid[mask_hidden, 2] = 122
        grid[mask_obstacles, 2] = 0

        world.write_grid(grid)
        #image = reader.show_terran_2d(grid)
        #image = reader.show_terran_opencv(grid)
        #stepper += 1
        #if stepper == 6:
        #    stepper = 0
        world.eval_weights()
        image = world.show_world()
        color_mapped_image = cv2.applyColorMap((image).astype(np.uint8) , cv2.COLORMAP_BONE)


        image = cv2.resize(color_mapped_image, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Terrain', image)
        q = cv2.waitKey(200)
        #reader.reset()
    cv2.destroyAllWindows()