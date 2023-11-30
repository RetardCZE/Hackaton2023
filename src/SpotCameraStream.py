
from bosdyn.client.image import ImageClient
import bosdyn.client as BC
import cv2
import io
import time
import os
import numpy as np
from bosdyn.client.robot import Robot

class SpotCameraStream:
    def __init__(self, robot: Robot, sources: list=[]):

        image_client = robot.ensure_client(ImageClient.default_service_name)
        sources = image_client.list_image_sources()

if __name__ == '__main__':
    os.environ['BOSDYN_CLIENT_USERNAME'] = 'user'
    os.environ['BOSDYN_CLIENT_PASSWORD'] = 'rik36otsjn73'

    sdk = BC.create_standard_sdk('our-spot')
    robot = sdk.create_robot('192.168.80.3')
    BC.util.authenticate(robot)

    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()
    for s in sources:
        print(s.name)

    q = -1
    while q != 113:
        image_responses = image_client.get_image_from_sources(['right_depth_in_visual_frame', 'right_fisheye_image'])
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
            cv_visual, cv2.COLOR_GRAY2RGB)

        # Map depth ranges to color

        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        # Add the two images together.
        # out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

        cv2.imshow('win', depth_color)  # rik36otsjn73
        q = cv2.waitKey(10)
