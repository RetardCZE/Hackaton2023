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
                                         get_se2_a_tform_b)
import time
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive

def relative_move(dx, frame_name, robot_command_client, robot_state_client, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=0, angle=0)
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


sdk = BC.create_standard_sdk('our-spot')
robot = sdk.create_robot('192.168.80.3')
BC.util.authenticate(robot)
lease_client = robot.ensure_client(BC.lease.LeaseClient.default_service_name)
lease = lease_client.take()

image_client = robot.ensure_client(ImageClient.default_service_name)
sources = image_client.list_image_sources()

for i in range(5):
    image_response = image_client.get_image_from_sources(["left_fisheye_image"])[0]

img = Image.open(io.BytesIO(image_response.shot.image.data))
img.show() #rik36otsjn73

sdk = BC.create_standard_sdk('192.168.80.3')
robot = sdk.create_robot('192.168.80.3')


BC.util.authenticate(robot)

# Check that an estop is connected with the robot so that the robot commands can be executed.
assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                'such as the estop SDK example, to configure E-Stop.'

# Create the lease client.
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()
# Setup clients for the robot state and robot command services.
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

estop_client = robot.ensure_client(EstopClient.default_service_name)
ep = EstopEndpoint(estop_client, 'name', 30)
ep.force_simple_setup()

# Begin periodic check-in between keep-alive and robot
estop_keep_alive = EstopKeepAlive(ep)
estop_keep_alive.allow()

with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    # Power on the robot and stand it up.
    robot.time_sync.wait_for_sync()
    robot.power_on()
    blocking_stand(robot_command_client)

    try:
        relative_move(1, ODOM_FRAME_NAME, robot_command_client, robot_state_client, stairs=False)
    finally:
        # Send a Stop at the end, regardless of what happened.
        robot_command_client.robot_command(RobotCommandBuilder.stop_command())
