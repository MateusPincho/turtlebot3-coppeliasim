import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import numpy as np
import matplotlib.pyplot as plt


def draw_robot(x, y, q, s, h):
    p = np.zeros(36).reshape(12, 3)
    p[0, :] = [1, 1 / 7, 1 / s]
    p[1, :] = [-3 / 7, 1, 1 / s]
    p[2, :] = [-5 / 7, 6 / 7, 1 / s]
    p[3, :] = [-5 / 7, 5 / 7, 1 / s]
    p[4, :] = [-3 / 7, 2 / 7, 1 / s]
    p[5, :] = [-3 / 7, 0, 1 / s]
    p[6, :] = [-3 / 7, -2 / 7, 1 / s]
    p[7, :] = [-5 / 7, -5 / 7, 1 / s]
    p[8, :] = [-5 / 7, -6 / 7, 1 / s]
    p[9, :] = [-3 / 7, -1, 1 / s]
    p[10, :] = [1, -1 / 7, 1 / s]
    p[11, :] = [1, 1 / 7, 1 / s]

    p = s * p

    r = np.zeros(6).reshape(3, 2)
    r[0, :] = [np.cos(q), np.sin(q)]
    r[1, :] = [-np.sin(q), np.cos(q)]
    r[2, :] = [x, y]

    p = np.dot(p, r)
    X = p[:, 0]
    Y = p[:, 1]
    h.plot(X, Y, "r-")


def read_lidar(point_cloud_handle):
    point_cloud = np.array(sim.getPointCloudPoints(point_cloud_handle, 0)).reshape(-1, 3).T

    return point_cloud


def lidar_to_world(point_cloud, lidar_handle):
    H = np.array(sim.getObjectMatrix(lidar_handle, -1)).reshape(3, 4)

    R = H[:, :3]
    T = H[:, 3].reshape(3, 1)

    point_cloud = (R @ point_cloud) + T

    return point_cloud


print("Starting ZMQ Client")
client = RemoteAPIClient()
sim = client.getObject("sim")
sim.setStepping(True)
hd = sim.getSimulationTimeStep()

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Get handles
robot_handle = sim.getObject("/Turtlebot3/base_link")
lidar_handle = sim.getObject("/Turtlebot3/scan_joint")

# Simulation parameters
npts = 200
tf = npts * hd
kd = 0
tc = 0
ta = np.zeros(npts)
xp = np.zeros(npts)
yp = np.zeros(npts)
fp = np.zeros(npts)
point_cloud_map = []

# Starting simulation
sim.startSimulation()

point_cloud_handle = sim.getObject("/Turtlebot3/scan_joint/point_cloud")

while tc < tf:
    # Save robot states
    ta[kd] = tc
    robot_position = sim.getObjectPosition(robot_handle, -1)
    robot_orientation = sim.getObjectOrientation(robot_handle, -1)

    xp[kd] = robot_position[0]
    yp[kd] = robot_position[1]
    fp[kd] = robot_orientation[2]

    # Map environment
    point_cloud = read_lidar(point_cloud_handle)
    point_cloud_world = lidar_to_world(point_cloud, lidar_handle)
    point_cloud_map.append(point_cloud_world)

    # Update time variables
    tc += hd
    kd += 1
    sim.step()

sim.stopSimulation()

# If you need to make sure we really stopped
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

# Restore the original idle loop frequency
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

# Join all mapping into a dataset
point_cloud_map = np.hstack(point_cloud_map)

# Plot trajectory + map 
fig, ax = plt.subplots()
ax.axis("equal")
ax.plot(xp, yp, color="blue", linestyle="dashed", linewidth=1)
plt.grid()
plt.title("Top view: robot trajectory")
plt.xlabel("x, m")
plt.ylabel("y, m")
plt.show(block=False)

for i in range(0, len(xp) - 1, int(round(len(xp) / 20))):
    draw_robot(xp[i], yp[i], fp[i], 0.01, ax)

ax.scatter(x=point_cloud_map[0], y=point_cloud_map[1], s=1, color="red")

plt.show()
