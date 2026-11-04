# RC car and MID-360 recording guide

This guide starts with a powered-off RC car and ends with a ROS 2 recording on
the laptop, ready to inspect in CloudCompare. It is for **manual driving**: the
car's Follow-the-Gap/autonomous node and the original Hokuyo LiDAR are not
started by the manual bringup.

Keep the car lifted while testing motor/steering commands. Use the radio's
manual-driving mode before placing it on the ground.

## Network overview

| Device | Connection | Address |
| --- | --- | --- |
| Laptop | Wi-Fi `arc26s_team3` | assigned by the car Wi-Fi network, normally `192.168.103.x` |
| Car computer | Wi-Fi | `192.168.103.10` |
| MID-360 | Ethernet on the car | `192.168.1.104` |
| Car Ethernet interface | Ethernet on the car | `192.168.1.10` in addition to any existing Ethernet address |

Wi-Fi and SSH credentials are deliberately not stored in this repository. Use
the credentials supplied with the car.

## 1. Power on and connect from the laptop

1. Power on the car and wait for its Wi-Fi network to appear.
2. Connect the laptop to `arc26s_team3`.
3. In a laptop terminal, verify that the car is reachable and log in:

   ```bash
   ping -c 3 192.168.103.10
   ssh username@192.168.103.10
   ```

   Replace `username` with the car account name. The first SSH connection asks
   whether to trust the car's host key; verify it with the team if it ever
   changes unexpectedly.

If ping reports `Destination Host Unreachable`, confirm that the laptop is
actually connected to `arc26s_team3` and that the car is fully powered on.

## 2. Check the MID-360 Ethernet link

On the car over SSH, run:

```bash
ip -4 -br addr show dev enp86s0
ping -I enp86s0 -c 3 192.168.1.104
```

The ping should receive replies from `192.168.1.104`. The Ethernet interface
must retain `192.168.1.10/24` for the MID-360. It may also show another
Ethernet address (for example `192.168.0.15/24`); that is fine and supports
other devices on the same port.

If `192.168.1.10/24` is missing, restore the saved NetworkManager profile:

```bash
sudo nmcli connection up lidar
ip -4 -br addr show dev enp86s0
```

Do not start a recording until the MID-360 responds to ping.

## 3. Start the car for manual driving

Open the first SSH terminal on the car. Use Kilted only; do **not** source the
Jazzy ROS installation in this workspace.

```bash
source /opt/ros/kilted/setup.bash
source ~/racecar_ws/install/setup.bash
ros2 launch ftg_py ftg_bringup.launch.py
```

This starts the VESC, joystick, manual command mux, odometry, transform, and
telemetry nodes. It should not start `ftg_py_node` or `urg_node`.

In a second SSH terminal, the following confirms that manual-driving hardware
is available:

```bash
source /opt/ros/kilted/setup.bash
source ~/racecar_ws/install/setup.bash
ros2 node list
ros2 topic list -t
```

Expected useful topics include `/joy`, `/drive`, `/odom`, `/sensors/core`, and
`/tf`.

## 4. Start the MID-360 ROS driver

Open a second (or third) SSH terminal on the car:

```bash
source /opt/ros/kilted/setup.bash
source ~/livox_ws/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

Wait for messages indicating that the lidar changed to `Normal` work mode and
that `/livox/lidar` is publishing. Check the signal from another terminal:

```bash
source /opt/ros/kilted/setup.bash
source ~/livox_ws/install/setup.bash
ros2 topic list -t | grep livox
timeout 5 ros2 topic hz /livox/lidar
```

The expected topics are `/livox/lidar` with type
`livox_ros_driver2/msg/CustomMsg` and `/livox/imu` with type
`sensor_msgs/msg/Imu`. A rate close to 10 Hz is normal for the configured
driver.

## 5. Record a smoke drive

Keep the manual bringup and the Livox driver running. In a fresh SSH terminal,
choose a new, descriptive bag name; do not reuse an existing directory.

```bash
source /opt/ros/kilted/setup.bash
source ~/racecar_ws/install/setup.bash
source ~/livox_ws/install/setup.bash
mkdir -p ~/bags
ros2 bag record -o ~/bags/mid360_smoke_001 \
  /livox/lidar /livox/imu /odom /tf /tf_static /sensors/core
```

Start driving only after the recorder reports that it is subscribing. When the
run is finished, stop the recorder with `Ctrl-C` and wait until it says it has
finished writing. Then verify the recording:

```bash
ros2 bag info ~/bags/mid360_smoke_001
```

The output should list an `.mcap` file and the six recorded topics. In
particular, `/livox/lidar` must have type
`livox_ros_driver2/msg/CustomMsg` and a non-zero message count.

For the next run, increment the name, for example `mid360_smoke_002`. Record a
short clean-background run before a smoke session whenever possible; it will
help with later motion-aware labeling.

## 6. Transfer the finished bag to the laptop

Run this on the **laptop**, not on the car, after the bag recorder has stopped:

```bash
cd /home/user/Smoke/LidarSmokeDetection
scp -r username@192.168.103.10:~/bags/mid360_smoke_001 data/raw_car/
```

It creates `data/raw_car/mid360_smoke_001/` containing the MCAP recording and
metadata. Verify the local copy:

```bash
labeler/.venv/bin/smoke-label inspect data/raw_car/mid360_smoke_001
```

Do not delete the car-side recording until this check succeeds and the copied
files have the expected size.

## 7. Inspect the recording visually in CloudCompare

CloudCompare cannot open a ROS 2 MCAP bag directly. Export a lightweight PLY
preview on the laptop:

```bash
cd /home/user/Smoke/LidarSmokeDetection
labeler/.venv/bin/smoke-label raw-preview data/raw_car/mid360_smoke_001 \
  --output data/raw_car/mid360_smoke_001/mid360_smoke_001_preview.ply
```

Open the generated `.ply` in CloudCompare using **File → Open**. The preview
combines sampled, raw sensor-frame scans and colours points using return
intensity. It is intended to check that the scene and smoke signal were
captured; it does not motion-align frames from a moving car.

For a denser or smaller preview, adjust `--frame-stride`, `--point-stride`, and
`--max-points`. Smaller stride values retain more data.

## 8. Shut down

1. Stop the bag recorder first (`Ctrl-C`).
2. Stop the Livox driver (`Ctrl-C`).
3. Stop the manual car bringup (`Ctrl-C`).
4. Power down the car only after the software has stopped.

The raw bag, its metadata, and any exported PLY should remain together under
`data/raw_car/<recording-name>/` on the laptop.
