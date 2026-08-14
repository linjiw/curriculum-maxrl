"""CPU-only BARN/Gazebo adapter for the ICRA 2027 campaign.

Teacher tasks are frozen difficulty strata.  On each requested stratum, one
training course is sampled uniformly and a group of N stochastic rollouts is
run in the canonical BARN Gazebo world.  This makes the posterior sufficiently
dense to be useful while retaining course-level outcome and trajectory logs.
Held-out evaluation addresses immutable courses directly and never mutates the
training RNG, posterior, policy, or budget counters.
"""

from __future__ import annotations

import json
import heapq
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from frontier_rl.interfaces import GroupResult


START_XY = (-2.25, 3.0)
START_YAW = math.pi / 2.0
GOAL_XY = (-2.25, 13.0)


@dataclass(frozen=True)
class BarnCourse:
    env_id: str
    barn_index: int
    difficulty: float
    world_path: Path
    path_path: Path
    asset_sha256: str


def load_courses(manifest: Path, dataset_root: Path,
                 ids: Iterable[str]) -> list[BarnCourse]:
    wanted = set(ids)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()
            if line.strip()]
    by_id = {row["env_id"]: row for row in rows}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"course ids missing from manifest: {missing}")
    courses = []
    for env_id in wanted:
        row = by_id[env_id]
        path = (dataset_root / row["asset"]).resolve()
        course_path = (dataset_root / row["path_asset"]).resolve()
        if not path.is_file() or not course_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else course_path)
        courses.append(BarnCourse(
            env_id=str(env_id), barn_index=int(row["barn_index"]),
            difficulty=float(row["difficulty"]), world_path=path,
            path_path=course_path,
            asset_sha256=str(row["asset_sha256"])))
    return sorted(courses, key=lambda c: (c.difficulty, c.env_id))


class LidarVelocityPolicy:
    """Small NumPy lidar/subgoal-to-velocity residual policy.

    A deterministic reactive prior keeps initial rollouts informative.  A
    13→24→2 tanh network supplies a learned residual and is optimized by the
    score-function weights supplied by ``FrontierTrainer``.  It has no torch,
    CUDA, or other accelerator dependency.
    """

    input_dim = 13
    hidden_dim = 24

    def __init__(self, seed: int = 0, lr: float = 0.015):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.04, (self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0.0, 0.025, (2, self.hidden_dim))
        self.b2 = np.zeros(2)
        self.lr = float(lr)
        self.std = np.array([0.05, 0.12])
        self.output_scale = np.array([0.30, 0.90])
        self.updates = 0

    @staticmethod
    def _yaw(orientation) -> float:
        x, y = float(orientation.x), float(orientation.y)
        z, w = float(orientation.z), float(orientation.w)
        return math.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def features_and_prior(self, ranges: np.ndarray, position,
                           orientation, goal_xy=GOAL_XY
                           ) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.nan_to_num(np.asarray(ranges, dtype=float),
                               nan=0.08, posinf=6.0, neginf=0.08)
        ranges = np.clip(ranges, 0.08, 6.0)
        sectors = np.array([part.min() for part in np.array_split(ranges, 10)])
        sector_features = np.clip(sectors / 6.0, 0.0, 1.0)

        dx = float(goal_xy[0]) - float(position.x)
        dy = float(goal_xy[1]) - float(position.y)
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        goal_angle = self._wrap(bearing - self._yaw(orientation))
        features = np.concatenate((
            sector_features,
            [math.sin(goal_angle), math.cos(goal_angle),
             min(distance / 2.0, 1.5)],
        ))

        # Laser angles are ordered right-to-left from -pi to +pi.  The two
        # central sectors cover the forward cone.
        front = float(min(sectors[4], sectors[5]))
        right = float(np.mean(sectors[2:5]))
        left = float(np.mean(sectors[5:8]))
        clearance = np.clip((front - 0.32) / 1.20, 0.0, 1.0)
        base_v = 1.25 * clearance * max(0.08, math.cos(goal_angle) ** 2)
        base_w = 2.80 * goal_angle + 0.55 * np.clip(left - right, -1.5, 1.5)
        if front < 0.48:
            base_w += 0.9 if left >= right else -0.9
        return features, np.array([
            np.clip(base_v, 0.0, 1.8), np.clip(base_w, -1.8, 1.8)])

    def act(self, ranges: np.ndarray, position, orientation,
            rng: np.random.Generator, *, stochastic: bool = True,
            goal_xy=GOAL_XY
            ) -> tuple[np.ndarray, dict]:
        x, prior = self.features_and_prior(
            ranges, position, orientation, goal_xy=goal_xy)
        h = np.tanh(self.W1 @ x + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)
        mean = prior + self.output_scale * out
        epsilon = rng.normal(size=2) if stochastic else np.zeros(2)
        raw = mean + self.std * epsilon
        command = np.array([
            np.clip(raw[0], 0.0, 2.0),
            np.clip(raw[1], -2.0, 2.0),
        ])
        return command, {"x": x, "h": h, "out": out,
                         "epsilon": epsilon, "mean": mean}

    def trajectory_score(self, records: list[dict]) -> dict:
        grads = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
        }
        if not records:
            return grads
        for row in records:
            score_mu = row["epsilon"] / self.std
            d2 = score_mu * self.output_scale * (1.0 - row["out"] ** 2)
            grads["W2"] += np.outer(d2, row["h"])
            grads["b2"] += d2
            d1 = (self.W2.T @ d2) * (1.0 - row["h"] ** 2)
            grads["W1"] += np.outer(d1, row["x"])
            grads["b1"] += d1
        for key in grads:
            grads[key] /= len(records)
        return grads

    def update(self, trajectories, weights) -> None:
        total = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
        }
        for trajectory, weight in zip(trajectories, np.asarray(weights)):
            for key in total:
                total[key] += float(weight) * trajectory["policy_score"][key]
        norm = math.sqrt(sum(float(np.sum(value * value))
                             for value in total.values()))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        for key in total:
            setattr(self, key, getattr(self, key) + self.lr * scale * total[key])
        self.updates += 1

    def state_dict(self) -> dict:
        return {key: getattr(self, key).tolist()
                for key in ("W1", "b1", "W2", "b2")}


class GazeboCourseRuntime:
    """One isolated, headless Gazebo Classic course process."""

    def __init__(self, course: BarnCourse, robot_sdf: Path,
                 stepper_path: Path, runtime_dir: Path, *,
                 domain_id: int, master_port: int,
                 simulator_seed: int, max_step_size: float = 0.005,
                 real_time_update_rate: int = 2000):
        self.course = course
        self.robot_sdf = robot_sdf.resolve()
        self.stepper_path = stepper_path.resolve()
        self.runtime_dir = runtime_dir.resolve()
        self.domain_id = int(domain_id)
        self.master_port = int(master_port)
        self.simulator_seed = int(simulator_seed)
        self.max_step_size = float(max_step_size)
        self.real_time_update_rate = int(real_time_update_rate)
        self.process = None
        self.stepper = None
        self.node = None
        self.latest_scan = None
        self.latest_odom = None
        self.clock_seconds = 0.0
        self.gazebo_clock_seconds = None
        self.sim_iterations = 0
        self.scan_sequence = 0
        self.collided = False
        self.context = None
        self.executor = None
        self.planned_clearance = None
        self.course_path_odom = self._load_course_path()

    def _load_course_path(self) -> np.ndarray:
        points = np.load(self.course.path_path, allow_pickle=False).astype(float)
        radius = 0.075
        world = np.column_stack((
            points[:, 0] * (radius * 2) - radius - 30 * radius * 2,
            points[:, 1] * (radius * 2) + radius + 5.0,
        ))
        # The published Dijkstra path covers the benchmark segment, not the
        # connector from the fixed challenge start.  Build that global plan
        # from the immutable world geometry exactly as Nav2's global planner
        # would, using a circularized Jackal footprint.  The published exit
        # remains the goal inside the obstacle field; beyond y=10 the world is
        # open and two deterministic connector points lead to the final goal.
        failure = None
        for clearance in (0.42, 0.38, 0.36, 0.34, 0.32, 0.30):
            try:
                planned = self._astar_path(
                    np.array(START_XY), world[-1], clearance=clearance)
                self.planned_clearance = clearance
                break
            except RuntimeError as error:
                failure = error
        else:
            raise failure
        sparse = np.vstack((planned, np.array([
            [world[-1, 0], 10.2], [GOAL_XY[0], 10.2], GOAL_XY])))
        dense = [sparse[0]]
        for first, second in zip(sparse[:-1], sparse[1:]):
            distance = float(np.linalg.norm(second - first))
            pieces = max(1, int(math.ceil(distance / 0.15)))
            dense.extend(first + (second - first) * (step / pieces)
                         for step in range(1, pieces + 1))
        return np.asarray(dense)

    def _obstacle_centers(self) -> np.ndarray:
        text = self.course.world_path.read_text()
        pattern = re.compile(
            r"<model name='unit_cylinder_\d+'>.*?"
            r"<pose frame=''>\s*([-0-9.]+)\s+([-0-9.]+)\s+0\.000000",
            re.DOTALL)
        centers = np.array([(float(x), float(y))
                            for x, y in pattern.findall(text)])
        if not len(centers):
            raise ValueError(f"no BARN obstacles parsed from {self.course.world_path}")
        return centers

    def _astar_path(self, start: np.ndarray, goal: np.ndarray,
                    *, resolution: float = 0.075,
                    clearance: float = 0.42) -> np.ndarray:
        xs = np.arange(-4.5, 0.0001, resolution)
        ys = np.arange(0.0, 10.0001, resolution)
        free = np.ones((len(ys), len(xs)), dtype=bool)
        radius = int(math.ceil(clearance / resolution))
        for ox, oy in self._obstacle_centers():
            ix = int(round((ox - xs[0]) / resolution))
            iy = int(round((oy - ys[0]) / resolution))
            x0, x1 = max(0, ix - radius), min(len(xs), ix + radius + 1)
            y0, y1 = max(0, iy - radius), min(len(ys), iy + radius + 1)
            xx, yy = np.meshgrid(xs[x0:x1], ys[y0:y1])
            free[y0:y1, x0:x1] &= ((xx - ox) ** 2 + (yy - oy) ** 2
                                     >= clearance ** 2)

        def index(point):
            return (int(round((point[1] - ys[0]) / resolution)),
                    int(round((point[0] - xs[0]) / resolution)))

        source, target = index(start), index(goal)
        # The challenge endpoints are known-valid oriented Jackal poses.  A
        # circularized footprint is conservative at those two cells, so keep
        # the endpoints traversable while retaining the inflated clearance
        # everywhere else.
        free[source] = True
        free[target] = True
        queue = [(0.0, source)]
        cost = {source: 0.0}
        parent = {}
        moves = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))
        while queue:
            _, current = heapq.heappop(queue)
            if current == target:
                break
            cy, cx = current
            for dy, dx in moves:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < len(ys) and 0 <= nx < len(xs)):
                    continue
                if not free[ny, nx]:
                    continue
                if dy and dx and (not free[cy + dy, cx]
                                  or not free[cy, cx + dx]):
                    continue
                step = math.sqrt(2.0) if dy and dx else 1.0
                candidate = cost[current] + step
                neighbor = (ny, nx)
                if candidate >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = candidate
                parent[neighbor] = current
                heuristic = math.hypot(target[0] - ny, target[1] - nx)
                heapq.heappush(queue, (candidate + heuristic, neighbor))
        if target not in cost:
            raise RuntimeError(f"no global path for {self.course.env_id}")
        cells = [target]
        while cells[-1] != source:
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = np.array([[xs[ix], ys[iy]] for iy, ix in cells])
        path[0], path[-1] = start, goal
        return path

    def _lookahead_goal(self, position, progress: int,
                        distance: float = 0.35) -> tuple[np.ndarray, int]:
        xy = np.array([float(position.x), float(position.y)])
        start = max(0, min(progress, len(self.course_path_odom) - 1))
        stop = min(len(self.course_path_odom), start + 20)
        local = self.course_path_odom[start:stop]
        nearest = start + int(np.argmin(np.linalg.norm(local - xy, axis=1)))
        target = nearest
        accumulated = 0.0
        while target + 1 < len(self.course_path_odom) and accumulated < distance:
            accumulated += float(np.linalg.norm(
                self.course_path_odom[target + 1] - self.course_path_odom[target]))
            target += 1
        return self.course_path_odom[target], nearest

    def _patched_world(self) -> Path:
        source = self.course.world_path.read_text()
        source = source.replace(
            "<max_step_size>0.001</max_step_size>",
            f"<max_step_size>{self.max_step_size}</max_step_size>", 1)
        source = source.replace(
            "<real_time_factor>1</real_time_factor>",
            "<real_time_factor>10</real_time_factor>", 1)
        source = source.replace(
            "<real_time_update_rate>1000</real_time_update_rate>",
            f"<real_time_update_rate>{self.real_time_update_rate}</real_time_update_rate>",
            1)
        source = source.replace(
            "</world>",
            "    <plugin name='barn_exact_step' "
            "filename='libbarn_exact_step.so'/>\n  </world>", 1)
        path = self.runtime_dir / "active_course.world"
        path.write_text(source)
        return path

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({
            "ROS_DOMAIN_ID": str(self.domain_id),
            "GAZEBO_MASTER_URI": f"http://127.0.0.1:{self.master_port}",
            "CUDA_VISIBLE_DEVICES": "",
            "LIBGL_ALWAYS_SOFTWARE": "1",
        })
        plugin_dir = str(self.stepper_path.parent)
        previous_plugins = env.get("GAZEBO_PLUGIN_PATH", "")
        env["GAZEBO_PLUGIN_PATH"] = (
            plugin_dir if not previous_plugins else
            f"{plugin_dir}:{previous_plugins}")
        return env

    def start(self) -> None:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from gazebo_msgs.srv import SpawnEntity
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from gazebo_msgs.msg import ContactsState
        from std_srvs.srv import Empty

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.runtime_dir / "gazebo.log"
        log_handle = log_path.open("a")
        command = [
            "gzserver", "--pause", "--seed", str(self.simulator_seed),
            str(self._patched_world()), "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ]
        self.process = subprocess.Popen(
            command, env=self._environment(), stdout=log_handle,
            stderr=subprocess.STDOUT, start_new_session=True)
        self._log_handle = log_handle

        self.context = rclpy.context.Context()
        rclpy.init(context=self.context, domain_id=self.domain_id)
        self.node = rclpy.create_node(
            f"barn_adapter_{os.getpid()}_{self.course.barn_index}",
            context=self.context)
        self.executor = rclpy.executors.SingleThreadedExecutor(
            context=self.context)
        self.executor.add_node(self.node)
        self.node.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.node.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.node.create_subscription(
            ContactsState, "/bumper_states", self._bumper_cb, 10)
        self.node.create_subscription(
            Clock, "/clock", self._clock_cb, qos_profile_sensor_data)
        self.cmd_publisher = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.spawn_client = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.pause_client = self.node.create_client(Empty, "/pause_physics")
        self.unpause_client = self.node.create_client(Empty, "/unpause_physics")
        self.reset_client = self.node.create_client(Empty, "/reset_simulation")
        if not self.spawn_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("Gazebo /spawn_entity did not become ready")

        request = SpawnEntity.Request()
        request.name = "barn_robot"
        request.xml = self.robot_sdf.read_text()
        request.initial_pose.position.x = START_XY[0]
        request.initial_pose.position.y = START_XY[1]
        request.initial_pose.position.z = 0.09
        request.initial_pose.orientation.z = math.sin(START_YAW / 2.0)
        request.initial_pose.orientation.w = math.cos(START_YAW / 2.0)
        request.reference_frame = "world"
        response = self._call(self.spawn_client, request, timeout=30.0)
        if not response.success:
            raise RuntimeError(f"robot spawn failed: {response.status_message}")
        if not self.stepper_path.is_file():
            raise FileNotFoundError(
                f"Gazebo stepper missing: {self.stepper_path}; run "
                "icra2027/build_gazebo_stepper.sh")
        self.stepper = subprocess.Popen(
            [str(self.stepper_path)], env=self._environment(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_handle, text=True, bufsize=1,
            start_new_session=True)
        deadline = time.monotonic() + 20.0
        while (self.latest_scan is None or self.latest_odom is None):
            self._advance(10)
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for Gazebo lidar/odometry")

    def _spin(self, timeout: float = 0.01) -> None:
        self.executor.spin_once(timeout_sec=timeout)
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"gzserver exited with {self.process.returncode}")

    def _call(self, client, request, *, timeout: float):
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            self._spin(0.02)
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out calling {client.srv_name}")
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def _scan_cb(self, message) -> None:
        self.latest_scan = message
        self.scan_sequence += 1

    def _odom_cb(self, message) -> None:
        self.latest_odom = message

    def _bumper_cb(self, message) -> None:
        # The chassis can brush the ground plane as the suspension settles;
        # only contact with a frozen BARN obstacle is a navigation collision.
        for state in message.states:
            names = f"{state.collision1_name} {state.collision2_name}"
            if "unit_cylinder_" in names:
                self.collided = True
                break

    def _clock_cb(self, message) -> None:
        self.gazebo_clock_seconds = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1e-9)

    def _publish(self, linear: float, angular: float) -> None:
        from geometry_msgs.msg import Twist
        message = Twist()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self.cmd_publisher.publish(message)
        # Gazebo remains paused here; allow its ROS executor to receive the
        # command before the exact multi-step message advances physics.
        time.sleep(0.002)

    def _advance(self, iterations: int) -> None:
        if self.stepper is None or self.stepper.poll() is not None:
            raise RuntimeError("Gazebo stepper is not running")
        self.stepper.stdin.write(f"{int(iterations)}\n")
        self.stepper.stdin.flush()
        receipt = self.stepper.stdout.readline().strip()
        if receipt != f"ok {int(iterations)}":
            raise RuntimeError(f"invalid Gazebo step receipt: {receipt!r}")
        # The helper acknowledges only after Gazebo's own WorldStatistics
        # iteration counter reaches the requested target.  Derive campaign
        # time from that exact count: Gazebo Classic does not reliably emit
        # ROS /clock while it remains paused between multi-step commands.
        self.sim_iterations += int(iterations)
        self.clock_seconds = self.sim_iterations * self.max_step_size
        for _ in range(5):
            self._spin(0.002)

    def run_episode(self, policy: LidarVelocityPolicy, *, episode_seed: int,
                    timeout_seconds: float = 25.0,
                    stochastic: bool = True) -> dict:
        from std_srvs.srv import Empty

        self._publish(0.0, 0.0)
        self._call(self.reset_client, Empty.Request(), timeout=10.0)
        self.clock_seconds = 0.0
        self.gazebo_clock_seconds = None
        self.sim_iterations = 0
        self.latest_scan = None
        self.latest_odom = None
        self.collided = False
        deadline = time.monotonic() + max(30.0, timeout_seconds * 2.0)
        iterations_run = 0
        while self.latest_scan is None or self.latest_odom is None:
            self._advance(10)
            iterations_run += 10
            if time.monotonic() > deadline:
                raise RuntimeError("episode warmup timed out")
        start_clock = self.clock_seconds
        rng = np.random.default_rng(int(episode_seed))
        records, positions, commands = [], [], []
        path_progress = 0
        success = False
        status = "timeout"

        while True:
            pose = self.latest_odom.pose.pose
            position = pose.position
            distance = math.hypot(
                GOAL_XY[0] - float(position.x),
                GOAL_XY[1] - float(position.y))
            elapsed = max(0.0, self.clock_seconds - start_clock)
            positions.append([float(position.x), float(position.y), elapsed])
            if distance <= 1.0:
                success, status = True, "succeeded"
                break
            if self.collided:
                status = "collided"
                break
            if elapsed >= timeout_seconds:
                status = "timeout"
                break
            subgoal, path_progress = self._lookahead_goal(
                position, path_progress)
            command, record = policy.act(
                np.asarray(self.latest_scan.ranges), position,
                pose.orientation, rng, stochastic=stochastic,
                goal_xy=subgoal)
            records.append(record)
            commands.append([float(command[0]), float(command[1]), elapsed])
            self._publish(command[0], command[1])
            previous_sequence = self.scan_sequence
            self._advance(10)
            iterations_run += 10
            while self.scan_sequence == previous_sequence:
                self._spin(0.001)
                if time.monotonic() > deadline:
                    raise RuntimeError("episode exceeded wall-clock safety limit")

        self._publish(0.0, 0.0)
        elapsed = max(0.0, self.clock_seconds - start_clock)
        return {
            "success": success,
            "status": status,
            "sim_seconds": elapsed,
            "sim_steps": iterations_run,
            "positions": positions,
            "commands": commands,
            "policy_score": policy.trajectory_score(records),
            "planned_clearance_m": self.planned_clearance,
        }

    def close(self) -> None:
        if self.stepper is not None:
            if self.stepper.stdin is not None:
                self.stepper.stdin.close()
            try:
                self.stepper.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.stepper.terminate()
                self.stepper.wait(timeout=5.0)
            self.stepper = None
        if self.node is not None:
            if self.executor is not None:
                self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.node = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.context is not None:
            self.context.shutdown()
            self.context = None
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
            self.process = None
        if hasattr(self, "_log_handle"):
            self._log_handle.close()

    def __enter__(self):
        try:
            self.start()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class BarnGazeboSpace:
    """NavigationSpace-compatible training and held-out evaluation adapter."""

    def __init__(self, courses: list[BarnCourse], robot_sdf: Path,
                 runtime_root: Path, *, seed: int, n_strata: int = 10,
                 domain_id: int = 80, master_port: int = 11500,
                 episode_timeout: float = 25.0,
                 max_step_size: float = 0.005,
                 real_time_update_rate: int = 2000,
                 policy: LidarVelocityPolicy | None = None,
                 stepper_path: Path | None = None):
        if len(courses) < n_strata:
            raise ValueError("need at least one course per stratum")
        self.courses = sorted(courses, key=lambda c: (c.difficulty, c.env_id))
        self.strata = [list(chunk) for chunk in np.array_split(
            np.array(self.courses, dtype=object), n_strata)]
        self.robot_sdf = robot_sdf
        self.stepper_path = (
            Path(stepper_path).resolve() if stepper_path is not None else
            robot_sdf.resolve().parent.parent / "bin/gazebo_stepper")
        self.runtime_root = runtime_root
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.domain_id = int(domain_id)
        self.master_port = int(master_port)
        self.episode_timeout = float(episode_timeout)
        self.max_step_size = float(max_step_size)
        self.real_time_update_rate = int(real_time_update_rate)
        self.policy = policy or LidarVelocityPolicy(seed=seed)
        self.training_episodes = 0
        self.training_sim_steps = 0
        self.course_launches = 0
        self._episode_counter = 0

    @property
    def n_tasks(self) -> int:
        return len(self.strata)

    def _run_course(self, course: BarnCourse, n: int, *, seed: int,
                    training: bool) -> list[dict]:
        launch_dir = self.runtime_root / (
            f"launch_{self.course_launches:06d}_{course.env_id}")
        runtime = GazeboCourseRuntime(
            course, self.robot_sdf, self.stepper_path, launch_dir,
            domain_id=self.domain_id, master_port=self.master_port,
            simulator_seed=seed, max_step_size=self.max_step_size,
            real_time_update_rate=self.real_time_update_rate)
        self.course_launches += 1
        episodes = []
        with runtime:
            for episode in range(n):
                result = runtime.run_episode(
                    self.policy, episode_seed=int(np.random.SeedSequence(
                        [int(seed), int(episode)]).generate_state(1)[0]),
                    timeout_seconds=self.episode_timeout,
                    stochastic=True)
                result.update({"course_id": course.env_id,
                               "difficulty": course.difficulty})
                episodes.append(result)
        return episodes

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        if not 0 <= task_id < self.n_tasks:
            raise IndexError(task_id)
        course = self.strata[task_id][int(self.rng.integers(len(self.strata[task_id])))]
        launch_seed = int(self.rng.integers(0, 2**31 - 1))
        episodes = self._run_course(
            course, n_rollouts, seed=launch_seed, training=True)
        self.training_episodes += len(episodes)
        self.training_sim_steps += sum(row["sim_steps"] for row in episodes)
        self._episode_counter += len(episodes)
        trajectories = [{
            "course_id": row["course_id"],
            "positions": row["positions"],
            "commands": row["commands"],
            "policy_score": row["policy_score"],
        } for row in episodes]
        infos = [{key: row[key] for key in (
            "course_id", "difficulty", "status", "sim_seconds", "sim_steps",
            "planned_clearance_m")}
                 for row in episodes]
        rewards = np.array([float(row["success"]) for row in episodes])
        return GroupResult(task_id, rewards, trajectories, infos)

    def relabel(self, group: GroupResult):
        return None

    def update(self, task_id: int, trajectories, weights) -> None:
        self.policy.update(trajectories, weights)

    def evaluate_course(self, course: BarnCourse, n: int, *, seed: int
                        ) -> tuple[int, int, list[dict]]:
        episodes = self._run_course(course, n, seed=seed, training=False)
        return (sum(int(row["success"]) for row in episodes),
                sum(int(row["sim_steps"]) for row in episodes), episodes)
