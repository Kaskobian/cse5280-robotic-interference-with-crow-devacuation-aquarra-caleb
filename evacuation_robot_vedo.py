import argparse
import os
import numpy as np

from vedo import (
    Plotter, Axes, Sphere, Box, Points, Text2D, load,
    DashedLine, settings, Cylinder
)

from robot_arm_vedo import RobotArm
from fetch_robot_parts import ensure_robot_parts


class InterferenceController:
    def __init__(self, exits, room_half=420.0, exit_radius=170.0, predict_horizon=18.0, cluster_k=2, smoothing=0.18):
        self.exits = np.array(exits, dtype=float)
        self.room_half = room_half
        self.exit_radius = exit_radius
        self.predict_horizon = predict_horizon
        self.cluster_k = cluster_k
        self.smoothing = smoothing
        self.smoothed_prediction = None
        self.last_clusters = None
        self.last_prediction = None
        self.last_target_exit = None

    def particles_near_exits(self, positions):
        d = np.linalg.norm(positions[:, None, :] - self.exits[None, :, :], axis=2)
        return np.where(np.min(d, axis=1) < self.exit_radius)[0]

    def kmeans_simple(self, points, k, max_iter=12):
        if len(points) == 0:
            return np.array([], dtype=int), np.empty((0, 2))
        if len(points) <= k:
            return np.arange(len(points)), points.copy()
        rng = np.random.default_rng(0)
        centers = points[rng.choice(len(points), size=k, replace=False)].copy()
        for _ in range(max_iter):
            d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                pts = points[labels == j]
                if len(pts) > 0:
                    new_centers[j] = pts.mean(axis=0)
            if np.allclose(new_centers, centers, atol=1e-2):
                break
            centers = new_centers
        return labels, centers

    def choose_cluster(self, points, velocities, labels, centers):
        best = None
        best_score = -1e18
        for cid in range(len(centers)):
            pts = points[labels == cid]
            vels = velocities[labels == cid]
            if len(pts) == 0:
                continue
            center = pts.mean(axis=0)
            avg_vel = vels.mean(axis=0)

            exit_dists = np.linalg.norm(self.exits - center[None, :], axis=1)
            exit_idx = int(np.argmin(exit_dists))
            target_exit = self.exits[exit_idx]
            to_exit = target_exit - center
            to_exit_norm = np.linalg.norm(to_exit)
            if to_exit_norm < 1e-8:
                continue
            to_exit_dir = to_exit / to_exit_norm
            speed_toward_exit = float(np.dot(avg_vel, to_exit_dir))

            score = (
                3.0 * len(pts)
                + 120.0 / (exit_dists[exit_idx] + 1.0)
                + 12.0 * max(speed_toward_exit, 0.0)
            )
            if score > best_score:
                best_score = score
                best = (cid, center, avg_vel, exit_idx)
        return best

    def update(self, positions, velocities):
        idx = self.particles_near_exits(positions)
        if len(idx) < 2:
            self.last_clusters = None
            return self.last_prediction

        pts = positions[idx]
        vels = velocities[idx]
        k = min(self.cluster_k, len(pts))
        labels, centers = self.kmeans_simple(pts, k)
        chosen = self.choose_cluster(pts, vels, labels, centers)
        if chosen is None:
            self.last_clusters = None
            return self.last_prediction

        cid, center, avg_vel, exit_idx = chosen
        target_exit = self.exits[exit_idx]
        raw_pred = center + self.predict_horizon * avg_vel

        exit_interior_bias = target_exit.copy()
        exit_interior_bias[1] += -70.0 if target_exit[1] > 0 else 70.0
        raw_pred = 0.65 * raw_pred + 0.35 * exit_interior_bias
        raw_pred = np.clip(raw_pred, -self.room_half + 35, self.room_half - 35)

        if self.smoothed_prediction is None:
            self.smoothed_prediction = raw_pred.copy()
        else:
            self.smoothed_prediction = (1.0 - self.smoothing) * self.smoothed_prediction + self.smoothing * raw_pred

        self.last_target_exit = target_exit.copy()
        self.last_clusters = {
            "points": pts,
            "labels": labels,
            "centers": centers,
            "chosen": cid,
            "chosen_center": center,
            "raw_prediction": raw_pred,
            "target_exit": target_exit.copy(),
        }
        self.last_prediction = self.smoothed_prediction.copy()
        return self.last_prediction


class CrowdSim:
    def __init__(self, n=140, room_half=420.0):
        self.room_half = room_half
        self.exit_width = 80.0
        self.positions = self._spawn_positions(n)
        self.velocities = np.zeros_like(self.positions)
        self.exits = np.array([[0.0, room_half], [0.0, -room_half]], dtype=float)
        self.max_speed = 16.0
        self.goal_strength = 26.0

    def _spawn_positions(self, n):
        rng = np.random.default_rng(7)
        pts = []
        while len(pts) < n:
            p = rng.uniform(-260, 260, size=2)
            if np.linalg.norm(p) > 50:
                pts.append(p)
        return np.array(pts, dtype=float)

    def goal_forces(self):
        d = np.linalg.norm(self.positions[:, None, :] - self.exits[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        targets = self.exits[nearest]
        diff = targets - self.positions
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        return self.goal_strength * diff / norm

    def wall_forces(self):
        h = self.room_half
        margin = 28.0
        gain = 100.0
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        in_top_door = (np.abs(x) < self.exit_width / 2) & (y > h - 45)
        in_bot_door = (np.abs(x) < self.exit_width / 2) & (y < -h + 45)

        f = np.zeros_like(self.positions)
        left_mask = x < -h + margin
        f[left_mask, 0] += gain / np.maximum(x[left_mask] + h, 1.0)

        right_mask = x > h - margin
        f[right_mask, 0] -= gain / np.maximum(h - x[right_mask], 1.0)

        bot_mask = (y < -h + margin) & (~in_bot_door)
        f[bot_mask, 1] += gain / np.maximum(y[bot_mask] + h, 1.0)

        top_mask = (y > h - margin) & (~in_top_door)
        f[top_mask, 1] -= gain / np.maximum(h - y[top_mask], 1.0)
        return f

    def social_forces(self):
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        mask = dist < 28.0
        if not np.any(mask):
            return np.zeros_like(self.positions)
        dir_vec = np.zeros_like(diff)
        dir_vec[mask] = diff[mask] / dist[mask][:, None]
        strength = np.zeros_like(dist)
        strength[mask] = (28.0 - dist[mask]) * 1.1
        return np.sum(dir_vec * strength[:, :, None], axis=1)

    def robot_force(self, robot_xy, target_exit=None, influence_radius=100.0):
        diff = self.positions - robot_xy[None, :]
        dist = np.linalg.norm(diff, axis=1)
        safe_dist = np.maximum(dist, 1e-6)
        inside = safe_dist < influence_radius

        force = np.zeros_like(self.positions)

        radial = np.zeros_like(self.positions)
        radial[inside] = diff[inside] / safe_dist[inside, None]
        radial_strength = np.zeros(len(self.positions))
        radial_strength[inside] = 280.0 * ((influence_radius - safe_dist[inside]) / influence_radius) ** 1.6
        force += radial * radial_strength[:, None]

        core_radius = 48.0
        core = safe_dist < core_radius
        if np.any(core):
            core_strength = 620.0 * ((core_radius - safe_dist[core]) / core_radius + 0.2)
            force[core] += radial[core] * core_strength[:, None]

        if target_exit is not None:
            to_exit = target_exit - robot_xy
            n = np.linalg.norm(to_exit)
            if n > 1e-6:
                exit_dir = to_exit / n
                lateral = np.array([-exit_dir[1], exit_dir[0]])
                frontness = np.abs((self.positions - robot_xy[None, :]) @ exit_dir)
                lane = inside & (frontness < 70.0)
                side_sign = np.sign((self.positions[lane] - robot_xy[None, :]) @ lateral)
                side_sign[side_sign == 0] = 1.0
                force[lane] += side_sign[:, None] * lateral[None, :] * 120.0

        return force

    def step(self, robot_xy, target_exit=None, dt=0.05):
        f = self.goal_forces()
        f += self.wall_forces()
        f += self.social_forces()
        f += self.robot_force(robot_xy, target_exit=target_exit)

        self.velocities = 0.85 * self.velocities + dt * f
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speed[:, 0] > self.max_speed
        if np.any(mask):
            self.velocities[mask] *= self.max_speed / speed[mask]

        self.positions = self.positions + dt * self.velocities

        h = self.room_half
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        vx = self.velocities[:, 0]
        vy = self.velocities[:, 1]

        top_gap = np.abs(x) < self.exit_width / 2
        bot_gap = np.abs(x) < self.exit_width / 2

        left_hit = x < -h
        x[left_hit] = -h
        vx[left_hit] *= -0.1

        right_hit = x > h
        x[right_hit] = h
        vx[right_hit] *= -0.1

        top_hit = (y > h) & (~top_gap)
        y[top_hit] = h
        vy[top_hit] *= -0.1

        bot_hit = (y < -h) & (~bot_gap)
        y[bot_hit] = -h
        vy[bot_hit] *= -0.1

        self.positions[:, 0] = x
        self.positions[:, 1] = y
        self.velocities[:, 0] = vx
        self.velocities[:, 1] = vy


class Scene:
    def __init__(self, offscreen=False, size=(1400, 900)):
        settings.default_backend = "vtk"
        self.plt = Plotter(size=size, bg="white", offscreen=offscreen, interactive=False)
        self.static_actors = []
        self.robot_meshes = None
        self.ee_sphere = None
        self.ee_zone = None
        self.crowd_actor = None
        self.cluster_actors = []
        self.center_actors = []
        self.pred_actor = None
        self.raw_pred_actor = None
        self.link_actor = None
        self.info = Text2D("", pos="top-right", s=0.8)
        self.exit_target_actor = None

    def _set_camera(self):
        cam = self.plt.camera
        cam.SetPosition(920, -1180, 760)
        cam.SetFocalPoint(0, 0, 120)
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(34)
        cam.SetClippingRange(1, 5000)

    def build_static_scene(self, room_half, exits):
        floor = Box(pos=(0, 0, -6), length=2 * room_half + 80, width=2 * room_half + 80, height=12).c("linen")
        wall_h = 120
        t = 10
        walls = [
            Box(pos=(-room_half - t / 2, 0, wall_h / 2), length=t, width=2 * room_half, height=wall_h).c("gray7"),
            Box(pos=(room_half + t / 2, 0, wall_h / 2), length=t, width=2 * room_half, height=wall_h).c("gray7"),
            Box(pos=(-room_half / 2 - 60, room_half + t / 2, wall_h / 2), length=room_half - 120, width=t, height=wall_h).c("gray7"),
            Box(pos=(room_half / 2 + 60, room_half + t / 2, wall_h / 2), length=room_half - 120, width=t, height=wall_h).c("gray7"),
            Box(pos=(-room_half / 2 - 60, -room_half - t / 2, wall_h / 2), length=room_half - 120, width=t, height=wall_h).c("gray7"),
            Box(pos=(room_half / 2 + 60, -room_half - t / 2, wall_h / 2), length=room_half - 120, width=t, height=wall_h).c("gray7"),
        ]
        exit_actors = [Box(pos=(ex[0], ex[1], 2), length=80, width=20, height=6).c("green4") for ex in exits]
        axes = Axes(xrange=(-600, 600), yrange=(-600, 600), zrange=(0, 650))
        title = Text2D("Evacuation Interference Robot", pos="top-left", s=1.0)
        self.static_actors = [floor, *walls, *exit_actors, axes, title, self.info]
        self.plt.show(*self.static_actors, resetcam=True)
        self._set_camera()

    def _update_crowd(self, crowd):
        pts3 = np.c_[crowd.positions, np.full(len(crowd.positions), 8.0)]
        if self.crowd_actor is None:
            self.crowd_actor = Points(pts3, r=7).c("dodgerblue")
            self.plt += self.crowd_actor
        else:
            self.crowd_actor.vertices = pts3

    def initialize_dynamic(self, crowd, robot, phi):
        self.robot_meshes = robot.update_pose(phi)
        for a in self.robot_meshes:
            self.plt += a
        _, _, _, _, _, ee = robot.forward_kinematics(phi)
        self.ee_sphere = Sphere(ee, r=18, c="red5")
        self.ee_zone = Cylinder(pos=(ee[0], ee[1], 15), r=100, height=30, axis=(0, 0, 1), c="red", alpha=0.16)
        self.plt += self.ee_sphere
        self.plt += self.ee_zone
        self._update_crowd(crowd)
        self.pred_actor = Sphere((0, 0, 30), r=14, c="magenta").alpha(0.0)
        self.raw_pred_actor = Sphere((0, 0, 18), r=9, c="pink").alpha(0.0)
        self.exit_target_actor = Sphere((0, 0, 20), r=10, c="green").alpha(0.0)
        self.link_actor = DashedLine((0, 0, 0), (0, 0, 0), spacing=0.15, lw=2).c("magenta").alpha(0.0)
        for a in [self.pred_actor, self.raw_pred_actor, self.exit_target_actor, self.link_actor]:
            self.plt += a

    def update_dynamic(self, crowd, controller, robot, phi, step_idx):
        robot.update_pose(phi)
        _, _, _, _, _, ee = robot.forward_kinematics(phi)
        self.ee_sphere.pos(ee)
        self.ee_zone.pos((ee[0], ee[1], 15))
        self._update_crowd(crowd)

        for a in self.cluster_actors + self.center_actors:
            try:
                self.plt.remove(a)
            except Exception:
                pass
        self.cluster_actors = []
        self.center_actors = []

        if controller.last_clusters is not None:
            points = controller.last_clusters["points"]
            labels = controller.last_clusters["labels"]
            centers = controller.last_clusters["centers"]
            palette = ["orange", "purple", "gold", "cyan"]
            for cid in np.unique(labels):
                pts = points[labels == cid]
                if len(pts):
                    actor = Points(np.c_[pts, np.full(len(pts), 12.0)], r=9).c(palette[int(cid) % len(palette)])
                    self.cluster_actors.append(actor)
                    self.plt += actor
            for c in centers:
                s = Sphere((c[0], c[1], 14), r=8, c="black")
                self.center_actors.append(s)
                self.plt += s

            raw = controller.last_clusters.get("raw_prediction")
            if raw is not None:
                self.raw_pred_actor.pos((raw[0], raw[1], 18)).alpha(0.45)
            target_exit = controller.last_clusters.get("target_exit")
            if target_exit is not None:
                self.exit_target_actor.pos((target_exit[0], target_exit[1], 20)).alpha(0.75)

        if controller.last_prediction is not None:
            pred = controller.last_prediction
            self.pred_actor.pos((pred[0], pred[1], 30)).alpha(1.0)
            try:
                self.plt.remove(self.link_actor)
            except Exception:
                pass
            self.link_actor = DashedLine((ee[0], ee[1], ee[2]), (pred[0], pred[1], 30), spacing=0.15, lw=2).c("magenta").alpha(0.9)
            self.plt += self.link_actor
        else:
            self.pred_actor.alpha(0.0)
            self.raw_pred_actor.alpha(0.0)
            self.exit_target_actor.alpha(0.0)

        self.info.text(f"step={step_idx}  particles={len(crowd.positions)}")
        self.plt.render()

    def screenshot(self, save_path):
        self.plt.screenshot(save_path)


def build_robot():
    ensure_robot_parts("robot")
    robot_dir = "robot"
    base = load(os.path.join(robot_dir, "Base.stl")).color("blue5")
    base_rot = load(os.path.join(robot_dir, "BaseRot.stl")).color("lightblue")
    humerus = load(os.path.join(robot_dir, "Humerus.stl")).color("gray5")
    radius = load(os.path.join(robot_dir, "Radius.stl")).color("red5")
    parts = [base, base_rot, humerus, radius]

    base_h = 105
    base_rot_h = 81
    humerus_h = 217
    radius_h = 416
    circle_radius = 550
    arm_location = np.array([[circle_radius], [0.0], [base_h]], dtype=float)
    return RobotArm([base_rot_h, humerus_h, radius_h, 0], parts, arm_location)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--particles", type=int, default=110)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--frames-dir", default="frames_vedo")
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    crowd = CrowdSim(n=args.particles, room_half=420.0)
    controller = InterferenceController(crowd.exits, room_half=crowd.room_half, exit_radius=180.0, predict_horizon=20.0, cluster_k=2, smoothing=0.16)
    robot = build_robot()
    phi = np.array([0.0, -8.0, 22.0, 0.0], dtype=float)

    scene = Scene(offscreen=args.offscreen)
    scene.build_static_scene(crowd.room_half, crowd.exits)

    if args.save_frames:
        os.makedirs(args.frames_dir, exist_ok=True)

    smoothed_target = np.array([0.0, 250.0, 155.0], dtype=float)
    robot.update_pose(phi)
    scene.initialize_dynamic(crowd, robot, phi)

    for step_idx in range(args.steps):
        pred = controller.update(crowd.positions, crowd.velocities)
        if pred is not None:
            desired = np.array([pred[0], pred[1], 88.0], dtype=float)
            smoothed_target = 0.86 * smoothed_target + 0.14 * desired
            robot.target = smoothed_target

        phi = robot.ik_step_toward_target(phi, step_scale=0.08)
        _, _, _, _, _, ee = robot.forward_kinematics(phi)

        crowd.step(ee[:2], target_exit=controller.last_target_exit, dt=args.dt)
        scene.update_dynamic(crowd, controller, robot, phi, step_idx)

        if args.save_frames:
            save_path = os.path.join(args.frames_dir, f"frame_{step_idx:04d}.png")
            scene.screenshot(save_path)

    if not args.offscreen:
        scene.plt.interactive().close()
    else:
        scene.plt.close()


if __name__ == "__main__":
    main()
