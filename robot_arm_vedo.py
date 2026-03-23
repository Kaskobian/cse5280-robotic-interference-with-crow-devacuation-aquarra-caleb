import numpy as np
from vedo import Arrow, Sphere, LinearTransform


class RobotArm:
    """
    STL-backed robot arm with damped least-squares IK and persistent meshes.
    """

    def __init__(self, part_lengths, parts, arm_location):
        self.arm_location = np.array(arm_location, dtype=float).reshape(3, 1)
        self.L1, self.L2, self.L3, self.L4 = part_lengths
        self.source_parts = parts

        self.delta_phi = 0.5
        self.target = np.array([0.0, 100.0, 170.0], dtype=float)
        self.target_tolerance = 14.0
        self.target_lambda = 18.0
        self.max_joint_step = 1.2
        self.meshes = None
        self.transforms = None
        self.initialize_meshes()

    def initialize_meshes(self):
        base = self.source_parts[0].clone()
        part1 = self.source_parts[1].clone()
        part2 = self.source_parts[2].clone()
        part3 = self.source_parts[3].clone()
        part4 = self.create_coordinate_frame_mesh()
        self.meshes = [base, part1, part2, part3, part4]
        self.transforms = [None] * len(self.meshes)

    def rotation_matrix(self, theta, axis_name):
        c = np.cos(np.deg2rad(theta))
        s = np.sin(np.deg2rad(theta))
        if axis_name == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis_name == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        if axis_name == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        raise ValueError(f"Unknown axis_name: {axis_name}")

    def create_coordinate_frame_mesh(self):
        shaft_radius = 0.05
        head_radius = 0.10
        unit = 30
        x_axis = Arrow((0, 0, 0), (unit, 0, 0), shaft_radius=shaft_radius, head_radius=head_radius, c="red")
        y_axis = Arrow((0, 0, 0), (0, unit, 0), shaft_radius=shaft_radius, head_radius=head_radius, c="green")
        z_axis = Arrow((0, 0, 0), (0, 0, unit), shaft_radius=shaft_radius, head_radius=head_radius, c="blue")
        origin = Sphere((0, 0, 0), r=0.10 * unit, c="black")
        return x_axis + y_axis + z_axis + origin

    def get_local_frame_matrix(self, r_ij, t_ij):
        return np.block([[r_ij, t_ij], [np.zeros((1, 3)), np.array([[1]])]])

    def forward_kinematics(self, phi):
        radius = 0.4
        phi1, phi2, phi3, phi4 = phi

        r_00 = self.rotation_matrix(0, "z")
        t_00 = self.arm_location.copy()
        t_00[-1] = 0
        t_01 = self.arm_location.copy()

        t_12 = np.array([[0.0], [0.0], [self.L1 + 2 * radius]])
        t_23 = np.array([[0.0], [0.0], [self.L2 + 2 * radius]])
        t_34 = np.array([[-28.4], [0.0], [self.L3 + radius]])

        t00 = self.get_local_frame_matrix(r_00, t_00)
        t01 = self.get_local_frame_matrix(self.rotation_matrix(phi1, "z"), t_01)
        t12 = self.get_local_frame_matrix(self.rotation_matrix(phi2, "y"), t_12)
        t23 = self.get_local_frame_matrix(self.rotation_matrix(phi3, "y"), t_23)
        t34 = self.get_local_frame_matrix(self.rotation_matrix(phi4, "y"), t_34)

        t02 = t01 @ t12
        t03 = t01 @ t12 @ t23
        t04 = t01 @ t12 @ t23 @ t34
        e = t04[:3, -1]
        return t00, t01, t02, t03, t04, e

    def get_pose_transforms(self, phi):
        t00, t01, t02, t03, t04, _ = self.forward_kinematics(phi)
        return [t00, t01, t02, t03, t04]

    def update_pose(self, phi):
        transforms = self.get_pose_transforms(phi)
        if self.meshes is None:
            self.initialize_meshes()
        for i, (mesh, t) in enumerate(zip(self.meshes, transforms)):
            lt = LinearTransform(t)
            try:
                if self.transforms[i] is None:
                    mesh.apply_transform(lt)
                else:
                    mesh.apply_transform(self.transforms[i].compute_inverse())
                    mesh.apply_transform(lt)
            except Exception:
                mesh.apply_transform(lt)
            self.transforms[i] = lt
        return self.meshes

    def jacobian_matrix(self, phi):
        step = self.delta_phi
        _, _, _, _, _, e = self.forward_kinematics(phi)
        cols = []
        for i in range(3):
            dphi = np.zeros(4)
            dphi[i] = step
            _, _, _, _, _, e2 = self.forward_kinematics(phi + dphi)
            cols.append(((e2 - e) / step).reshape(3, 1))
        return np.concatenate(cols, axis=1)

    def ik_step_toward_target(self, phi, step_scale=0.08):
        phi = np.array(phi, dtype=float).copy()
        _, _, _, _, _, e = self.forward_kinematics(phi)
        err = np.array(self.target, dtype=float) - e

        if np.linalg.norm(err) < self.target_tolerance:
            return phi

        j = self.jacobian_matrix(phi)
        lam = self.target_lambda
        a = j @ j.T + (lam ** 2) * np.eye(3)
        dls = j.T @ np.linalg.solve(a, step_scale * err)
        phi_delta = np.append(dls, [0.0])
        phi_delta = np.clip(phi_delta, -self.max_joint_step, self.max_joint_step)
        phi_new = phi + phi_delta

        phi_new[0] = np.clip(phi_new[0], -175, 175)
        phi_new[1] = np.clip(phi_new[1], -85, 85)
        phi_new[2] = np.clip(phi_new[2], -115, 95)
        phi_new[3] = 0.0
        return phi_new
