"""G1-D FK, bounded dual-arm IK, and gravity, adapted from robotskills drawing.

Full G1-D URDFs with continuous wheel joints and configurable waist names work
without loading meshes. Absolute poses are expressed in reference_frame.
"""

from pathlib import Path
import threading

import numpy as np

from ..utils.transforms import vector

ARM_JOINT_NAMES = tuple(
    f"{side}_{name}_joint" for side in ("left", "right")
    for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")
)


class G1DKinematics:
    def __init__(self, config):
        import pinocchio as pin

        self._solvers = {}
        self.pin = pin
        self.settings = config["model"]
        self.frame_id = self.settings["reference_frame"]
        self.model = pin.buildModelFromUrdf(str(Path(self.settings["urdf"])))
        self.data = self.model.createData()
        self._lock = threading.RLock()
        self._base = pin.neutral(self.model)
        self._qi, self._vi = [], []
        for name in ARM_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid == 0 or jid >= self.model.njoints:
                raise ValueError(f"missing arm joint: {name}")
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"arm joint must be scalar: {name}")
            self._qi.append(joint.idx_q)
            self._vi.append(joint.idx_v)
        self.lower = np.asarray(self.model.lowerPositionLimit)[self._qi].copy()
        self.upper = np.asarray(self.model.upperPositionLimit)[self._qi].copy()
        if not np.isfinite([self.lower, self.upper]).all():
            raise ValueError("arm limits must be finite")
        for name, value in self.settings.get("reference_positions", {}).items():
            self._set_nonarm(name, value)
        self._waist_map = self.settings.get("waist_joint_map", {})
        for name, index in self._waist_map.items():
            if type(index) is not int or index not in (0, 2):
                raise ValueError("waist_joint_map indices must select yaw=0 or pitch=2")
            self._set_nonarm(name, 0.0)
        self._reference = self._frame(self.frame_id)
        self._frames = {side: self._frame(self.settings[f"{side}_frame"])
                        for side in ("left", "right")}
        self._tools = {}
        for side in self._frames:
            tool = np.asarray(self.settings[f"{side}_tool_transform"], dtype=float)
            if (tool.shape != (4, 4) or not np.isfinite(tool).all() or
                    not np.allclose(tool[3], [0, 0, 0, 1]) or
                    not np.allclose(tool[:3, :3].T @ tool[:3, :3], np.eye(3)) or
                    not np.isclose(np.linalg.det(tool[:3, :3]), 1.0)):
                raise ValueError(f"invalid {side} tool transform")
            self._tools[side] = tool.copy()
        self.position_tolerance = float(self.settings.get("position_tolerance_m", 0.003))
        self.rotation_tolerance = float(self.settings.get("rotation_tolerance_rad", 0.03))
        if not (np.isfinite([self.position_tolerance, self.rotation_tolerance]).all()
                and self.position_tolerance > 0 and self.rotation_tolerance > 0):
            raise ValueError("IK tolerances must be finite and positive")
        # Relative-to-base targets assume the reference does not move with an arm.
        jid = self.model.frames[self._reference].parentJoint
        while jid:
            if self.model.names[jid] in ARM_JOINT_NAMES:
                raise ValueError("reference_frame cannot be attached to an arm")
            jid = self.model.parents[jid]

    def _frame(self, name):
        index = self.model.getFrameId(name)
        if index >= self.model.nframes:
            raise ValueError(f"unknown URDF frame: {name}")
        return index

    def _set_nonarm(self, name, value):
        jid = self.model.getJointId(name)
        if jid == 0 or jid >= self.model.njoints or name in ARM_JOINT_NAMES:
            raise ValueError(f"invalid non-arm joint: {name}")
        joint = self.model.joints[jid]
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"non-finite reference: {name}")
        if joint.nq == 1 and joint.nv == 1:
            if not (self.model.lowerPositionLimit[joint.idx_q] <= value <=
                    self.model.upperPositionLimit[joint.idx_q]):
                raise ValueError(f"non-arm position outside limits: {name}")
            self._base[joint.idx_q] = value
        elif joint.nq == 2 and joint.nv == 1:
            self._base[joint.idx_q:joint.idx_q+2] = [np.cos(value), np.sin(value)]
        else:
            raise ValueError(f"unsupported non-arm joint: {name}")

    def update_state(self, state):
        waist = vector(state["waist_q3"], 3, "waist")
        with self._lock:
            for name, index in self._waist_map.items():
                self._set_nonarm(name, waist[index])

    def check_joints(self, q):
        q = vector(q, 14, "q14")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("arm configuration exceeds URDF limits")
        return q

    def fk(self, q):
        with self._lock:
            full = self._base.copy()
            full[self._qi] = self.check_joints(q)
            self.pin.framesForwardKinematics(self.model, self.data, full)
            reference = self.data.oMf[self._reference].homogeneous.copy()
            inverse = np.linalg.inv(reference)
            return {side: inverse @ self.data.oMf[index].homogeneous.copy() @ self._tools[side]
                    for side, index in self._frames.items()}

    def gravity(self, q):
        with self._lock:
            full = self._base.copy()
            full[self._qi] = self.check_joints(q)
            tau = self.pin.rnea(self.model, self.data, full,
                                np.zeros(self.model.nv), np.zeros(self.model.nv))
            return vector(np.asarray(tau)[self._vi], 14, "gravity torque")

    def _solver(self, sides):
        """Cache symbolic FK and IPOPT by the arms being controlled."""
        if sides in self._solvers:
            return self._solvers[sides]
        try:
            import casadi as ca
            from pinocchio import casadi as cpin
        except ImportError as exc:
            raise ImportError("G1D IK requires CasADi and Pinocchio built with CasADi bindings") from exc
        indices = [i for side in sides for i in
                   (range(7) if side == "left" else range(7, 14))]
        q = ca.SX.sym("q", len(indices))
        base = ca.SX.sym("base", self.model.nq)
        active = {self._qi[i]: j for j, i in enumerate(indices)}
        full = ca.vertcat(*[q[active[i]] if i in active else base[i]
                            for i in range(self.model.nq)])
        model = cpin.Model(self.model)
        data = model.createData()
        cpin.framesForwardKinematics(model, data, full)
        reference = data.oMf[self._reference]
        target_symbols, errors = [], []
        for side in sides:
            target = ca.SX.sym("target_" + side, 4, 4)
            target_symbols.append(target)
            pose = reference.inverse() * data.oMf[self._frames[side]]
            tool = self._tools[side]
            position = pose.translation + pose.rotation @ ca.DM(tool[:3, 3])
            rotation = pose.rotation @ ca.DM(tool[:3, :3])
            errors.extend([(position - target[:3, 3]) / self.position_tolerance,
                           cpin.log3(target[:3, :3].T @ rotation) / self.rotation_tolerance])
        error = ca.Function("eef_error", [q, base, *target_symbols], [ca.vertcat(*errors)])
        opti = ca.Opti()
        values = opti.variable(len(indices))
        base_param = opti.parameter(self.model.nq)
        seed_param = opti.parameter(len(indices))
        target_params = [opti.parameter(4, 4) for _ in sides]
        opti.minimize(ca.sumsqr(error(values, base_param, *target_params)) +
                      1e-8 * ca.sumsqr(values - seed_param))
        opti.subject_to(opti.bounded(self.lower[indices], values, self.upper[indices]))
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 150, "tol": 1e-7})
        result = (opti, values, base_param, seed_param, target_params, indices)
        self._solvers[sides] = result
        return result

    def solve(self, targets, seed):
        seed = self.check_joints(seed)
        if not targets or set(targets) - {"left", "right"}:
            raise ValueError("IK needs left and/or right targets")
        targets = {side: np.asarray(target, dtype=float).copy() for side, target in targets.items()}
        for target in targets.values():
            if (target.shape != (4, 4) or not np.isfinite(target).all() or
                    not np.allclose(target[3], [0, 0, 0, 1]) or
                    not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3)) or
                    not np.isclose(np.linalg.det(target[:3, :3]), 1.0)):
                raise ValueError("IK target must be a rigid 4x4 transform")
        sides = tuple(side for side in ("left", "right") if side in targets)
        with self._lock:
            opti, values, base_param, seed_param, params, indices = self._solver(sides)
            base = self._base.copy()
            base[self._qi] = seed
            opti.set_value(base_param, base)
            opti.set_value(seed_param, seed[indices])
            opti.set_initial(values, seed[indices])
            for param, side in zip(params, sides):
                opti.set_value(param, targets[side])
            try:
                solution = opti.solve()
            except RuntimeError as exc:
                raise ValueError("CasADi/IPOPT IK failed to converge") from exc
            result = seed.copy()
            result[indices] = np.asarray(solution.value(values)).reshape(-1)
            result = self.check_joints(result)
            actual = self.fk(result)
            for side, target in targets.items():
                position = np.linalg.norm(actual[side][:3, 3] - target[:3, 3])
                angle = np.linalg.norm(self.pin.log3(target[:3, :3].T @ actual[side][:3, :3]))
                if position > self.position_tolerance or angle > self.rotation_tolerance:
                    raise ValueError(f"{side} IK target unreachable: {position:.4f}m, {angle:.4f}rad")
            return result
