"""
v0.44 — robot-arm simulation (sim-to-real): the closed loop drives a virtual 2-link arm.

The Interpreter's commands move a virtual arm: gesture -> command -> JOINT motion ->
end-effector pose -> collision check, with NO hardware. Same command API drives sim or
(future) a real arm = sim-to-real.

Backends:
  - stdlib  : exact 2-link planar forward kinematics + geometric collision. Always available,
              runs in CI (zero deps). The verified core.
  - pybullet: real rigid-body physics + contact collision when `pip install pybullet`, headless
              `p.DIRECT` (no window, CI-safe). Same commands, real dynamics. Falls back to
              stdlib if pybullet is absent or errors.

  python arm_sim.py
"""
import re
import math

from snn_data_lake.arm_config import TRAJ_DEV_MAX        # trajectory-deviation reflex bound (calibration)

DEG = math.pi / 180.0
FLOOR_Y = -0.20                 # end-effector below this = ground collision
OBSTACLE = (0.4, 1.3)           # point obstacle (x, y)
OBSTACLE_R = 0.30

_JOINT = re.compile(r"JOINT_([AB])_ROTATE\(([+-]?\d+(?:\.\d+)?)deg\)")


class ArmSim:
    """2-joint planar arm. apply(command) executes one Interpreter command and returns state."""

    def __init__(self, l1=1.0, l2=1.0, use_pybullet=False):
        self.l1, self.l2 = l1, l2
        self.theta = [0.0, 0.0]     # joint A, B angles (radians)
        self.gripper = 0.0          # 0 = open .. 1 = closed
        self.stopped = False
        self.backend = "stdlib"
        self._p = None
        self.expected_ee = self.ee()   # where the EE SHOULD be given the last command (trajectory ref)
        if use_pybullet:
            self._init_pybullet()

    # ---- end-effector + collision (geometric, authoritative + deterministic) ----
    def ee(self):
        t0, t01 = self.theta[0], self.theta[0] + self.theta[1]
        x = self.l1 * math.cos(t0) + self.l2 * math.cos(t01)
        y = self.l1 * math.sin(t0) + self.l2 * math.sin(t01)
        return (x, y)

    def collision(self):
        x, y = self.ee()
        if y < FLOOR_Y:
            return "GROUND"
        if math.hypot(x - OBSTACLE[0], y - OBSTACLE[1]) < OBSTACLE_R:
            return "OBSTACLE"
        return None

    # ---- command execution -------------------------------------------------
    def apply(self, command):
        if command == "EMERGENCY_STOP":
            self.stopped = True
            return self.state()
        if self.stopped and command not in ("RETRACT_ALL", "HOME"):
            return self.state()                       # frozen until retract / home
        m = _JOINT.fullmatch(command or "")
        if m:
            j = 0 if m.group(1) == "A" else 1
            self.theta[j] += float(m.group(2)) * DEG
        elif command == "GRIPPER_CLOSE":
            self.gripper = 1.0
        elif command == "GRIPPER_OPEN":
            self.gripper = 0.0
        elif command in ("HOME", "RETRACT_ALL"):
            self.theta = [0.0, 0.0]
            self.gripper = 0.0
            self.stopped = False
        # HOLD / unknown -> no motion
        self.expected_ee = self.ee()       # a command defines where the arm intends to be
        if self._p is not None:
            self._sync_pybullet()
        return self.state()

    # ---- trajectory-deviation comparator (the "differential position comparator" reflex) ----
    def perturb(self, da_deg=0.0, db_deg=0.0):
        """External disturbance (collision, load shift, bump) — moves the arm WITHOUT a command,
        so the commanded `expected_ee` is unchanged and `deviation()` grows. This is the off-path
        event the local trajectory comparator must catch before the cloud lake even sees it."""
        self.theta[0] += da_deg * DEG
        self.theta[1] += db_deg * DEG
        if self._p is not None:
            self._sync_pybullet()
        return self.state()

    def deviation(self):
        """Euclidean distance between the ACTUAL end-effector and the intended (commanded) pose."""
        x, y = self.ee()
        ex, ey = self.expected_ee
        return math.hypot(x - ex, y - ey)

    def trajectory_breach(self, bound=TRAJ_DEV_MAX):
        """Edge trajectory reflex: True when the arm has strayed too far from its commanded path."""
        return self.deviation() > bound

    def state(self):
        x, y = self.ee()
        return {"theta_deg": [round(t / DEG, 1) for t in self.theta],
                "ee": (round(x, 3), round(y, 3)), "gripper": self.gripper,
                "stopped": self.stopped, "collision": self.collision(),
                "deviation": round(self.deviation(), 3), "backend": self.backend}

    # ---- optional pybullet physics (headless) ------------------------------
    def _init_pybullet(self):
        try:
            import pybullet as p
            self.cid = p.connect(p.DIRECT)            # headless: no OpenGL window (CI-safe)
            p.setGravity(0, 0, 0, physicsClientId=self.cid)   # kinematic planar arm
            # base + 2 revolute links rotating about Z, lying in the XY plane
            link_l = [self.l1, self.l2]
            col = [p.createCollisionShape(p.GEOM_BOX, halfExtents=[ll / 2, 0.05, 0.05],
                                          physicsClientId=self.cid) for ll in link_l]
            self.arm = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1, basePosition=[0, 0, 0],
                linkMasses=[1, 1], linkCollisionShapeIndices=col,
                linkVisualShapeIndices=[-1, -1],
                linkPositions=[[link_l[0] / 2, 0, 0], [link_l[1], 0, 0]],
                linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                linkParentIndices=[0, 1], linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
                linkJointAxis=[[0, 0, 1], [0, 0, 1]], physicsClientId=self.cid)
            self.obstacle = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_SPHERE, radius=OBSTACLE_R, physicsClientId=self.cid),
                basePosition=[OBSTACLE[0], OBSTACLE[1], 0], physicsClientId=self.cid)
            self._p = p
            self.backend = "pybullet"
        except Exception:
            self._p = None                            # any failure -> stdlib backend

    def _sync_pybullet(self):
        p = self._p
        for j in (0, 1):
            p.resetJointState(self.arm, j, self.theta[j], physicsClientId=self.cid)
        p.stepSimulation(physicsClientId=self.cid)

    def pybullet_contacts(self):
        """Real contact count from physics (only meaningful on the pybullet backend)."""
        if self._p is None:
            return None
        pts = self._p.getClosestPoints(self.arm, self.obstacle, distance=0.0,
                                       physicsClientId=self.cid)
        return len(pts)

    def close(self):
        if self._p is not None:
            self._p.disconnect(physicsClientId=self.cid)


def drive(arm, commands):
    """Run a list of Interpreter commands through the arm; return the trace of states."""
    return [arm.apply(c) for c in commands]


def main():
    arm = ArmSim()                                    # stdlib backend (CI path)

    home = arm.apply("HOME")
    assert home["ee"] == (2.0, 0.0) and home["collision"] is None, "HOME pose wrong"

    s = arm.apply("JOINT_A_ROTATE(+15deg)")
    assert s["theta_deg"][0] == 15.0 and s["ee"][1] > 0, "joint A did not raise the arm"

    arm.apply("GRIPPER_CLOSE")
    assert arm.gripper == 1.0, "gripper did not close"

    # drive into the ground -> collision detected (reflex would EMERGENCY_STOP)
    arm.apply("HOME")
    coll = arm.apply("JOINT_A_ROTATE(-90deg)")
    assert coll["collision"] == "GROUND", f"expected ground collision, got {coll['collision']}"

    # EMERGENCY_STOP freezes motion until RETRACT_ALL
    arm.apply("HOME")
    arm.apply("EMERGENCY_STOP")
    frozen = arm.apply("JOINT_A_ROTATE(+30deg)")
    assert frozen["theta_deg"][0] == 0.0 and frozen["stopped"], "stop did not freeze the arm"
    back = arm.apply("RETRACT_ALL")
    assert not back["stopped"] and back["ee"] == (2.0, 0.0), "retract did not recover"

    # trajectory-deviation comparator: a command stays on-path; an external perturb trips it
    arm.apply("HOME")
    assert arm.deviation() < 1e-9 and not arm.trajectory_breach(), "commanded pose must read on-path"
    arm.perturb(da_deg=20)                              # external shove, NOT a command
    assert arm.trajectory_breach(), "comparator missed an off-path perturbation"
    dev = arm.deviation()
    arm.apply("HOME")                                   # corrective re-home recommits the path
    assert not arm.trajectory_breach(), "corrective HOME did not clear the trajectory breach"

    # integration: Interpreter command -> arm motion
    from snn_data_lake.interpreter import Interpreter
    cmd = Interpreter().interpret({"match": "JOINT_B_ROTATE"})[0]   # -> "JOINT_B_ROTATE(-10deg)"
    arm.apply("HOME")
    moved = arm.apply(cmd)
    assert moved["theta_deg"][1] == -10.0, f"interpreter cmd '{cmd}' did not move joint B"

    print("=" * 56)
    print("ARM SIM (2-link, sim-to-real) — closed loop drives the arm")
    print("=" * 56)
    print(f"backend     : {arm.backend}  (pybullet if installed, else stdlib)")
    print(f"HOME        : ee={home['ee']}")
    print(f"JOINT_A +15 : ee={s['ee']}")
    print(f"collision   : JOINT_A -90deg -> {coll['collision']}")
    print(f"trajectory  : perturb +20deg -> deviation {dev:.3f}m (bound {TRAJ_DEV_MAX}) -> breach")
    print(f"interpreter : '{cmd}' -> joint B = {moved['theta_deg'][1]} deg")
    print("=" * 56)
    print("self-check OK: commands move joints, collision detected, E-STOP freezes, "
          "trajectory comparator catches off-path perturbation, Interpreter drives the arm")
    arm.close()


if __name__ == "__main__":
    main()
