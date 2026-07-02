"""
v0.52 — NEF-style spiking PID controller (Gemini brief: "NEF-Based Spiking PID").

A standard PID runs on a digital CPU, which negates the latency/energy point of a spiking
system. The Neural Engineering Framework maps the PID math onto neuron populations:

  Proportional population : firing represents the live positional error (population-coded,
                            so sensor noise on any one neuron barely moves the estimate).
  Integral population     : a leaky accumulator that sustains activity = the error integral.
  Derivative population   : the rate of change of the error (predictive damping).

We keep it stdlib: the error is represented through the RBF population code
(population_encoding.PopulationEncoder) — that's the NEF "encode into a population, decode a
value" step — then combined with PID gains and used to drive arm_sim's joint. No Nengo, no torch.

We VERIFY convergence + noise robustness (not the brief's 30%-vs-digital claim).

  python spiking_pid.py        # self-check
"""
import random

from snn_data_lake.population_encoding import PopulationEncoder
from snn_data_lake.arm_sim import ArmSim, DEG


class SpikingPID:
    """Population-coded PID. step(error_deg) -> control command (delta degrees)."""

    def __init__(self, kp=0.6, ki=0.05, kd=0.15, dt=1.0, err_max=90.0, n=15):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.err_max = err_max
        self.enc = PopulationEncoder(n=n, lo=-err_max, hi=err_max, max_spikes=20)
        self.integral = 0.0          # integral population's sustained state
        self.prev = 0.0              # last error (for the derivative population)

    def _encode_error(self, e):
        """NEF step: error -> population spikes -> decoded error (robust to per-neuron noise)."""
        e = max(-self.err_max + 1e-6, min(self.err_max - 1e-6, e))   # clamp into the coded range
        est = self.enc.decode(self.enc.encode(e))
        return e if est is None else est

    def step(self, error):
        e = self._encode_error(error)
        self.integral += e * self.dt                 # integral population accumulates
        deriv = (e - self.prev) / self.dt            # derivative population
        self.prev = e
        return self.kp * e + self.ki * self.integral + self.kd * deriv


def drive_to(target_deg, steps=60, noise=0.0, spiking=True, seed=0,
             perturb_at=None, perturb_deg=25.0):
    """Drive arm_sim joint A to target_deg; return (error trace, final angle, breach count).
    If perturb_at is set, an external shove is injected at that step and arm_sim's
    trajectory-deviation comparator (v0.51) flags the off-path event — the controller then
    drives the joint back. Controller + edge safety reflex working together."""
    rng = random.Random(seed)
    arm = ArmSim()
    arm.apply("HOME")
    pid = SpikingPID() if spiking else None
    kp, ki, kd, integ, prev = 0.6, 0.05, 0.15, 0.0, 0.0
    errs, breaches = [], 0
    for step in range(steps):
        measured = arm.state()["theta_deg"][0] + (rng.uniform(-noise, noise) if noise else 0.0)
        e = target_deg - measured
        errs.append(target_deg - arm.state()["theta_deg"][0])      # true error (no noise)
        if spiking:
            u = pid.step(e)
        else:                                                       # plain numeric PID baseline
            integ += e * 1.0
            u = kp * e + ki * integ + kd * (e - prev)
            prev = e
        u = max(-15.0, min(15.0, u))                                # per-step slew limit
        arm.apply(f"JOINT_A_ROTATE({u:+.4f}deg)")                   # commanded pose -> expected_ee
        if perturb_at is not None and step == perturb_at:
            arm.perturb(da_deg=perturb_deg)                         # external shove off the path
            if arm.trajectory_breach():                            # comparator catches it
                breaches += 1
    return errs, arm.state()["theta_deg"][0], breaches


def _rmse(errs):
    return (sum(e * e for e in errs) / len(errs)) ** 0.5


def main():
    target = 30.0

    errs_s, final_s, _ = drive_to(target, spiking=True)
    errs_n, final_n, _ = drive_to(target, spiking=False)
    # noise robustness: 4 deg measurement jitter
    errs_sn, final_sn, _ = drive_to(target, noise=4.0, spiking=True)
    errs_nn, final_nn, _ = drive_to(target, noise=4.0, spiking=False)
    # perturbation rejection: external shove mid-run, trajectory comparator flags it, PID recovers
    errs_p, final_p, breaches = drive_to(target, spiking=True, perturb_at=20, perturb_deg=25)

    print("=" * 62)
    print("SPIKING PID (NEF population-coded) — drive arm_sim joint A to setpoint")
    print("=" * 62)
    print(f"target            : {target} deg")
    print(f"spiking PID       : final {final_s:.2f} deg  | settle RMSE {_rmse(errs_s):.2f}")
    print(f"numeric PID (base): final {final_n:.2f} deg  | settle RMSE {_rmse(errs_n):.2f}")
    print(f"+4deg noise spk   : final {final_sn:.2f} deg | steady |err| {abs(target-final_sn):.2f}")
    print(f"+4deg noise num   : final {final_nn:.2f} deg | steady |err| {abs(target-final_nn):.2f}")
    print(f"perturb +25deg@20 : comparator breaches {breaches} -> recovered to {final_p:.2f} deg")
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert abs(final_s - target) < 1.0, f"spiking PID did not converge: {final_s}"
    assert abs(errs_s[0]) > abs(errs_s[-1]), "error did not shrink over the run"
    overshoot = max(0.0, -min(errs_s))          # error went negative => overshoot past target
    assert overshoot < 0.4 * target, f"excessive overshoot: {overshoot:.1f} deg"
    # under measurement noise the population-coded error still settles close
    assert abs(final_sn - target) < 2.0, f"spiking PID not noise-robust: {final_sn}"
    # trajectory comparator catches the injected shove, and the controller drives back
    assert breaches >= 1, "trajectory comparator missed the injected perturbation"
    assert abs(final_p - target) < 2.0, f"controller did not recover from the perturbation: {final_p}"
    print(f"self-check OK: spiking PID -> {final_s:.2f}deg (target {target}), overshoot {overshoot:.1f}deg, "
          f"noise-robust {abs(target-final_sn):.2f}deg, comparator caught perturb ({breaches}) + recovered to {final_p:.2f}deg")


if __name__ == "__main__":
    main()
