"""
v0.57 — true joint world-model RL (RESEARCH, own venv, NOT CI).

v0.54/v0.56 trained the Policy Network and the Forward Dynamics Model SEPARATELY and supervised.
This closes them into ONE loop, the Dreamer / model-based-RL pattern:

  1. ACT:   roll the spiking policy in the REAL arm and collect transitions (theta, action, theta').
  2. LEARN: train the spiking WORLD MODEL on those real transitions (it must learn the arm's
            NONLINEAR actuation map — commanded delta -> actual joint change, with a per-joint
            gain + saturation, so the model has something real to learn).
  3. IMAGINE: improve the policy by rolling it out THROUGH the frozen world model over a
            multi-step horizon and backpropagating the reach reward (analytic gradients through
            the learned spiking model). No real-arm samples spent on the policy update.
  Repeat — both nets improve together. The policy never sees real dynamics directly; it learns
  to control through what the world model has learned. That is the "joint world-model" claim.

Both nets are spiking (snnTorch surrogate gradients). Scope: 2-DOF, short horizon, analytic
gradients (the reliable regime for a small smooth problem). True high-dim model-based RL
(stochastic latents, value learning, 6-DOF) stays out of scope.

Run (own venv):  .venv-arm/Scripts/python snn_world_model_rl.py
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from snn_motor_control import fk            # differentiable 2-link forward kinematics (reuse)

GAIN = torch.tensor([1.0, 0.5])             # per-joint actuation gain (joint B weaker) — the
H = 5                                       # rollout horizon (steps per episode)   nonlinearity
STEPS = 10                                  # LIF timesteps per net evaluation


def real_step(theta, delta):
    """The REAL arm dynamics the world model must learn: nonlinear, saturating, per-joint gain."""
    return theta + GAIN * torch.tanh(delta)


class _SpikingNet(nn.Module):
    """Shared 2-layer spiking MLP; readout = time-averaged output membrane (analog)."""

    def __init__(self, in_dim, out_dim, scale, hidden=64):
        super().__init__()
        self.scale = scale
        sg = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=sg)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=sg, reset_mechanism="none")

    def forward(self, x):
        m1, m2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        acc = 0.0
        for _ in range(STEPS):
            s1, m1 = self.lif1(self.fc1(x), m1)
            _, m2 = self.lif2(self.fc2(s1), m2)
            acc = acc + m2
        return self.scale * torch.tanh(acc / STEPS)


def Policy():       # (theta, target_ee) -> action delta
    return _SpikingNet(4, 2, scale=1.2)


def WorldModel():   # (theta, delta) -> predicted joint change dtheta
    return _SpikingNet(4, 2, scale=1.2)


def start_and_target(batch, gen):
    theta = (torch.rand(batch, 2, generator=gen) * 2 - 1)
    goal = theta + (torch.rand(batch, 2, generator=gen) * 2 - 1) * 0.6   # reachable in H steps
    return theta, fk(goal)


def collect_real(policy, gen, batch=128):
    """Roll the policy in the REAL arm; return transitions + the mean final reach distance."""
    theta, target = start_and_target(batch, gen)
    ths, dts, nxt = [], [], []
    with torch.no_grad():
        for _ in range(H):
            delta = policy(torch.cat([theta, target], 1))
            nt = real_step(theta, delta)
            ths.append(theta); dts.append(delta); nxt.append(nt - theta)   # store real dtheta
            theta = nt
        reach = (fk(theta) - target).pow(2).sum(1).sqrt().mean().item()
    return (torch.cat(ths), torch.cat(dts), torch.cat(nxt)), reach


def train_world_model(wm, opt, trans, iters=8):
    th, dt, real_dtheta = trans
    lossf = nn.MSELoss()
    for _ in range(iters):
        pred = wm(torch.cat([th, dt], 1))
        loss = lossf(pred, real_dtheta)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def improve_policy(policy, wm, opt, gen, batch=128, iters=8):
    """Imagined rollout THROUGH the frozen world model; backprop the reach reward to the policy."""
    for p in wm.parameters():
        p.requires_grad_(False)
    for _ in range(iters):
        theta, target = start_and_target(batch, gen)
        cost = 0.0
        for _ in range(H):
            delta = policy(torch.cat([theta, target], 1))
            theta = theta + wm(torch.cat([theta, delta], 1))      # imagine via learned dynamics
            cost = cost + (fk(theta) - target).pow(2).sum(1).mean()
        opt.zero_grad(); (cost / H).backward(); opt.step()
    for p in wm.parameters():
        p.requires_grad_(True)


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    policy, wm = Policy(), WorldModel()
    p_opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    w_opt = torch.optim.Adam(wm.parameters(), lr=3e-3)

    _, noop_reach = collect_real(policy, gen)            # untrained baseline
    first_reach = noop_reach
    wm_err = None
    for epoch in range(60):
        trans, reach = collect_real(policy, gen)         # 1. ACT in the real arm
        wm_err = train_world_model(wm, w_opt, trans)     # 2. LEARN the dynamics
        improve_policy(policy, wm, p_opt, gen)           # 3. IMAGINE -> improve policy
    _, final_reach = collect_real(policy, gen)

    # world-model accuracy on fresh real transitions
    (th, dt, real_dtheta), _ = collect_real(policy, gen)
    with torch.no_grad():
        wm_pred_err = (wm(torch.cat([th, dt], 1)) - real_dtheta).pow(2).sum(1).sqrt().mean().item()

    print("=" * 64)
    print("JOINT WORLD-MODEL RL — spiking policy learns THROUGH a learned spiking model")
    print("=" * 64)
    print(f"backend         : torch {torch.__version__}, snntorch (surrogate, analytic-gradient MBRL)")
    print(f"dynamics        : theta' = theta + [1.0,0.5]*tanh(delta)  (nonlinear, per-joint gain)")
    print(f"horizon / steps : H={H} rollout, {STEPS} LIF steps/net")
    print(f"real reach dist : no-op {noop_reach:.3f} m  ->  trained {final_reach:.3f} m")
    print(f"world-model err : last-train {wm_err:.4f}  | held-out dtheta {wm_pred_err:.3f} rad")
    print("=" * 64)

    # ---- self-checks --------------------------------------------------------
    assert final_reach < 0.5 * noop_reach, f"policy did not learn to reach: {noop_reach:.3f}->{final_reach:.3f}"
    assert wm_pred_err < 0.1, f"world model never learned the actuation map: {wm_pred_err:.3f}"
    print(f"self-check OK: closed RL loop — policy learned to reach THROUGH the learned spiking world "
          f"model (real reach {noop_reach:.3f}->{final_reach:.3f} m, world-model err {wm_pred_err:.3f} rad)")


if __name__ == "__main__":
    main()
