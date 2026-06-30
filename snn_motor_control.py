"""
v0.54 — surrogate-gradient SNN motor control (Gemini brief: continuous actuation via
surrogate gradients). RESEARCH TRACK — needs torch+snntorch, runs in its own venv, NOT in CI.

A spike is non-differentiable (Dirac delta), so plain BPTT can't train a spiking controller.
snnTorch's surrogate gradient swaps a smooth function in on the BACKWARD pass only — the
forward pass stays binary spikes. We use it to train a **Policy Network**: given the arm's
current joint angles + a target end-effector position, output the joint deltas that move the
2-link arm's tip to the target. Trained end-to-end through a differentiable forward-kinematics
loss (BPTT-with-surrogate).

Scope (per plan): this is the POLICY half on a 2-DOF reach task — the minimal continuous-control
target. The brief's full Forward-Dynamics-Model + 6-DOF Franka is deliberately out of scope.

Run (own venv):
  .venv-arm/Scripts/python snn_motor_control.py
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

L1, L2 = 1.0, 1.0            # link lengths (match arm_sim)


def fk(theta):
    """Differentiable forward kinematics: joint angles [.,2] -> end-effector (x,y) [.,2]."""
    t0, t01 = theta[:, 0], theta[:, 0] + theta[:, 1]
    x = L1 * torch.cos(t0) + L2 * torch.cos(t01)
    y = L1 * torch.sin(t0) + L2 * torch.sin(t01)
    return torch.stack([x, y], dim=1)


class SpikingPolicy(nn.Module):
    """2 spiking LIF layers; readout = time-averaged output membrane = continuous joint deltas."""

    def __init__(self, in_dim=4, hidden=64, out_dim=2, steps=20, beta=0.9):
        super().__init__()
        self.steps = steps
        spike_grad = surrogate.fast_sigmoid()                 # the surrogate gradient
        self.fc1 = nn.Linear(in_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, out_dim)
        # output layer integrates with no reset -> its membrane is the analog control signal
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        acc = 0.0
        for _ in range(self.steps):                            # constant-input rate coding
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            acc = acc + mem2
        return 0.3 * torch.tanh(acc / self.steps)              # bounded joint deltas (rad)


def sample(batch, gen):
    """Random current pose + a REACHABLE target (current pose nudged by a small true delta)."""
    theta = (torch.rand(batch, 2, generator=gen) * 2 - 1)      # [-1,1] rad
    true_delta = (torch.rand(batch, 2, generator=gen) * 2 - 1) * 0.3
    target = fk(theta + true_delta)
    x = torch.cat([theta, target], dim=1)                      # net input: (t0,t1,tx,ty)
    return theta, target, x


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net = SpikingPolicy()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.MSELoss()

    # baseline error: do nothing (delta=0) -> tip stays put while target moved
    th0, tg0, x0 = sample(512, gen)
    noop_err = (fk(th0) - tg0).pow(2).sum(1).sqrt().mean().item()

    first_loss = None
    for it in range(400):
        theta, target, x = sample(128, gen)
        delta = net(x)
        reached = fk(theta + delta)
        loss = lossf(reached, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if first_loss is None:
            first_loss = loss.item()
    final_loss = loss.item()

    # held-out reach error
    with torch.no_grad():
        th, tg, x = sample(512, gen)
        reach_err = (fk(th + net(x)) - tg).pow(2).sum(1).sqrt().mean().item()

    print("=" * 62)
    print("SNN MOTOR CONTROL — surrogate-gradient spiking policy (2-DOF reach)")
    print("=" * 62)
    print(f"backend        : torch {torch.__version__}, snntorch (surrogate=fast_sigmoid)")
    print(f"train loss     : {first_loss:.4f} -> {final_loss:.4f}  ({first_loss/max(1e-9,final_loss):.1f}x lower)")
    print(f"reach error    : no-op {noop_err:.3f} m  ->  trained policy {reach_err:.3f} m")
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert final_loss < 0.3 * first_loss, f"surrogate-gradient training did not converge: {first_loss}->{final_loss}"
    assert reach_err < 0.5 * noop_err, f"policy no better than no-op: {reach_err} vs {noop_err}"
    print(f"self-check OK: surrogate-gradient BPTT trained a spiking policy — loss {first_loss:.3f}->{final_loss:.3f}, "
          f"reach error {noop_err:.3f}->{reach_err:.3f} m (gradient path through spikes works)")


if __name__ == "__main__":
    main()
