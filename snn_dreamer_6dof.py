"""
v0.58 — high-dim spiking MBRL: Dreamer-lite on a 6-DOF arm (RESEARCH, own venv, NOT CI).

v0.57 was deterministic 2-DOF. This adds the three pieces that make it a real world-model agent:

  * 6-DOF arm        : a 6-joint **planar redundant** arm (6 actuated joints -> a 2D tip). The
                       redundancy (6 -> 2) is the genuine high-dim control challenge — many joint
                       configs reach the same point. Planar (2D workspace), NOT a 3D Franka URDF
                       or its dynamics (honest ceiling); the point is high-dim *control*, not the
                       exact geometry.
  * Stochastic latent: the world model is a latent-variable model — encode state -> Gaussian
                       posterior, SAMPLE z (reparameterized); a transition prior predicts the
                       next-latent distribution. KL ties posterior <-> prior (the RSSM core).
  * Value function   : an actor-critic trained entirely in IMAGINATION — roll latents through the
                       learned transition, bootstrap returns with a learned critic V(z), train the
                       actor to maximize those returns (analytic gradients).

The **actor and critic are spiking** (snnTorch surrogate gradients) — the RL controller is the
neuromorphic part. The **world model (encoder/transition/decoder) is a plain MLP**: a small
spiking net cannot regress the 6-joint kinematics precisely enough, and a wrong world model
gives the actor wrong imagined gradients (measured: spiking-WM recon ~0.9 m, agent fails to
reach). MLP world model + spiking actor-critic is the honest, working split. The world model is
warmed up on real data before the agent learns through it. Scope: small latent, short horizon,
analytic-gradient actor. True high-dim Dreamer (pixels, discrete latents, 3D dynamics) is out of
scope.

Run (own venv):  .venv-arm/Scripts/python snn_dreamer_6dof.py
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

NJ, ZD, ED = 6, 16, 2                    # joints, latent dim, end-effector dim (planar)
H, STEPS, HID = 4, 8, 128                # horizon, LIF steps/net, hidden width
LINK = 0.5


def fk6(theta):
    """6 joint angles [B,6] -> 2D planar end-effector [B,2] (cumulative-angle link chain)."""
    cum = torch.cumsum(theta, dim=1)
    return torch.stack([(LINK * torch.cos(cum)).sum(1), (LINK * torch.sin(cum)).sum(1)], -1)


class SNet(nn.Module):
    """2-layer spiking MLP; readout = time-averaged output membrane."""

    def __init__(self, in_dim, out_dim, scale=1.0, hidden=HID):
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
        return self.scale * (acc / STEPS)


class MLP(nn.Module):
    """Plain MLP for the world model (needs precise FK regression the spiking net can't match)."""

    def __init__(self, in_dim, out_dim, hidden=HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def sample(mu, logsig):
    return mu + torch.exp(logsig) * torch.randn_like(mu)


def _split(out):
    """Split a net output into (mu, logsig) with logsig clamped LOW -> stochastic but low-noise
    latent (so the decoder can still ground it). KL keeps posterior near the transition prior."""
    mu, ls = out.chunk(2, -1)
    return mu, ls.clamp(-4.0, -2.0)            # sigma ~0.018-0.135: stochastic but low-noise


def kl(mu_q, ls_q, mu_p, ls_p):
    vq, vp = torch.exp(2 * ls_q), torch.exp(2 * ls_p)
    return (ls_p - ls_q + (vq + (mu_q - mu_p) ** 2) / (2 * vp) - 0.5).sum(-1).mean()


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = MLP(NJ, 2 * ZD)                # state -> posterior (mu, logsig)
        self.trans = MLP(ZD + NJ, 2 * ZD)         # (z, action) -> next-latent prior
        self.dec = MLP(ZD, ED)                    # z -> end-effector (grounds the latent)

    def posterior(self, angles):
        return _split(self.enc(angles))

    def prior(self, z, action):
        return _split(self.trans(torch.cat([z, action], -1)))


def reach(theta, target):
    return (fk6(theta) - target).pow(2).sum(-1).sqrt()


LIM = 0.4                                # joint limit (rad) — keeps the 6-link FK smooth/learnable


def start_target(batch, gen):
    theta = (torch.rand(batch, NJ, generator=gen) * 2 - 1) * LIM
    goal = (torch.rand(batch, NJ, generator=gen) * 2 - 1) * LIM
    return theta, fk6(goal)


def collect(actor, wm, gen, batch=128, explore=0.3):
    """Roll the actor in the REAL arm; return (target, per-step [angles, action, ee], reach)."""
    theta, target = start_target(batch, gen)
    seq = []
    with torch.no_grad():
        for _ in range(H):
            z = sample(*wm.posterior(theta))
            a = actor(torch.cat([z, target], -1)) + explore * torch.randn(batch, NJ)
            theta = (theta + 0.2 * torch.tanh(a)).clamp(-LIM, LIM)   # actuation + joint limits
            seq.append((theta.clone(), a.clone(), fk6(theta)))
        return target, seq, reach(theta, target).mean().item()


def update_world_model(wm, w_opt, seq, iters=10):
    """Three terms: recon (ground latent->EE), PRED (transition predicts next EE — what the actor
    relies on), KL (posterior<->prior consistency). Returns recon RMSE."""
    angles = torch.stack([s[0] for s in seq])
    acts = torch.stack([s[1] for s in seq])
    ees = torch.stack([s[2] for s in seq])
    recon = None
    for _ in range(iters):
        recon = pred = kl_loss = 0.0
        z_prev = None
        for t in range(H):
            mu_q, ls_q = wm.posterior(angles[t])
            z = sample(mu_q, ls_q)
            recon = recon + (wm.dec(z) - ees[t]).pow(2).sum(-1).mean()
            if z_prev is not None:
                mu_p, ls_p = wm.prior(z_prev, acts[t - 1])
                pred = pred + (wm.dec(sample(mu_p, ls_p)) - ees[t]).pow(2).sum(-1).mean()
                kl_loss = kl_loss + kl(mu_q, ls_q, mu_p, ls_p)
            z_prev = z.detach()
        loss = recon / H + pred / max(1, H - 1) + 0.1 * kl_loss / max(1, H - 1)
        w_opt.zero_grad(); loss.backward(); w_opt.step()
    return (recon / H).item() ** 0.5


def update_actor_critic(wm, actor, critic, a_opt, c_opt, gen, iters=4):
    for p in wm.parameters():
        p.requires_grad_(False)
    th0, tgt = start_target(128, gen)
    B = th0.shape[0]
    c_loss_val = None
    for _ in range(iters):
        # ACTOR: differentiable imagined rollout, bootstrap with the (detached) critic
        z = sample(*wm.posterior(th0)).detach()
        disc_r = 0.0
        for t in range(H):
            a = actor(torch.cat([z, tgt], -1))
            z = sample(*wm.prior(z, a))
            disc_r = disc_r + (0.95 ** t) * (-(wm.dec(z) - tgt).pow(2).sum(-1))
        boot = (0.95 ** H) * critic(torch.cat([z, tgt], -1)).squeeze(-1).detach()
        a_loss = -(disc_r + boot).mean()
        a_opt.zero_grad(); a_loss.backward(); a_opt.step()

        # CRITIC: no-grad rollout -> Monte-Carlo returns, fit V(z_t)
        with torch.no_grad():
            z = sample(*wm.posterior(th0))
            zs, rs = [], []
            for t in range(H):
                a = actor(torch.cat([z, tgt], -1))
                z = sample(*wm.prior(z, a))
                zs.append(z); rs.append(-(wm.dec(z) - tgt).pow(2).sum(-1))
            ret, tgts = torch.zeros(B), []
            for t in reversed(range(H)):
                ret = rs[t] + 0.95 * ret; tgts.insert(0, ret)
            targets, zstack = torch.stack(tgts), torch.stack(zs)
        tgt_rep = tgt.unsqueeze(0).expand(H, B, ED)
        value = critic(torch.cat([zstack, tgt_rep], -1).reshape(H * B, ZD + ED)).reshape(H, B)
        c_loss = (value - targets).pow(2).mean()
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()
        c_loss_val = c_loss.item()
    for p in wm.parameters():
        p.requires_grad_(True)
    return c_loss_val


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    wm = WorldModel()
    actor = SNet(ZD + ED, NJ, scale=1.5)
    critic = SNet(ZD + ED, 1, scale=3.0)
    w_opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    a_opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
    c_opt = torch.optim.Adam(critic.parameters(), lr=2e-3)

    _, _, noop = collect(actor, wm, gen)

    # WARMUP: ground the world model on real transitions before the agent learns through it
    recon0 = None
    for _ in range(40):
        _, seq, _ = collect(actor, wm, gen)
        r = update_world_model(wm, w_opt, seq)
        if recon0 is None:
            recon0 = r

    # JOINT: world model + actor-critic together
    crit0 = None
    for _ in range(70):
        _, seq, _ = collect(actor, wm, gen)
        recon_err = update_world_model(wm, w_opt, seq)
        c = update_actor_critic(wm, actor, critic, a_opt, c_opt, gen)
        if crit0 is None:
            crit0 = c
    crit_final = c

    _, _, final = collect(actor, wm, gen, explore=0.0)

    print("=" * 66)
    print("DREAMER-LITE 6-DOF — spiking world-model agent (stochastic latent + value)")
    print("=" * 66)
    print(f"backend         : torch {torch.__version__}, snntorch (surrogate)")
    print(f"arm             : {NJ}-joint planar redundant arm (6 actuators -> 2D tip); latent ZD={ZD}")
    print(f"real reach dist : no-op {noop:.3f} m  ->  trained {final:.3f} m")
    print(f"world-model recon (decoded EE): {recon0:.3f} -> {recon_err:.3f} m")
    print(f"critic loss     : {crit0:.3f} -> {crit_final:.3f}  (value fit to imagined returns)")
    print("=" * 66)

    # ---- self-checks --------------------------------------------------------
    # bound tolerant to snntorch-version init variance (0.26 on the pinned 1.0.0, ~0.34 on newer);
    # still >3x below the untrained recon (~1.33) -> the latent is grounded either way.
    assert recon_err < 0.40, f"latent world model never grounded to EE: recon {recon_err:.3f} m"
    assert final < 0.7 * noop, f"actor did not learn to reach the 6-joint arm: {noop:.3f}->{final:.3f}"
    assert crit_final < crit0, f"value function did not fit returns: {crit0:.3f}->{crit_final:.3f}"
    print(f"self-check OK: 6-DOF spiking Dreamer — stochastic-latent world model grounded "
          f"(recon {recon_err:.3f} m), value-guided actor reaches (real {noop:.3f}->{final:.3f} m)")


if __name__ == "__main__":
    main()
