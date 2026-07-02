"""
v0.51 — UDP back-channel that closes the split-process loop (long-standing open end).

In-process, closed_loop.py already feeds the Interpreter's OUTCOME reward back into the
dopamine/cortisol learners. The LIVE deployment splits the pieces across processes:

    live_arm.py --serial COM8        (sensor -> recognize -> Interpreter -> arm)
    interpreter.py --pipe            (emits commands + OUTCOME <reward>)

...but interpreter.send_outcome only wrote to stderr/file — there was no transport carrying
OUTCOME back to the running loop. This is that transport: a tiny stdlib UDP socket on
localhost. The Interpreter process `send_outcome(r, port)`s; the loop process `recv_outcome()`s
and applies the reward to its valence/cortisol learners. No NEURON, no deps (Gemini brief
proposed a UDP bridge — we keep it stdlib + localhost).

  python arm_bridge.py        # self-check (in-process loopback, no real network needed)
"""
import socket

PREFIX = "OUTCOME"


def parse_outcome(msg):
    """'OUTCOME <reward>' -> float, else None (malformed / non-outcome packets are ignored)."""
    parts = msg.strip().split()
    if len(parts) == 2 and parts[0] == PREFIX:
        try:
            return float(parts[1])
        except ValueError:
            return None
    return None


def send_outcome(reward, port, host="127.0.0.1"):
    """Fire one OUTCOME datagram at the loop process (called by the Interpreter side)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(f"{PREFIX} {reward}".encode("ascii"), (host, port))


class OutcomeBridge:
    """Loop-side receiver. Binds a UDP port; recv_outcome() returns the next reward or None.
    Port 0 = OS-assigned ephemeral (read it back via .port)."""

    def __init__(self, host="127.0.0.1", port=0, timeout=1.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(timeout)               # never block the control loop forever
        self.port = self.sock.getsockname()[1]

    def recv_outcome(self):
        """Next reward float, or None on timeout / malformed packet. Non-blocking past timeout."""
        try:
            data, _ = self.sock.recvfrom(256)
        except socket.timeout:
            return None
        return parse_outcome(data.decode("ascii", "ignore"))

    def close(self):
        self.sock.close()


def main():
    bridge = OutcomeBridge(timeout=1.0)             # ephemeral port, short timeout (CI-safe)
    port = bridge.port

    # 1) a positive reward survives the round trip
    send_outcome(0.5, port)
    r1 = bridge.recv_outcome()
    # 2) a negative (aversive) reward too
    send_outcome(-1.0, port)
    r2 = bridge.recv_outcome()
    # 3) malformed packet is ignored (not a crash, not a false reward)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"garbage payload here", ("127.0.0.1", port))
    r3 = bridge.recv_outcome()
    # 4) closing the loop: route received rewards into a learner-like accumulator
    value = 0.0
    for reward in (r1, r2):
        if reward is not None:
            value += 0.5 * reward                   # stand-in for valence/cortisol .learn(reward)

    bridge.close()

    print("=" * 60)
    print("ARM BRIDGE — UDP OUTCOME back-channel (split-process loop)")
    print("=" * 60)
    print(f"ephemeral port : {port}")
    print(f"recv +0.5      : {r1}")
    print(f"recv -1.0      : {r2}")
    print(f"recv garbage   : {r3} (ignored)")
    print(f"loop value after applying rewards: {value:+.3f}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert r1 == 0.5, f"positive reward lost in transit: {r1}"
    assert r2 == -1.0, f"negative reward lost in transit: {r2}"
    assert r3 is None, "malformed packet must be ignored, not parsed as a reward"
    assert abs(value - (-0.25)) < 1e-9, f"rewards did not reach the learner: {value}"
    assert parse_outcome("OUTCOME 0.3") == 0.3 and parse_outcome("HELLO") is None
    print("self-check OK: OUTCOME round-trips over UDP, malformed ignored, reward reaches the loop")


if __name__ == "__main__":
    main()
