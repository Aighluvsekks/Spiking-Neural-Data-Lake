"""
v0.37 — closed-loop driver: ARM signal -> loop -> Interpreter -> reward -> loop.

Wires signal_loop's pieces (reflex, encode -> lake -> match, valence/dopamine, cortisol) to
the Interpreter end to end, IN-PROCESS, feeding the Interpreter's OUTCOME reward straight back
into dopamine + cortisol every tick. This is the schematic's full reaction loop on one machine
— no FIFO/socket needed. The live deployment splits the same pieces across processes:

    signal_loop.py --serial COM3 --feedback | interpreter.py --pipe --outcome loop_in

Run:
  python closed_loop.py        # scripted episode + self-check
"""
import signal_loop as S
from reflex import Reflex
from interpreter import Interpreter
import arm_bridge as AB


def run_episode(signals, lib=None, assume_success=True, base_threshold=0.10, trace=None,
                bridge=None):
    """Drive raw signal lines through the whole stack with live reward feedback.
    Returns the live state objects so callers can inspect what was learned."""
    lib = S.build_default_library() if lib is None else lib
    matcher = S.build_matcher(lib)
    reflex = Reflex()
    valence = S.ValenceLearner()
    cort = S.Cortisol(tau=30.0)
    itp = Interpreter(assume_success=assume_success)
    last_window = None

    for raw in signals:
        p = S.parse_line(raw) if isinstance(raw, str) else raw
        if p is None:
            continue

        # reflex (threshold sharpened by cortisol) — instinct preempts cognition
        reflex.threshold = base_threshold * cort.reflex_threshold_scale()
        ract = reflex.step(p)
        if ract:
            cmd, reward = itp.interpret({"reflex": ract})
            cort.step(1.0)                                   # danger = stressor (raises cortisol)
            # NOTE: we do NOT punish the previous gesture's value here — a collision is its
            # own danger signal, not necessarily caused by the last action. Causal credit
            # assignment ("did that command cause the danger?") is a separate open problem.
            if trace is not None:
                trace.append({"signal": "DANGER", "command": cmd, "reward": reward,
                              "stress": round(cort.level, 3)})
            continue

        # recognize -> valence overlay -> interpret -> command + reward
        window = S.normalize(S.resize_to_n(p))
        label, d, _, _ = matcher(window)
        v_action, v = valence.act(window, bias=cort.caution_bias())
        cmd, reward = itp.interpret({"match": label, "instinct": v_action,
                                     "valence": round(v, 3), "stress": round(cort.level, 3)})
        last_window = window
        # feed the Interpreter's outcome straight back into dopamine + cortisol.
        # bridge set -> route the reward over the arm_bridge UDP back-channel first (the
        # split-process deployment: Interpreter and loop on different processes). Transparent.
        if reward is not None:
            if bridge is not None:
                AB.send_outcome(reward, bridge.port)
                reward = bridge.recv_outcome()
            valence.learn(window, reward, lr_scale=cort.learn_rate_scale(reward))
            cort.step(max(0.0, -reward))
        else:
            cort.step(0.0)
        if trace is not None:
            trace.append({"match": label, "command": cmd, "instinct": v_action,
                          "reward": reward, "value": round(valence.valence(window), 3),
                          "stress": round(cort.level, 3)})

    return {"valence": valence, "cortisol": cort, "reflex": reflex, "lib": lib}


def main():
    lib = S.build_default_library()
    good_line = ",".join(f"{x:.4f}" for x in lib["GRIPPER_CLOSE"])
    good_win = S.normalize(S.resize_to_n([float(x) for x in good_line.split(",")]))
    danger = "500,500,500,500,500,500,950,0"

    # episode: repeat the GRIPPER_CLOSE gesture (executed, rewarded), then a collision
    signals = [good_line] * 30 + [danger]
    trace = []
    state = run_episode(signals, lib=lib, assume_success=True, trace=trace)
    val, cort = state["valence"], state["cortisol"]

    print("=" * 60)
    print("CLOSED LOOP — signal -> match -> command -> reward -> learn")
    print("=" * 60)
    first, last, collision = trace[0], trace[29], trace[30]
    print(f"first  : match={first['match']} cmd={first['command']} "
          f"value={first['value']:+.2f}")
    print(f"learned: match={last['match']} cmd={last['command']} "
          f"value={last['value']:+.2f}  (dopamine raised it)")
    print(f"danger : {collision['signal']} cmd={collision['command']} "
          f"reward={collision['reward']} stress={collision['stress']:.2f}")
    print(f"final  : value(GRIPPER_CLOSE)={val.valence(good_win):+.2f} "
          f"({val.act(good_win)[0]})  stress={cort.level:.2f}  "
          f"reflex_thresh_scale={cort.reflex_threshold_scale():.2f}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    cmds = [t.get("command") for t in trace]
    assert "GRIPPER_CLOSE" in cmds, "matched gesture never produced its command"
    assert "EMERGENCY_STOP" in cmds, "collision never produced an emergency stop"
    assert last["value"] > first["value"], "dopamine did not raise the rewarded gesture's value"
    assert val.act(good_win)[0] == "APPROACH", "rewarded gesture not learned as APPROACH"
    assert collision["stress"] > last["stress"] + 0.1, "collision did not raise cortisol stress"
    assert cort.reflex_threshold_scale() < 1.0, "stress did not sharpen the reflex"

    # split-process feedback: same loop with the reward routed over the arm_bridge UDP
    # back-channel must learn the SAME thing (transport transparent).
    br = AB.OutcomeBridge(timeout=1.0)
    bstate = run_episode([good_line] * 30, lib=lib, assume_success=True, bridge=br)
    br.close()
    assert bstate["valence"].act(good_win)[0] == "APPROACH", \
        "reward routed over the UDP back-channel did not close the loop"
    print(f"bridged: OUTCOME routed over arm_bridge UDP -> "
          f"value(GRIPPER_CLOSE)={bstate['valence'].valence(good_win):+.2f} (loop still closes)")
    print("self-check OK: gesture -> command, reward raised its value to APPROACH, "
          "collision -> EMERGENCY_STOP + stress -> sharpened reflex, UDP back-channel transparent (loop closed)")


if __name__ == "__main__":
    main()
