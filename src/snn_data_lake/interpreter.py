"""
v0.37 — the Interpreter: matched signal -> robot command + outcome feedback.

The downstream box of signal_loop.py (the schematic's "Reaction / Command Translation").
It reads the loop's JSON events on stdin, translates a matched signature LABEL into a
concrete robot command (JOINT_A_ROTATE(+15deg), GRIPPER_CLOSE, ...), and closes the loop
by sending an `OUTCOME <reward>` back to `signal_loop --feedback`.

Priority (instinct over cognition):
  1. reflex event (STOP / WITHDRAW)  -> preempts everything, emergency command, reward -1
  2. AVOID instinct (learned aversion / high stress) -> vetoes the command (HOLD), reward -0.3
  3. confident match                 -> the mapped robot command

Reward policy (HONEST PLACEHOLDER): the Interpreter cannot know true task success without the
arm reporting it. The default emits only what it is sure of — reflex -> -1 (a protective stop
is a bad outcome), AVOID veto -> -0.3, executed command -> 0.0 (neutral, await real sensing).
`--assume-success` adds +0.5 to executed commands for bootstrapping/demo. Replace
`reward_policy` with real task-success sensing once the arm provides it.

  signal_loop.py --serial COM3 --feedback | python interpreter.py --pipe --outcome loop_in
  python interpreter.py                    # run the self-check (no stdin)
"""
import sys
import json

# matched signature label -> concrete robot command (params are placeholders to tune)
COMMANDS = {
    "JOINT_A_ROTATE": "JOINT_A_ROTATE(+15deg)",
    "JOINT_B_ROTATE": "JOINT_B_ROTATE(-10deg)",
    "GRIPPER_CLOSE":  "GRIPPER_CLOSE",
    "GRIPPER_OPEN":   "GRIPPER_OPEN",
    "HOME":           "HOME",
}
REFLEX_CMD = {"STOP": "EMERGENCY_STOP", "WITHDRAW": "RETRACT_ALL"}


class Interpreter:
    def __init__(self, commands=None, assume_success=False):
        self.commands = COMMANDS if commands is None else commands
        self.assume_success = assume_success

    def interpret(self, ev):
        """One loop JSON event (dict) -> (command:str|None, reward:float|None).
        command goes to the controller; reward (if not None) goes back to the loop."""
        # 1. reflex preempts everything (the instinctive fast-path already fired)
        if "reflex" in ev:
            return REFLEX_CMD.get(ev["reflex"], "EMERGENCY_STOP"), -1.0
        # the loop's own outcome echoes are not ours to act on
        if "outcome" in ev:
            return None, None
        # 2. learned aversion vetoes the command
        if ev.get("instinct") == "AVOID":
            return "HOLD", -0.3
        # 3. recognition
        label = ev.get("match")
        if label is None:
            return None, 0.0                      # novel/ambiguous -> no command, no opinion
        cmd = self.commands.get(label)
        if cmd is None:
            return None, 0.0                      # known label, but no command mapped yet
        return cmd, (0.5 if self.assume_success else 0.0)


def run_pipe(itp, outcome_path=None):
    """Read loop JSON events from stdin -> robot commands to stdout, OUTCOME back-channel.
    To truly close the loop, connect the OUTCOME stream to the loop's stdin (FIFO/socket)."""
    def send_outcome(reward):
        line = f"OUTCOME {reward}"
        if outcome_path:
            with open(outcome_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            sys.stderr.write(line + "\n")         # default: stderr (tee/redirect to loop stdin)
            sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        cmd, reward = itp.interpret(ev)
        if cmd is not None:
            print(cmd); sys.stdout.flush()         # -> controller / servo drivers
        if reward is not None:
            send_outcome(reward)


def selfcheck():
    itp = Interpreter(assume_success=True)

    # ---- unit: each event class translates correctly --------------------------
    assert itp.interpret({"reflex": "STOP", "preempt": True}) == ("EMERGENCY_STOP", -1.0)
    assert itp.interpret({"reflex": "WITHDRAW"}) == ("RETRACT_ALL", -1.0)
    assert itp.interpret({"match": "GRIPPER_CLOSE", "instinct": None})[0] == "GRIPPER_CLOSE"
    assert itp.interpret({"match": "HOME", "instinct": "AVOID"}) == ("HOLD", -0.3)  # veto wins
    assert itp.interpret({"match": None})[0] is None                                 # novel -> no cmd
    assert itp.interpret({"match": "UNMAPPED_LABEL"})[0] is None                     # no mapping
    assert itp.interpret({"outcome": -1.0}) == (None, None)                          # echo ignored
    # reward policy: executed command gets +0.5 only with assume_success
    assert Interpreter().interpret({"match": "HOME"})[1] == 0.0
    assert Interpreter(assume_success=True).interpret({"match": "HOME"})[1] == 0.5

    # ---- integration: drive real loop events through the Interpreter ----------
    from snn_data_lake import signal_loop as S
    lib = S.build_default_library()
    matcher = S.build_matcher(lib)

    # a) the GRIPPER_CLOSE reference window should translate to its command
    ref_line = ",".join(f"{x:.4f}" for x in lib["GRIPPER_CLOSE"])
    events = []
    S.run_live(iter([ref_line]), lib, matcher, None,
               S.ValenceLearner(), S.Cortisol(), emit=events.append)
    cmds = [itp.interpret(e)[0] for e in events]
    assert "GRIPPER_CLOSE" in cmds, f"reference signal did not map to its command: {cmds}"

    # b) a collision through a reflex-enabled loop -> EMERGENCY_STOP + bad reward
    from snn_data_lake.reflex import Reflex
    danger = "500,500,500,500,500,500,950,0"
    ev2 = []
    S.run_live(iter([danger]), lib, matcher, Reflex(),
               S.ValenceLearner(), S.Cortisol(), emit=ev2.append)
    out2 = [itp.interpret(e) for e in ev2]
    assert ("EMERGENCY_STOP", -1.0) in out2, f"collision did not yield emergency stop: {out2}"

    print("=" * 58)
    print("INTERPRETER (matched signal -> robot command + outcome)")
    print("=" * 58)
    print(f"command map : {list(COMMANDS)}")
    print(f"reflex map  : {REFLEX_CMD}")
    print("priority    : reflex > AVOID-veto > match;  reward: reflex -1, veto -0.3, exec 0/+0.5")
    print("self-check OK: reflex preempts, AVOID vetoes, matches map to commands, "
          "reference signal -> GRIPPER_CLOSE, collision -> EMERGENCY_STOP")


def main():
    args = sys.argv[1:]
    if "--pipe" in args:
        outcome = args[args.index("--outcome") + 1] if "--outcome" in args else None
        run_pipe(Interpreter(assume_success="--assume-success" in args), outcome)
    else:
        selfcheck()


if __name__ == "__main__":
    main()
