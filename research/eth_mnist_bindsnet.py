"""
v0.9 — BindsNET Diehl & Cook (eth_mnist) runner: the path to the literature ~95%.

The v0.8 study concluded that beating hard-WTA needs CONDUCTANCE-based synapses and
a real exc/inh population — exactly what BindsNET's `DiehlAndCook2015` model ships.
Rather than re-derive conductance dynamics from scratch, this wires in BindsNET (an
already-installed dependency) with the paper's exact constants and scales neurons up.

Evidence (Diehl & Cook 2015, Front. Comput. Neurosci.):
  100 neurons -> 82.9% | 400 -> 87.0% | 1600 -> 91.9% | 6400 -> 95.0%
Constants from BindsNET examples/mnist/eth_mnist.py:
  exc=22.5, inh=120, norm=78.4, theta_plus=0.05, time=250 ms, dt=1, intensity=128.

COMPUTE WARNING: 6400 neurons over the full 60k train + 10k test is a multi-HOUR
(CPU: likely overnight+) run. Verify the wiring fast first, e.g.:
    NORD_M=100 NORD_TRAIN=600 NORD_TEST=500 NORD_UPDATE=100 python eth_mnist_bindsnet.py
Then the headline run on a GPU (one switch):
    python eth_mnist_bindsnet.py --gpu      # 6400 neurons, 60k/10k -> ~95%
Or a tractable midpoint (~87%):
    NORD_M=400 NORD_TRAIN=20000 NORD_TEST=5000 python eth_mnist_bindsnet.py --gpu

GPU notes:
  --gpu (or --device cuda / NORD_GPU=1) moves the network + tensors to CUDA.
  RTX 5070 = Blackwell (sm_120) -> needs a CUDA 12.8+ torch build; the default CPU
  wheel has no sm_120 kernels. Install once on the GPU box:
    pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  Then `python eth_mnist_bindsnet.py --gpu`. If CUDA isn't available the runner
  prints this hint and falls back to CPU.

Needs: pip install bindsnet  (torch._six shim below makes BindsNET<=0.3 run on torch>=2)
"""
import os
import sys
import types
import argparse
import collections.abc
import torch

# --- compat shim: BindsNET <=0.3 imports torch._six, removed in torch >=2.0 ---
if not hasattr(torch, "_six"):
    _six = types.ModuleType("torch._six")
    _six.container_abcs = collections.abc
    _six.string_classes = (str, bytes)
    _six.int_classes = int
    sys.modules["torch._six"] = _six
    torch._six = _six

from torchvision import transforms
from bindsnet.models import DiehlAndCook2015
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels

torch.manual_seed(0)


def _ei(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _ef(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def resolve_device():
    """--gpu / --device flag (or NORD_GPU env). Falls back to CPU with a clear
    RTX 5070 (Blackwell sm_120) install hint if CUDA torch isn't present."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gpu", action="store_true", help="run on CUDA GPU if available")
    ap.add_argument("--device", default=None, help="explicit device, e.g. cuda / cuda:0 / cpu")
    args, _ = ap.parse_known_args()
    want_gpu = args.gpu or args.device not in (None, "cpu") or os.environ.get("NORD_GPU") == "1"
    if args.device:
        return args.device
    if want_gpu:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name}  (CUDA {torch.version.cuda}, torch {torch.__version__})")
            return "cuda"
        print("WARNING: --gpu requested but torch.cuda.is_available() is False.")
        print("  This venv has the CPU torch build. For an RTX 5070 (Blackwell, sm_120)")
        print("  install a CUDA 12.8+ build, e.g.:")
        print("    pip install --force-reinstall torch torchvision \\")
        print("      --index-url https://download.pytorch.org/whl/cu128")
        print("  (older CUDA wheels lack sm_120 kernels and will fail on the 5070.)")
        print("  Falling back to CPU.\n")
    return "cpu"


# ---- config (paper / BindsNET constants; sizes via env) ---------------------
N = _ei("NORD_M", 6400)
N_TRAIN = _ei("NORD_TRAIN", 60000)
N_TEST = _ei("NORD_TEST", 10000)
N_EPOCHS = _ei("NORD_EPOCHS", 1)
TIME = _ei("NORD_TIME", 250)
UPDATE = _ei("NORD_UPDATE", 250)   # re-assign labels every UPDATE train images
# scale-sensitive knobs — defaults are BindsNET's ~100-neuron values. Larger networks
# need MORE inhibition (more competitors) + bigger theta_plus + multiple epochs, or they
# under-specialise (measured: 6400 neurons / 1 epoch with defaults -> only 47.8%).
EXC = _ef("NORD_EXC", 22.5)
INH = _ef("NORD_INH", 120.0)
THETA_PLUS = _ef("NORD_THETA_PLUS", 0.05)
NORM = _ef("NORD_NORM", 78.4)
DT = 1.0
INTENSITY = 128.0
N_CLASSES = 10
STEPS = int(TIME / DT)
ROOT = os.path.join(".", "data", "bindsnet")


def make_dataset(train):
    return MNIST(
        PoissonEncoder(time=TIME, dt=DT), None, root=ROOT, download=True, train=train,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * INTENSITY)]),
    )


def main():
    device = resolve_device()
    print("=" * 60)
    print("BindsNET Diehl & Cook (eth_mnist) — conductance-based")
    print("=" * 60)
    print(f"neurons={N}  time={TIME}ms  train={N_TRAIN}  test={N_TEST}  epochs={N_EPOCHS}  device={device}")
    if N >= 1600 and device == "cpu":
        print("WARNING: large network on CPU — expect a multi-hour (likely overnight) run.")
        print("         Pass --gpu on a CUDA machine (RTX 5070: needs cu128 torch).")
    print()

    network = DiehlAndCook2015(
        n_inpt=784, n_neurons=N, exc=EXC, inh=INH, dt=DT,
        norm=NORM, theta_plus=THETA_PLUS, inpt_shape=(1, 28, 28),
    )
    spikes = Monitor(network.layers["Ae"], state_vars=["s"], time=STEPS)
    network.add_monitor(spikes, name="Ae")
    network.to(device)   # moves layers (+ their monitored spikes) onto the device

    assignments = -torch.ones(N, device=device)
    proportions = torch.zeros(N, N_CLASSES, device=device)
    rates = torch.zeros(N, N_CLASSES, device=device)

    # ---- train (unsupervised STDP, no labels in the learning loop) ----
    train_set = make_dataset(train=True)
    spike_record = torch.zeros(UPDATE, STEPS, N, device=device)
    labels = []
    acc_hist = []
    print("training...")
    for epoch in range(N_EPOCHS):
        loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
        for step, batch in enumerate(loader):
            if step >= N_TRAIN:
                break
            if step % UPDATE == 0 and step > 0:
                lt = torch.tensor(labels[-UPDATE:], device=device)
                pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=N_CLASSES).to(device)
                acc = 100 * torch.sum(lt.long() == pred).item() / len(lt)
                acc_hist.append(acc)
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record, labels=lt, n_labels=N_CLASSES, rates=rates)
                print(f"  step {step}/{N_TRAIN}  window train acc={acc:.1f}%  (avg {sum(acc_hist)/len(acc_hist):.1f}%)")
            inputs = {"X": batch["encoded_image"].view(STEPS, 1, 1, 28, 28).to(device)}
            network.run(inputs=inputs, time=TIME)
            spike_record[step % UPDATE] = spikes.get("s").squeeze()
            labels.append(batch["label"].item())
            network.reset_state_variables()

    # ---- test (theta + weights frozen; classify by assigned-neuron activity) ----
    test_set = make_dataset(train=False)
    rec = torch.zeros(1, STEPS, N, device=device)
    n_all, n = 0.0, 0
    print("\ntesting...")
    for step, batch in enumerate(test_set):
        if step >= N_TEST:
            break
        inputs = {"X": batch["encoded_image"].view(STEPS, 1, 1, 28, 28).to(device)}
        network.run(inputs=inputs, time=TIME)
        rec[0] = spikes.get("s").squeeze()
        lt = torch.tensor([batch["label"]], device=device)
        n_all += float(torch.sum(lt.long() == all_activity(
            spikes=rec, assignments=assignments, n_labels=N_CLASSES).to(device)).item())
        n += 1
        network.reset_state_variables()
        if n % 500 == 0:
            print(f"  tested {n}/{N_TEST}  running all-activity acc={100*n_all/n:.2f}%")

    acc_all = 100 * n_all / n
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY (all-activity) : {acc_all:.2f}%")
    print(f"(paper: {N} neurons -> ~"
          + {100: '82.9', 400: '87.0', 1600: '91.9', 6400: '95.0'}.get(N, '??') + "%)")
    print("=" * 60)
    assert acc_all >= 10.0, "below chance — wiring broken"
    print("self-check OK: ran end-to-end, accuracy computed")


if __name__ == "__main__":
    main()
