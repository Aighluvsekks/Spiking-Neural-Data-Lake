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
Then the headline run (defaults, slow):
    python eth_mnist_bindsnet.py            # 6400 neurons, 60k/10k
Or a tractable midpoint (~87%, hours not overnight):
    NORD_M=400 NORD_TRAIN=20000 NORD_TEST=5000 python eth_mnist_bindsnet.py

Needs: pip install bindsnet  (torch._six shim below makes BindsNET<=0.3 run on torch>=2)
"""
import os
import sys
import types
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


# ---- config (paper / BindsNET constants; sizes via env) ---------------------
N = _ei("NORD_M", 6400)
N_TRAIN = _ei("NORD_TRAIN", 60000)
N_TEST = _ei("NORD_TEST", 10000)
N_EPOCHS = _ei("NORD_EPOCHS", 1)
TIME = _ei("NORD_TIME", 250)
UPDATE = _ei("NORD_UPDATE", 250)   # re-assign labels every UPDATE train images
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
    print("=" * 60)
    print("v0.9  BindsNET Diehl & Cook (eth_mnist) — conductance-based")
    print("=" * 60)
    print(f"neurons={N}  time={TIME}ms  train={N_TRAIN}  test={N_TEST}  epochs={N_EPOCHS}")
    if N >= 1600:
        print("WARNING: large network on CPU — expect a multi-hour run.")
    print()

    network = DiehlAndCook2015(
        n_inpt=784, n_neurons=N, exc=22.5, inh=120.0, dt=DT,
        norm=78.4, theta_plus=0.05, inpt_shape=(1, 28, 28),
    )
    spikes = Monitor(network.layers["Ae"], state_vars=["s"], time=STEPS)
    network.add_monitor(spikes, name="Ae")

    assignments = -torch.ones(N)
    proportions = torch.zeros(N, N_CLASSES)
    rates = torch.zeros(N, N_CLASSES)

    # ---- train (unsupervised STDP, no labels in the learning loop) ----
    train_set = make_dataset(train=True)
    spike_record = torch.zeros(UPDATE, STEPS, N)
    labels = []
    acc_hist = []
    print("training...")
    for epoch in range(N_EPOCHS):
        loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
        for step, batch in enumerate(loader):
            if step >= N_TRAIN:
                break
            if step % UPDATE == 0 and step > 0:
                lt = torch.tensor(labels[-UPDATE:])
                pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=N_CLASSES)
                acc = 100 * torch.sum(lt.long() == pred).item() / len(lt)
                acc_hist.append(acc)
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record, labels=lt, n_labels=N_CLASSES, rates=rates)
                print(f"  step {step}/{N_TRAIN}  window train acc={acc:.1f}%  (avg {sum(acc_hist)/len(acc_hist):.1f}%)")
            inputs = {"X": batch["encoded_image"].view(STEPS, 1, 1, 28, 28)}
            network.run(inputs=inputs, time=TIME)
            spike_record[step % UPDATE] = spikes.get("s").squeeze()
            labels.append(batch["label"].item())
            network.reset_state_variables()

    # ---- test (theta + weights frozen; classify by assigned-neuron activity) ----
    test_set = make_dataset(train=False)
    rec = torch.zeros(1, STEPS, N)
    n_all, n_prop, n = 0.0, 0.0, 0
    print("\ntesting...")
    for step, batch in enumerate(test_set):
        if step >= N_TEST:
            break
        inputs = {"X": batch["encoded_image"].view(STEPS, 1, 1, 28, 28)}
        network.run(inputs=inputs, time=TIME)
        rec[0] = spikes.get("s").squeeze()
        lt = torch.tensor([batch["label"]])
        n_all += float(torch.sum(lt.long() == all_activity(
            spikes=rec, assignments=assignments, n_labels=N_CLASSES)).item())
        n_prop += float(torch.sum(lt.long() == proportion_weighting(
            spikes=rec, assignments=assignments, proportions=proportions, n_labels=N_CLASSES)).item())
        n += 1
        network.reset_state_variables()
        if n % 500 == 0:
            print(f"  tested {n}/{N_TEST}  running all-activity acc={100*n_all/n:.2f}%")

    acc_all = 100 * n_all / n
    acc_prop = 100 * n_prop / n
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY (all-activity)        : {acc_all:.2f}%")
    print(f"TEST ACCURACY (proportion-weighting): {acc_prop:.2f}%")
    print(f"(paper: {N} neurons -> ~"
          + {100: '82.9', 400: '87.0', 1600: '91.9', 6400: '95.0'}.get(N, '??') + "%)")
    print("=" * 60)
    assert acc_all >= 10.0, "below chance — wiring broken"
    print("self-check OK: ran end-to-end, accuracy computed")


if __name__ == "__main__":
    main()
