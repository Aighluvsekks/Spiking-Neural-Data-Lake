"""
Extracted code from "Conceptual Data Storage Method.docx" (section 3,
SpikingStorageCore blueprint), cleaned into a runnable module.

The docx mashed the source onto a few lines and mixed two snnTorch idioms
(init_hidden=True *and* manual mem threading). Fixed here to the correct
manual-state idiom (init_hidden=False) so `spk, mem = self.lif(cur, mem)`
works as written in the doc.

Run: pip install torch snntorch && python snn_storage_core_snntorch.py
This is the faithful extraction. It only does a LIF forward pass (an encoder);
it does NOT store/recall data. The actual storage demo is in
spiking_storage_prototype.py (stdlib-only, no deps).
"""
import torch
import torch.nn as nn
import snntorch as snn


# 1. Define a simple Spiking Storage Core architecture
class SpikingStorageCore(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Linear mapping to project data into structural weights
        self.fc = nn.Linear(input_dim, latent_dim)
        # Leaky Integrate-and-Fire neuron to handle temporal state (RAM emulation)
        self.lif = snn.Leaky(beta=0.9, init_hidden=False)  # doc had True; False to thread mem manually

    def forward(self, spike_train_in, steps):
        # Initialize the internal membrane potential state
        mem = self.lif.init_leaky()
        spike_record = []

        # Step through time to process temporal inputs
        for step in range(steps):
            cur = self.fc(spike_train_in[step])
            spk, mem = self.lif(cur, mem)
            spike_record.append(spk)

        return torch.stack(spike_record), mem


# This architecture converts a dense data stream into sparse temporal spike
# trains, utilizing structural weights for memory extraction.

if __name__ == "__main__":
    steps, input_dim, latent_dim = 25, 64, 32
    core = SpikingStorageCore(input_dim, latent_dim)
    # Poisson spike train: [steps, batch, input_dim] of {0,1}
    spike_train = (torch.rand(steps, 1, input_dim) < 0.2).float()
    spikes, mem = core(spike_train, steps)
    print(f"output spikes shape : {tuple(spikes.shape)}")
    print(f"mean spike rate     : {spikes.mean().item():.3f}  (sparsity {1 - spikes.mean().item():.1%})")
