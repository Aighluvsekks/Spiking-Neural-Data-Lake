"""
Test publisher — push spike events to the Pub/Sub topic to exercise the streaming path.
Uses this repo's synthetic telemetry as the source. Needs google-cloud-pubsub.

  PROJECT=my-proj python gcp/publish_spikes.py
"""
import json
import os
import sys

from google.cloud import pubsub_v1
from spike_telemetry_hub import synth   # synthetic spike source (this repo)


def main(project, topic="spike-telemetry"):
    pub = pubsub_v1.PublisherClient()
    topic_path = pub.topic_path(project, topic)
    hub = synth(64, 5000, 0.01, {7, 42}, (2000, 2200))   # small burst stream
    futures = []
    for c in range(hub.n):
        for t in hub.ch[c]:
            data = json.dumps({"t": int(t), "channel": int(c)}).encode("utf-8")
            futures.append(pub.publish(topic_path, data))
    for f in futures:
        f.result()
    print(f"published {hub.n_events():,} spike events to {topic_path}")


if __name__ == "__main__":
    proj = os.environ.get("PROJECT") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not proj:
        sys.exit("set PROJECT env or pass project id")
    main(proj)
