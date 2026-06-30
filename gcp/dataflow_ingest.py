"""
Streaming ingest: Pub/Sub spike events -> windowed Parquet in GCS Bronze (Apache Beam /
Dataflow). The cloud version of "land data into Bronze" — continuous instead of a one-off
upload. Messages are JSON {"t": <int>, "channel": <int>}.

Run on Dataflow via gcp/submit_dataflow.sh. Needs apache-beam[gcp] + pyarrow.
"""
import argparse
import json
import sys
import os

import apache_beam as beam
import pyarrow as pa
from apache_beam.options.pipeline_options import PipelineOptions

# same data-quality gate the batch path (lakehouse/medallion.py) runs before Gold — streaming parity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_quality

_SCHEMA = pa.schema([pa.field("t", pa.int64()), pa.field("channel", pa.int32())])


def parse_event(msg):
    d = json.loads(msg.decode("utf-8"))
    return {"t": int(d["t"]), "channel": int(d["channel"])}


def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subscription", required=True,
                    help="projects/<PROJ>/subscriptions/spike-telemetry-sub")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--n-channels", type=int, default=64,
                    help="reject events whose channel is outside [0, n-channels)")
    args, beam_args = ap.parse_known_args(argv)

    opts = PipelineOptions(beam_args, streaming=True, save_main_session=True)
    with beam.Pipeline(options=opts) as p:
        (p
         | "ReadPubSub" >> beam.io.ReadFromPubSub(subscription=args.subscription)
         | "Parse" >> beam.Map(parse_event)
         | "Gate" >> beam.Map(data_quality.check_event, args.n_channels)   # block bad events pre-Bronze
         | "Window60s" >> beam.WindowInto(beam.window.FixedWindows(60))
         # ponytail: vanilla Parquet here = the small-file problem at high event rates. The
         # lakehouse-tier fix is a Delta sink (transaction log + OPTIMIZE compaction) — proven
         # locally in lakehouse/delta_demo.py; swap WriteToParquet for a Delta writer here.
         | "WriteBronze" >> beam.io.WriteToParquet(
             file_path_prefix=f"gs://{args.bucket}/bronze/stream",
             schema=_SCHEMA,
             file_name_suffix=".parquet"))


if __name__ == "__main__":
    run()
