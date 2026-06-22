"""
Streaming ingest: Pub/Sub spike events -> windowed Parquet in GCS Bronze (Apache Beam /
Dataflow). The cloud version of "land data into Bronze" — continuous instead of a one-off
upload. Messages are JSON {"t": <int>, "channel": <int>}.

Run on Dataflow via gcp/submit_dataflow.sh. Needs apache-beam[gcp] + pyarrow.
"""
import argparse
import json

import apache_beam as beam
import pyarrow as pa
from apache_beam.options.pipeline_options import PipelineOptions

_SCHEMA = pa.schema([pa.field("t", pa.int64()), pa.field("channel", pa.int32())])


def parse_event(msg):
    d = json.loads(msg.decode("utf-8"))
    return {"t": int(d["t"]), "channel": int(d["channel"])}


def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subscription", required=True,
                    help="projects/<PROJ>/subscriptions/spike-telemetry-sub")
    ap.add_argument("--bucket", required=True)
    args, beam_args = ap.parse_known_args(argv)

    opts = PipelineOptions(beam_args, streaming=True, save_main_session=True)
    with beam.Pipeline(options=opts) as p:
        (p
         | "ReadPubSub" >> beam.io.ReadFromPubSub(subscription=args.subscription)
         | "Parse" >> beam.Map(parse_event)
         | "Window60s" >> beam.WindowInto(beam.window.FixedWindows(60))
         | "WriteBronze" >> beam.io.WriteToParquet(
             file_path_prefix=f"gs://{args.bucket}/bronze/stream",
             schema=_SCHEMA,
             file_name_suffix=".parquet"))


if __name__ == "__main__":
    run()
