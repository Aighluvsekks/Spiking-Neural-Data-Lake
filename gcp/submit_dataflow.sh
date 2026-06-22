#!/usr/bin/env bash
# Launch the streaming ingest on Dataflow (Pub/Sub -> GCS Bronze Parquet).
#   PROJECT=my-proj BUCKET=my-proj-snn-lake bash gcp/submit_dataflow.sh
# Needs: pip install 'apache-beam[gcp]' pyarrow
set -euo pipefail
: "${PROJECT:?set PROJECT}"
: "${BUCKET:?set BUCKET}"
REGION="${REGION:-us-central1}"
SUB="projects/$PROJECT/subscriptions/spike-telemetry-sub"

python gcp/dataflow_ingest.py \
  --runner=DataflowRunner \
  --project="$PROJECT" \
  --region="$REGION" \
  --temp_location="gs://$BUCKET/tmp" \
  --staging_location="gs://$BUCKET/staging" \
  --subscription="$SUB" \
  --bucket="$BUCKET" \
  --job_name="snn-stream-ingest"

echo "Dataflow streaming job launched (it runs until cancelled)."
echo "cancel:  gcloud dataflow jobs cancel <JOB_ID> --region=$REGION"
