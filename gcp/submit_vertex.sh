#!/usr/bin/env bash
# Build the training image (Cloud Build), push to Artifact Registry, run a Vertex AI
# custom GPU job: eth_mnist_bindsnet.py --gpu at 6400/60k (the ~95% config) on an L4.
#   PROJECT=my-proj bash gcp/submit_vertex.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
IMG="$REGION-docker.pkg.dev/$PROJECT/snn-images/snn-train:latest"

# build + push the image (uses gcp/Dockerfile, repo root as build context)
gcloud builds submit --project="$PROJECT" \
  --config=gcp/cloudbuild.yaml --substitutions=_IMG="$IMG" .

# launch the GPU training job at the 6400/60k TUNED config (the ~95% attempt — NOT the image's
# 400/20k default, and NOT 6400-with-defaults which measured 47.8%). Env override needs a config
# file (inline --worker-pool-spec can't set container env). Needs Vertex L4 training quota > 0.
CFG="$(mktemp).yaml"
cat > "$CFG" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: g2-standard-8
      acceleratorType: NVIDIA_L4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: $IMG
      env:
        - name: NORD_M
          value: "6400"
        - name: NORD_TRAIN
          value: "60000"
        - name: NORD_TEST
          value: "10000"
        - name: NORD_EPOCHS
          value: "3"
        - name: NORD_INH
          value: "250"
        - name: NORD_THETA_PLUS
          value: "0.2"
YAML
gcloud ai custom-jobs create \
  --project="$PROJECT" --region="$REGION" \
  --display-name="snn-6400-train" \
  --config="$CFG"

echo "submitted. stream logs with:"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=$REGION"
