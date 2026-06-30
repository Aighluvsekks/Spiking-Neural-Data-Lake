# Spiking Neural Data Lake — GCP-native infrastructure (Terraform).
# Stands up the storage + query + image-registry + identity layer. ETL (Dataproc
# Serverless) and training (Vertex AI) are job submissions, not standing resources —
# see gcp/submit_dataproc.sh and gcp/submit_vertex.sh.
#
#   cd infra && terraform init && terraform apply -var project_id=YOUR_PROJECT
#
# Costs: GCS + BigQuery + Artifact Registry are pay-per-use and near-zero at rest.
# Dataproc/Vertex cost only while a job runs.

terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- storage: the lake (Bronze/Silver/Gold live under prefixes) ---
resource "google_storage_bucket" "lake" {
  name                        = "${var.project_id}-snn-lake"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  versioning { enabled = true } # cheap "time travel" on the object store
}

# --- query: BigQuery dataset + a BigLake connection for Iceberg/external tables ---
resource "google_bigquery_dataset" "lake" {
  dataset_id = "snn_lake"
  location   = var.region
}

resource "google_bigquery_connection" "biglake" {
  connection_id = "biglake-snn"
  location      = var.region
  cloud_resource {}
}

# --- registry: holds the Vertex AI training image ---
resource "google_artifact_registry_repository" "images" {
  repository_id = "snn-images"
  location      = var.region
  format        = "DOCKER"
}

# --- identity: service account for Dataproc + Vertex jobs ---
resource "google_service_account" "lake_sa" {
  account_id   = "snn-lake-sa"
  display_name = "SNN Data Lake jobs"
}

# BigLake connection's managed SA needs to read the bucket's data
resource "google_storage_bucket_iam_member" "biglake_read" {
  bucket = google_storage_bucket.lake.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_bigquery_connection.biglake.cloud_resource[0].service_account_id}"
}

resource "google_storage_bucket_iam_member" "sa_rw" {
  bucket = google_storage_bucket.lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.lake_sa.email}"
}

# --- streaming ingest: Pub/Sub topic + subscription (Dataflow reads the sub) ---
resource "google_pubsub_topic" "spikes" {
  name = "spike-telemetry"
}

resource "google_pubsub_subscription" "spikes_sub" {
  name                       = "spike-telemetry-sub"
  topic                      = google_pubsub_topic.spikes.id
  ack_deadline_seconds       = 30
  message_retention_duration = "86400s"
}

# NOTE: Cloud Composer (Airflow) is a heavyweight, standing environment (~$300+/mo).
# Create it on demand rather than in this baseline:
#   gcloud composer environments create snn-orchestrator --location=$REGION --image-version=composer-2-airflow-2
# then upload gcp/composer_dag.py to the env's dags/ bucket.

output "bucket" { value = google_storage_bucket.lake.name }
output "dataset" { value = google_bigquery_dataset.lake.dataset_id }
output "connection" { value = google_bigquery_connection.biglake.connection_id }
output "image_repo" { value = google_artifact_registry_repository.images.repository_id }
output "service_acct" { value = google_service_account.lake_sa.email }
output "pubsub_topic" { value = google_pubsub_topic.spikes.name }
output "pubsub_sub" { value = google_pubsub_subscription.spikes_sub.name }
