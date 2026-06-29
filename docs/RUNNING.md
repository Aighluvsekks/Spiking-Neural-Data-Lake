# Running the Spiking Neural Data Lake — Local & Cloud

Two ways to run everything: **local** (one machine, mostly zero-dependency) and **cloud**
(GCP-native, pay-per-use). Start local; the same data path scales to the cloud unchanged.

- Windows paths use `.venv\Scripts\python`; macOS/Linux use `.venv/bin/python`. Pick yours.
- The pure-Python core needs **nothing** but Python 3.11+. Only the lakehouse / real-data /
  GPU pieces need extra packages, each in its own throwaway venv so they don't collide.

---

## Local

### 0. Get it
```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
python --version          # need 3.11+
```

### 1. Zero-dependency core (no install)
Every stdlib module runs standalone with an assert-based self-check — the fastest way to see
each piece work:
```bash
python spiking_storage_prototype.py   # factored associative-memory storage
python spike_telemetry_hub.py         # Paradigm A: sparse .spk store + windowed queries
python paradigm_b_engine.py           # Paradigm B: in-storage spike-query engine
python spike_knowledge_graph.py       # Paradigm C: relational spike embeddings
```
Run the whole suite (what CI runs):
```bash
for f in spiking_storage_prototype test_prototype snn_classifier snn_moe_classifier \
         temporal_coding_storage spike_telemetry_hub streaming_hub paradigm_b_matcher \
         paradigm_b_engine spike_preprocessing spike_knowledge_graph \
         spike_knowledge_graph_rotate spike_kg_relations nmnist_ingest signal_loop \
         learned_matcher reflex valence_stdp cortisol interpreter closed_loop \
         make_sensor_dataset sensor_demo; do python "$f.py" || break; done
```

### 2. The robot-arm closed loop (the application)
Encode → data lake → match → command → reward → learn (with reflex + RPE dopamine + cortisol +
the Interpreter), all zero-dep, on `main`:
```bash
python closed_loop.py        # full stack end-to-end: gesture -> command, collision -> STOP
python signal_loop.py        # the loop alone (hybrid matcher) + self-check
python interpreter.py        # matched label -> robot command + self-check
# drive it with no hardware (outcomes back via OUTCOME lines):
cat windows.csv | python signal_loop.py --stdin --feedback --reflex
```

### 3. The Medallion lakehouse PoC (needs polars)
Bronze → Silver → Gold over Parquet, the data path that scales to the cloud:
```bash
python -m venv .venv-lake
.venv-lake/bin/pip install polars pyarrow          # Windows: .venv-lake\Scripts\pip
.venv-lake/bin/python lakehouse/medallion.py
# writes lakehouse/data/{bronze,silver,gold}.parquet + prints ICR, synchrony, SQL query
```
`lakehouse/data/bronze.parquet` is exactly what you upload to the cloud Bronze (below).

### 4. Real event-camera data — N-MNIST (optional, needs tonic)
```bash
python -m venv .venv-nmnist
.venv-nmnist/bin/pip install tonic                 # pulls ~1 GB N-MNIST on first run
.venv-nmnist/bin/python nmnist_ingest.py
# no tonic? `python nmnist_ingest.py` runs on synthetic events (zero-dep)
```

### 5. GPU training (optional, needs CUDA + torch)
For the literature ~95% run (6400 neurons). RTX 50-series (Blackwell) needs CUDA 12.8:
```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
.venv/bin/pip install snntorch bindsnet numpy
NORD_M=1600 NORD_INH=60 NORD_THETA_PLUS=0.20 .venv/bin/python eth_mnist_bindsnet.py --gpu
# CPU fallback works automatically (slower); 400/20k ~ 86% in minutes
```

### 6. Browse the knowledge graph (optional)
```bash
# open graphify-out/graph.html in a browser, or:
graphify query "how does the matcher reject novel signals?"
```

---

## Cloud (GCP-native)

Takes the **same Medallion data path** to a managed lakehouse: **GCS + BigLake/Iceberg +
BigQuery + Dataproc Serverless + Vertex AI + Pub/Sub/Dataflow + Composer**. Pay-per-use,
near-zero at rest. Full step-by-step is in [`gcp/README.md`](../gcp/README.md); the short
version:

```bash
# 0. project + APIs
gcloud auth login && gcloud config set project snn-data-lake-prod
gcloud services enable storage bigquery dataproc aiplatform artifactregistry cloudbuild \
  bigqueryconnection --project snn-data-lake-prod   # (full names in gcp/README.md)

# 1. infra (GCS bucket, BQ dataset, BigLake conn, Artifact Registry, SA, Pub/Sub)
cd infra && terraform init && terraform apply -var project_id=snn-data-lake-prod
export PROJECT=snn-data-lake-prod BUCKET=$(terraform output -raw bucket); cd ..

# 2. land the locally-produced Bronze (from lakehouse/medallion.py)
gcloud storage cp lakehouse/data/bronze.parquet gs://$BUCKET/bronze/

# 3. Medallion ETL on Dataproc Serverless (managed Spark, no cluster)
PROJECT=$PROJECT BUCKET=$BUCKET bash gcp/submit_dataproc.sh        # -> silver/ + gold/

# 4. query Gold via BigQuery over BigLake  (one-time EXTERNAL TABLE in gcp/README.md)
bq query --use_legacy_sql=false \
  'SELECT channel, rate, synchrony_cv FROM snn_lake.gold ORDER BY rate DESC LIMIT 5'

# 5. GPU training on Vertex AI (the 6400/60k ~95% run, impractical on CPU)
PROJECT=$PROJECT bash gcp/submit_vertex.sh

# 6. streaming ingest (Pub/Sub -> Dataflow -> GCS Bronze)
PROJECT=$PROJECT BUCKET=$BUCKET bash gcp/submit_dataflow.sh
PROJECT=$PROJECT python gcp/publish_spikes.py        # feed test spike events

# 7. daily orchestration (Cloud Composer / Airflow) — optional, heavyweight
gcloud storage cp gcp/dataproc_medallion.py gs://$BUCKET/code/
# create env + import gcp/composer_dag.py  (see gcp/README.md)
```

### Local → cloud mapping
| Local | Cloud (GCP) |
|-------|-------------|
| `lakehouse/medallion.py` (polars) | `gcp/dataproc_medallion.py` (PySpark on Dataproc Serverless) |
| `lakehouse/data/*.parquet` | `gs://$BUCKET/{bronze,silver,gold}/` |
| polars SQL over Parquet | BigQuery over BigLake/Iceberg |
| `eth_mnist_bindsnet.py --gpu` | Vertex AI custom job (`submit_vertex.sh`) |
| `signal_loop` stdin/serial ingest | Pub/Sub + Dataflow (`dataflow_ingest.py`) |
| run scripts by hand | Cloud Composer DAG `snn_medallion` |

### Cost
GCS / BigQuery / Artifact Registry ≈ free at rest. Dataproc Serverless bills per batch.
Vertex L4 ≈ $0.7–1/hr (the 6400/60k run is a few dollars). Composer ≈ $300+/mo — only if you
need always-on orchestration. **Don't leave jobs running.** Nothing here provisions your cloud
automatically; you run each step with your own auth/billing.

---

## Which do I use?
- **Just trying it / developing / the robot-arm loop** → local, sections 1–2 (zero install).
- **Lakehouse analytics on one machine** → local section 3.
- **Petabyte scale, streaming, the ~95% GPU run, sharing** → cloud.
The data path (Bronze → Silver → Gold) is identical; only the engine under it changes.
```
