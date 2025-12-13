# Grafana Dashboards

This folder stores versioned Grafana dashboards for the StarPU Inference Server and the related provisioning config.

## Structure

- `dashboards/`: place your Grafana JSON (e.g., `starpu-inference.json`).
- `provisioning/dashboards/dashboards.yml`: Grafana provider pointing to `dashboards/`.
- `provisioning/datasources/datasources.yml`: provisioned Prometheus datasource (default `http://localhost:9091`, `uid` = `prometheus`).

## Metrics prerequisites

- Host: Prometheus `node_exporter` for CPU, memory, network, disk (I/O and space).
- GPU: NVML/DCGM exporter (e.g., `nvml_exporter` or `dcgm-exporter`) for GPU metrics. Adjust dashboard queries if metric names differ (`nvidia_smi_*`, `nvml_*`, `DCGM_FI_*`, etc.).

### Run the base exporters

```bash
# Node exporter (host metrics)
docker run -d --name node-exporter --net=host prom/node-exporter:latest --path.rootfs=/host

# NVML/DCGM exporter (GPU metrics) – requires GPU access
docker run -d --name dcgm-exporter --net=host --gpus all nvidia/dcgm-exporter:latest
```

Prometheus scrape jobs to add (already present in `ops/prometheus/prometheus.yml`):
- `node_exporter` on `localhost:9100`
- `nvml_exporter` on `localhost:9400`

## Quick start with Grafana container

```bash
# From the repo root
docker run -d --name grafana -p 3000:3000 \
  -v "$(pwd)/ops/grafana/provisioning:/etc/grafana/provisioning" \
  -v "$(pwd)/ops/grafana/dashboards:/var/lib/grafana/dashboards" \
  grafana/grafana-oss:latest
```

- Provisioning auto-loads any `.json` under `ops/grafana/dashboards`.
- Change `path` in `provisioning/dashboards/dashboards.yml` if you mount dashboards elsewhere. Adjust the Prometheus URL in `provisioning/datasources/datasources.yml` if it is not `http://localhost:9091`.

## Manual import

If you prefer manual import, open Grafana → **Dashboards** → **Import** and upload the JSON from `ops/grafana/dashboards/`.
