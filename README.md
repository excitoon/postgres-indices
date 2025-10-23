# PostgreSQL Index Benchmark Demo

This project spins up two containers:
- Postgres 16 (`db`)
- Python 3.11 benchmark runner (`bench`)

They share volumes; the bench container writes directly to a bind-mounted folder so results appear on the host as the run progresses. The benchmark generates a source table with 30 columns and performs experiments across table sizes of 2^N rows, measuring:

1. Index storage as a percentage of table size
2. SELECT latency using an index (cold-ish and hot runs)
3. INSERT latency
4. SELECT latency without using an index (cold-ish and hot runs)
5. Table storage without indexes

Results are incrementally written to `results.json`.

## Config

Set via environment in `docker-compose.yml`:
- `MAX_N` (default 25): highest N, table rows = 2^N
- `M_MAX` (default 30): sweep M from 1..M_MAX (1 = PK only). Additional B-tree indexes created on `c01..c{M-1}`.
- `INSERT_BATCH` (default 1000): number of rows inserted for insert timing per N.
- `REPEATS` (default 3): target repeats for timing samples when operations are slow.
- `TIME_THRESHOLD_S` (default 5): if the first measured times for a group of operations are below this threshold, use `FEW_REPEATS` instead of `REPEATS`.
- `FEW_REPEATS` (default 2): reduced repeat count used when times are below `TIME_THRESHOLD_S`.

## Run

Quickest: use the provided helper script `run_benchmark.sh`.

Examples:

```bash
# Small smoke test with M sweep up to 5; writes results into ./results and tears down stack afterwards
M_MAX=5 ./run_benchmark.sh -n 4 -b 1000 -o ./results --down

# Larger run (be careful, 2^20 rows prepopulation); keep stack running after
M_MAX=8 REPEATS=5 TIME_THRESHOLD_S=5 FEW_REPEATS=2 ./run_benchmark.sh -n 20 -b 2000 -o ./results
```

Options:

- `-n MAX_N` highest N (rows = 2^N). Default 25
- Sweep for M is controlled via env `M_MAX` (default 30)
- `-b INSERT_BATCH` insert batch size for timing. Default 1000
- `-o OUT_DIR` host folder to copy results into. Default ./results
- `--down` bring down stack and remove volumes after run

Environment overrides: `MAX_N`, `M_MAX`, `INSERT_BATCH`, `OUT_DIR`, `BRING_DOWN=1`.

How results are written
- The runner binds your chosen OUT_DIR to `/bench` in the `bench` container, so files appear on the host immediately after each (N,M).
- Files:
  - `OUT_DIR/results.json` – compact tuples for plotting
  - `OUT_DIR/results_full.json` – full structured results

Or manually via docker compose:

```bash
docker compose build
docker compose up --remove-orphans
```

Watch the `bench` container logs for progress. Results will appear in `OUT_DIR` while the run is active; no copy step is required.

## Notes

- "Cold" timings are first-run approximations; true cold cache requires superuser privileges or OS cache control which are not applied here.
- The planner is nudged to use or avoid indexes via `enable_*` GUCs.
- The dataset is pseudo-random but correlated to reflect more realistic distributions.

## Commands

### Run

```bash
./run_benchmark.sh -n 25 -m 20 -r 1 -f 10 -b 10
```

### Fetch results

```bash
scp -r testdb:postgres-demo/results/'*'.json results/ && python3 plot.py
```

### Machine

Standard D4ads v6 (4 vcpus, 16 GiB memory).
