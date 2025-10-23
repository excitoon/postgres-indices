import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from statistics import median

import numpy as np
import psycopg
from psycopg.rows import dict_row

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "bench")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
SCHEMA = os.getenv("SCHEMA", "public")
MAX_N = int(os.getenv("MAX_N", "25"))
M_MAX = int(os.getenv("M_MAX", "30"))
INSERT_BATCH = int(os.getenv("INSERT_BATCH", "1000"))
RESULTS_PATH = "/bench/results.json"
REPEATS = int(os.getenv("REPEATS", "3"))
# If first measured time is small (< TIME_THRESHOLD_S), only do FEW_REPEATS to save time
TIME_THRESHOLD_S = float(os.getenv("TIME_THRESHOLD_S", "5"))
FEW_REPEATS = int(os.getenv("FEW_REPEATS", "2"))
NOIDX_RANDOM_PAGE_COST = float(os.getenv("NOIDX_RANDOM_PAGE_COST", "1000000"))

# Contract
# Inputs: MAX_N (1..25), M (1..30), INSERT_BATCH
# Outputs: JSON list of result tuples for each N, both indexed and no-index runs

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS source_data (
    id INT PRIMARY KEY,
    c01 INTEGER,
    c02 INTEGER,
    c03 INTEGER,
    c04 INTEGER,
    c05 INTEGER,
    c06 INTEGER,
    c07 INTEGER,
    c08 INTEGER,
    c09 INTEGER,
    c10 INTEGER,
    c11 INTEGER,
    c12 INTEGER,
    c13 INTEGER,
    c14 INTEGER,
    c15 INTEGER,
    c16 INTEGER,
    c17 INTEGER,
    c18 INTEGER,
    c19 INTEGER,
    c20 INTEGER,
    c21 TEXT,
    c22 TEXT,
    c23 TEXT,
    c24 TEXT,
    c25 NUMERIC(12,2),
    c26 NUMERIC(12,2),
    c27 TIMESTAMP WITHOUT TIME ZONE,
    c28 DATE,
    c29 BOOLEAN,
    c30 BYTEA
);
"""

# moderately randomized but with correlations/patterns so not fully random
# We'll generate using SQL for speed, combining generate_series with patterns

CREATE_FILL_FUNCTION = """
CREATE OR REPLACE FUNCTION fill_source_data(start_id bigint, n_rows bigint) RETURNS void AS $$
BEGIN
    INSERT INTO source_data (
        id,
        c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,
        c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
        c21,c22,c23,c24,c25,c26,c27,c28,c29,c30
    )
    SELECT
        gs AS id,
        mod(gs, 1000),                   -- c01 small cardinality
        mod((gs / 10), 10000),           -- c02 correlated range
        mod((gs * 17), 500000),          -- c03 permuted
        mod((gs * 37 + 13), 1000000),    -- c04 permuted different
        mod((gs >> 3), 20000),           -- c05 power-of-two pattern
        mod(gs, 2),                      -- c06 boolean-like int
        mod(gs, 7),                      -- c07 week-like
        mod(gs, 24),                     -- c08 hour-like
        mod(gs, 60),                     -- c09 minute-like
        mod(gs, 100),                    -- c10 bucket
        mod((gs * 3), 100000),           -- c11
        mod((gs * 5 + 1), 75000),        -- c12
        mod((gs * 7 + 3), 50000),        -- c13
        mod((gs * 11 + 5), 25000),       -- c14
        mod((gs * 13 + 7), 12500),       -- c15
        mod((gs * 17 + 9), 8000),        -- c16
        mod((gs * 19 + 11), 4000),       -- c17
        mod((gs * 23 + 13), 2000),       -- c18
        mod((gs * 29 + 15), 1000),       -- c19
        mod((gs * 31 + 17), 500),        -- c20
        md5(gs::text),               -- c21 pseudo text
        md5((gs*7)::text),           -- c22 pseudo text different
        left(md5((gs*13)::text), 16),-- c23 shorter text
        left(md5((gs*17)::text), 8), -- c24 shorter text
        (mod(gs, 100000)) / 3.0,         -- c25 numeric
        (mod(gs, 50000)) / 7.0,          -- c26 numeric
        to_timestamp(1700000000 + mod(gs, 1000000)), -- c27 recent timestamps
        date '2020-01-01' + (mod(gs, 2000))::int,    -- c28 dates
        (mod(gs, 2)) = 0,                -- c29 boolean
        decode(left(md5(gs::text), 16), 'hex')     -- c30 bytea
    FROM generate_series(start_id, start_id + n_rows - 1) AS gs;
END;
$$ LANGUAGE plpgsql;
"""

DROP_BENCH_TABLE = "DROP TABLE IF EXISTS bench_data;"

CREATE_BENCH_TABLE = """
-- Copy structure and constraints (including PK) but not indexes beyond PK are created here; additional indexes added explicitly
CREATE TABLE bench_data (
    LIKE source_data INCLUDING INDEXES INCLUDING DEFAULTS INCLUDING GENERATED INCLUDING IDENTITY INCLUDING CONSTRAINTS INCLUDING STORAGE INCLUDING COMPRESSION
);
"""

CREATE_BENCH_TABLE_NOIDX = """
-- Create a copy of structure WITHOUT constraints/indexes to ensure a pure heap table for no-index measurements
CREATE TABLE bench_data_noidx (
    LIKE source_data INCLUDING DEFAULTS INCLUDING GENERATED INCLUDING IDENTITY INCLUDING STORAGE INCLUDING COMPRESSION
);
"""

ANALYZE_TABLE = "ANALYZE VERBOSE %s;"

@dataclass
class Timings:
    select_cold_ms: float
    select_hot_ms: float
    insert_cold_ms: float
    insert_hot_ms: float

@dataclass
class Sizes:
    table_bytes: int
    index_bytes: int

@dataclass
class Result:
    N: int
    rows: int
    M: int
    with_indexes: Dict
    without_indexes: Dict

@contextmanager
def connect():
    with psycopg.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        autocommit=True,
    ) as conn:
        yield conn


def exec_sql(cur, sql: str, params: Tuple = None):
    if params is None:
        cur.execute(sql)
    else:
        cur.execute(sql, params)


def size_of_table(cur, relname: str) -> Tuple[int, int]:
    # returns (table_bytes, index_bytes)
    cur.execute(
        """
        SELECT
          COALESCE(pg_table_size(oid),0) AS table_bytes,
          COALESCE(pg_indexes_size(oid),0) AS index_bytes
        FROM pg_class
        WHERE relname = %s AND relkind = 'r'
        LIMIT 1;
        """,
        (relname,),
    )
    row = cur.fetchone()
    if row is None:
        return 0, 0
    return row[0], row[1]


def time_ms(fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def ensure_baseline(conn):
    with conn.cursor() as cur:
        exec_sql(cur, f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
        exec_sql(cur, DDL_TABLE)
        exec_sql(cur, CREATE_FILL_FUNCTION)

        # Determine current max rows and fill up to 2^MAX_N if needed
        cur.execute("SELECT COALESCE(MAX(id),0) FROM source_data;")
        current = cur.fetchone()[0]
        target = 2 ** MAX_N
        to_add = max(0, target - current)
        if to_add > 0:
            start_id = current + 1
            print(f"Populating source_data with {to_add} rows (target {target}), starting at id={start_id}...")
            exec_sql(cur, "SELECT fill_source_data(%s, %s);", (start_id, to_add))
            exec_sql(cur, "VACUUM ANALYZE source_data;")


def reset_bench_tables(conn):
    with conn.cursor() as cur:
        exec_sql(cur, DROP_BENCH_TABLE)
        exec_sql(cur, "DROP TABLE IF EXISTS bench_data_noidx;")
        exec_sql(cur, CREATE_BENCH_TABLE)
        exec_sql(cur, CREATE_BENCH_TABLE_NOIDX)


def copy_rows(cur, rows: int):
    exec_sql(cur, "TRUNCATE bench_data, bench_data_noidx;")
    exec_sql(cur, "INSERT INTO bench_data SELECT * FROM source_data ORDER BY id LIMIT %s;", (rows,))
    exec_sql(cur, "INSERT INTO bench_data_noidx SELECT * FROM source_data ORDER BY id LIMIT %s;", (rows,))
    exec_sql(cur, "VACUUM ANALYZE bench_data;")
    exec_sql(cur, "VACUUM ANALYZE bench_data_noidx;")


def add_indexes(cur, m: int):
    # Create B-tree indexes on the first m columns including PK (id). PK already indexed; we create on c01.. up to m-1 columns
    m = max(1, min(m, 30))
    # Ensure PK exists (already from BIGSERIAL primary key).
    # Create indexes starting from c01 up to c{m-1}
    for i in range(1, m):
        col = f"c{i:02d}"
        exec_sql(cur, f"CREATE INDEX IF NOT EXISTS idx_bench_{col} ON bench_data USING btree ({col});")
    exec_sql(cur, "VACUUM ANALYZE bench_data;")


def drop_indexes(cur):
    # Drop non-PK indexes on bench_data
    cur.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = current_schema() AND tablename = 'bench_data' AND indexname NOT LIKE '%_pkey';
    """)
    for (iname,) in cur.fetchall():
        exec_sql(cur, f"DROP INDEX IF EXISTS {iname};")
    exec_sql(cur, "VACUUM ANALYZE bench_data;")


def measure_select_pk_range(cur, table: str, with_index: bool, return_explain: bool = False) -> Tuple[float, float, str | None]:
    # Use a highly selective predicate based on PK id to guarantee small row counts and index use when available.
    # We'll select 10 consecutive ids around the midpoint.
    exec_sql(cur, f"SELECT COALESCE(MAX(id),0) FROM {table};")
    max_id = cur.fetchone()[0] or 0
    if max_id <= 10:
        low = 1
    else:
        low = max(1, max_id // 2)
    high = min(max_id, low + 9)

    if with_index:
        exec_sql(cur, "SET enable_seqscan TO off;")
        exec_sql(cur, "SET enable_indexscan TO on; SET enable_bitmapscan TO on; SET enable_indexonlyscan TO on;")
    else:
        exec_sql(cur, "SET enable_seqscan TO on;")
        exec_sql(cur, "SET enable_indexscan TO off; SET enable_bitmapscan TO off; SET enable_indexonlyscan TO off;")
        #exec_sql(cur, f"SET random_page_cost TO {NOIDX_RANDOM_PAGE_COST};")

    # Build SQL and params to reuse for EXPLAIN and timing
    sql = f"SELECT COUNT(*) FROM {table} WHERE id = {low}::int4;"
    params = tuple()#(low, high)

    explain_text = None
    if return_explain:
        exec_sql(cur, f"EXPLAIN (VERBOSE, FORMAT TEXT) {sql}", params)
        explain_lines = [r[0] for r in cur.fetchall()]
        explain_text = "\n".join(explain_lines)

    def run_query():
        exec_sql(cur, sql, params)
        cur.fetchone()

    t_cold = time_ms(run_query)
    t_hot = time_ms(run_query)

    exec_sql(cur, "RESET ALL;")
    return t_cold, t_hot, explain_text


def measure_insert_cold_hot(cur, rows: int, table: str) -> Tuple[float, float]:
    # Measure two consecutive insert batches into the specified table; then restore table to original size.
    batch = INSERT_BATCH

    def insert_batch():
        # Determine next id to insert and generate explicit ids
        exec_sql(cur, f"SELECT COALESCE(MAX(id),0) + 1 FROM {table};")
        start_id = cur.fetchone()[0]
        hi_id = start_id + batch - 1
        sql = f"""
                        INSERT INTO {table} (
                            id,
                            c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,
                            c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
                            c21,c22,c23,c24,c25,c26,c27,c28,c29,c30
                        )
                        SELECT
                            gs AS id,
                            mod((gs - %s + 1), 1000),
                            mod(((gs - %s + 1) / 10), 10000),
                            mod(((gs - %s + 1) * 17), 500000),
                            mod(((gs - %s + 1) * 37 + 13), 1000000),
                            mod(((gs - %s + 1) >> 3), 20000),
                            mod((gs - %s + 1), 2),
                            mod((gs - %s + 1), 7),
                            mod((gs - %s + 1), 24),
                            mod((gs - %s + 1), 60),
                            mod((gs - %s + 1), 100),
                            mod(((gs - %s + 1) * 3), 100000),
                            mod(((gs - %s + 1) * 5 + 1), 75000),
                            mod(((gs - %s + 1) * 7 + 3), 50000),
                            mod(((gs - %s + 1) * 11 + 5), 25000),
                            mod(((gs - %s + 1) * 13 + 7), 12500),
                            mod(((gs - %s + 1) * 17 + 9), 8000),
                            mod(((gs - %s + 1) * 19 + 11), 4000),
                            mod(((gs - %s + 1) * 23 + 13), 2000),
                            mod(((gs - %s + 1) * 29 + 15), 1000),
                            mod(((gs - %s + 1) * 31 + 17), 500),
                            md5((gs - %s + 1)::text),
                            md5(((gs - %s + 1)*7)::text),
                            left(md5(((gs - %s + 1)*13)::text), 16),
                            left(md5(((gs - %s + 1)*17)::text), 8),
                            (mod((gs - %s + 1), 100000)) / 3.0,
                            (mod((gs - %s + 1), 50000)) / 7.0,
                            to_timestamp(1700000000 + (mod((gs - %s + 1), 1000000))),
                            date '2020-01-01' + ((mod((gs - %s + 1), 2000)))::int,
                            (mod((gs - %s + 1), 2)) = 0,
                            decode(left(md5((gs - %s + 1)::text), 16), 'hex')
            FROM generate_series(%s::int, %s::int) AS gs;
            """
        # 30 occurrences of start_id for column expressions + 2 bounds for generate_series
        params = tuple([start_id] * 30 + [start_id, hi_id])
        exec_sql(cur, sql, params)

    cold_ms = time_ms(insert_batch)
    hot_ms = time_ms(insert_batch)

    # Restore state: keep exactly `rows` rows
    exec_sql(cur, f"TRUNCATE {table};")
    exec_sql(cur, f"INSERT INTO {table} SELECT * FROM source_data ORDER BY id LIMIT %s;", (rows,))
    exec_sql(cur, f"VACUUM ANALYZE {table};")
    return cold_ms, hot_ms


def _setup_bench_with_idx(cur, rows: int, m_idx: int):
    exec_sql(cur, "DROP TABLE IF EXISTS bench_data;")
    exec_sql(cur, CREATE_BENCH_TABLE)
    exec_sql(cur, "TRUNCATE bench_data;")
    exec_sql(cur, "INSERT INTO bench_data SELECT * FROM source_data ORDER BY id LIMIT %s;", (rows,))
    exec_sql(cur, "VACUUM ANALYZE bench_data;")
    drop_indexes(cur)
    add_indexes(cur, m_idx)


def _setup_bench_noidx(cur, rows: int):
    exec_sql(cur, "DROP TABLE IF EXISTS bench_data_noidx;")
    exec_sql(cur, CREATE_BENCH_TABLE_NOIDX)
    exec_sql(cur, "TRUNCATE bench_data_noidx;")
    exec_sql(cur, "INSERT INTO bench_data_noidx SELECT * FROM source_data ORDER BY id LIMIT %s;", (rows,))
    exec_sql(cur, "VACUUM ANALYZE bench_data_noidx;")


def run_for_N(conn, N: int, M_idx: int) -> Dict:
    rows = 2 ** N
    with conn.cursor() as cur:
        # Indexed run: compute sizes once, then take repeated timing samples (recreate each time)
        _setup_bench_with_idx(cur, rows, M_idx)
        t_table_bytes, t_index_bytes = size_of_table(cur, 'bench_data')
        exec_sql(cur, "DROP TABLE IF EXISTS bench_data;")

        sel_cold_samples_w: List[float] = []
        sel_hot_samples_w: List[float] = []
        ins_cold_samples_w: List[float] = []
        ins_hot_samples_w: List[float] = []
        explain_with_idx: str | None = None

        # Dynamic repeats: decide after the first iteration based on measured times
        threshold_ms = TIME_THRESHOLD_S * 1000.0
        target_repeats = max(1, REPEATS)  # will adjust after first run
        it = 0
        while it < target_repeats:
            # SELECT timings
            _setup_bench_with_idx(cur, rows, M_idx)
            sc, sh, plan_txt = measure_select_pk_range(cur, 'bench_data', with_index=True, return_explain=(it == 0))
            if it == 0:
                explain_with_idx = plan_txt
            sel_cold_samples_w.append(sc)
            sel_hot_samples_w.append(sh)
            exec_sql(cur, "DROP TABLE IF EXISTS bench_data;")

            # INSERT timings
            _setup_bench_with_idx(cur, rows, M_idx)
            ic, ih = measure_insert_cold_hot(cur, rows, 'bench_data')
            ins_cold_samples_w.append(ic)
            ins_hot_samples_w.append(ih)
            exec_sql(cur, "DROP TABLE IF EXISTS bench_data;")

            it += 1
            if it == 1:
                # If everything is fast, reduce to just a few repeats; otherwise keep as-is or at least 3 for stability
                if max(sc, sh, ic, ih) < threshold_ms:
                    target_repeats = max(1, FEW_REPEATS)
                else:
                    target_repeats = max(3, REPEATS)

        with_idx = {
            "storage_index_pct": (t_index_bytes / t_table_bytes * 100.0) if t_table_bytes else 0.0,
            "table_bytes": t_table_bytes,
            "index_bytes": t_index_bytes,
            "select_cold_ms": float(median(sel_cold_samples_w)) if sel_cold_samples_w else 0.0,
            "select_hot_ms": float(median(sel_hot_samples_w)) if sel_hot_samples_w else 0.0,
            "insert_cold_ms": float(median(ins_cold_samples_w)) if ins_cold_samples_w else 0.0,
            "insert_hot_ms": float(median(ins_hot_samples_w)) if ins_hot_samples_w else 0.0,
            "select_samples_cold_ms": sel_cold_samples_w,
            "select_samples_hot_ms": sel_hot_samples_w,
            "insert_samples_cold_ms": ins_cold_samples_w,
            "insert_samples_hot_ms": ins_hot_samples_w,
            "select_explain": explain_with_idx.splitlines(),
        }

        # No-index run: sizes once, then repeated timings with recreation each time
        _setup_bench_noidx(cur, rows)
        nt_table_bytes, nt_index_bytes = size_of_table(cur, 'bench_data_noidx')
        exec_sql(cur, "DROP TABLE IF EXISTS bench_data_noidx;")

        sel_cold_samples_n: List[float] = []
        sel_hot_samples_n: List[float] = []
        ins_cold_samples_n: List[float] = []
        ins_hot_samples_n: List[float] = []
        explain_no_idx: str | None = None

        threshold_ms = TIME_THRESHOLD_S * 1000.0
        target_repeats = max(1, REPEATS)
        it = 0
        while it < target_repeats:
            _setup_bench_noidx(cur, rows)
            sc, sh, plan_txt = measure_select_pk_range(cur, 'bench_data_noidx', with_index=False, return_explain=(it == 0))
            if it == 0:
                explain_no_idx = plan_txt
            sel_cold_samples_n.append(sc)
            sel_hot_samples_n.append(sh)
            exec_sql(cur, "DROP TABLE IF EXISTS bench_data_noidx;")

            _setup_bench_noidx(cur, rows)
            ic, ih = measure_insert_cold_hot(cur, rows, 'bench_data_noidx')
            ins_cold_samples_n.append(ic)
            ins_hot_samples_n.append(ih)
            exec_sql(cur, "DROP TABLE IF EXISTS bench_data_noidx;")

            it += 1
            if it == 1:
                if max(sc, sh, ic, ih) < threshold_ms:
                    target_repeats = max(1, FEW_REPEATS)
                else:
                    target_repeats = max(3, REPEATS)

        without_idx = {
            "table_bytes": nt_table_bytes,
            "index_bytes": nt_index_bytes,
            "select_cold_ms": float(median(sel_cold_samples_n)) if sel_cold_samples_n else 0.0,
            "select_hot_ms": float(median(sel_hot_samples_n)) if sel_hot_samples_n else 0.0,
            "storage_index_pct": (nt_index_bytes / nt_table_bytes * 100.0) if nt_table_bytes else 0.0,
            "insert_cold_ms": float(median(ins_cold_samples_n)) if ins_cold_samples_n else 0.0,
            "insert_hot_ms": float(median(ins_hot_samples_n)) if ins_hot_samples_n else 0.0,
            "select_samples_cold_ms": sel_cold_samples_n,
            "select_samples_hot_ms": sel_hot_samples_n,
            "insert_samples_cold_ms": ins_cold_samples_n,
            "insert_samples_hot_ms": ins_hot_samples_n,
            "select_explain": explain_no_idx.splitlines(),
        }

        return {
            "N": N,
            "rows": rows,
            "M": M_idx,
            "with_indexes": with_idx,
            "without_indexes": without_idx,
            "tuple": [
                N,
                rows,
                M_idx,
                with_idx["storage_index_pct"],
                with_idx["select_cold_ms"],
                with_idx["select_hot_ms"],
                with_idx["insert_cold_ms"],
                with_idx["insert_hot_ms"],
                without_idx["select_cold_ms"],
                without_idx["select_hot_ms"],
                without_idx["insert_cold_ms"],
                without_idx["insert_hot_ms"],
                with_idx["table_bytes"],
                with_idx["index_bytes"],
                without_idx["table_bytes"],
                without_idx["index_bytes"],
            ],
        }


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results: List[Dict] = []
    tuples: List[List] = []
    # Track progress for ETA
    with connect() as conn:
        ensure_baseline(conn)
        reset_bench_tables(conn)
        start_perf = time.perf_counter()
        start_wall = time.time()
        total_m = min(M_MAX, 30)
        total_combos = (MAX_N + 1) * total_m
        # Weighted ETA where time ~ C * (2^N) * M
        total_weight = 0.0
        for n_ in range(0, MAX_N + 1):
            pow_rows = 2 ** n_
            for m_ in range(1, total_m + 1):
                total_weight += pow_rows * m_
        completed = 0
        done_weight = 0.0
        for N in range(0, MAX_N + 1):
            for M_idx in range(1, total_m + 1):
                print(f"Running N={N} (rows={2**N}), M={M_idx}...")
                res = run_for_N(conn, N, M_idx)
                results.append(res)
                tuples.append(res["tuple"]) 
                # Stream partial results to avoid data loss on long runs
                with open(RESULTS_PATH, "w") as f:
                    json.dump(tuples, f, indent=4)
                with open("/bench/results_full.json", "w") as f2:
                    json.dump(results, f2, indent=4)
                # brief pause to reduce contention
                time.sleep(0.05)
                # Progress + ETA
                completed += 1
                elapsed = time.perf_counter() - start_perf
                # Update weighted progress
                weight = (2 ** N) * M_idx
                done_weight += weight
                remaining_combos = max(0, total_combos - completed)
                remaining_weight = max(0.0, total_weight - done_weight)
                # Weighted ETA based on elapsed per unit weight; fallback to simple avg early on
                if done_weight > 0:
                    time_per_weight = elapsed / done_weight
                    eta_seconds = remaining_weight * time_per_weight
                else:
                    avg_per = elapsed / completed if completed else 0.0
                    eta_seconds = remaining_combos * avg_per
                # format duration (HH:MM:SS, with days if needed)
                if eta_seconds >= 86400:
                    days = int(eta_seconds // 86400)
                    hhmmss = time.strftime("%H:%M:%S", time.gmtime(eta_seconds % 86400))
                    eta_str = f"{days}d {hhmmss}"
                else:
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                finish_ts = start_wall + (time.perf_counter() - start_perf) + eta_seconds
                finish_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_ts))
                pct = (completed / total_combos * 100.0) if total_combos else 100.0
                weighted_pct = (done_weight / total_weight * 100.0) if total_weight else 100.0
                print(f"Progress: {completed}/{total_combos} ({pct:.1f}%) | weighted {weighted_pct:.1f}% | ETA {eta_str} | finish ~{finish_str}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
