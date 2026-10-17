#!/usr/bin/env python3
"""
GPTuner experiment driver for adco-experiments harness.
Replaces BenchbaseRunner with HarnessWorkloadRunner (harness Python drivers).
GPTuner source is imported verbatim and adapted without modifying source files.
Supports parameterized hardware inputs (CPU cores, RAM GB, disk size/type)
via CLI arguments and configs/postgres.ini.
"""
import sys
import os
import time
import json
import re
import argparse
import shutil
import subprocess
from configparser import ConfigParser

# Ensure GPTuner src is on sys.path
gptuner_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(gptuner_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import psycopg2
from dbms.postgres import PgDBMS
from config_recommender.coarse_stage import CoarseStage
from config_recommender.fine_stage import FineStage
from knowledge_handler.knowledge_preparation import KGPre
from knowledge_handler.knowledge_transformation import KGTrans
from knowledge_handler.knowledge_update import KGUpdate
from knowledge_handler.utils import set_hardware_override
import space_optimizer.default_space as _ds


def sanitize_knob_value(knob, value, memory_gb=2.0, cpu_cores=2):
    """Ensure knob values dynamically scale with parameterized hardware memory limits and PostgreSQL 17 bounds."""
    if value is None:
        return None
    val_str = str(value).strip()

    # huge_pages: PostgreSQL in Docker crashes if set to "on" without host hugepage support
    if knob == "huge_pages":
        return "off"

    # Avoid loading non-existent libraries or failing on archive commands
    if knob in ("shared_preload_libraries", "archive_command", "cluster_name"):
        return ""

    if knob == "archive_mode":
        return "off"

    # Worker and connection limits
    worker_limits = {
        "autovacuum_max_workers": max(2, int(cpu_cores * 4)),
        "max_worker_processes": max(4, int(cpu_cores * 8)),
        "max_parallel_workers": max(2, int(cpu_cores * 4)),
        "max_parallel_workers_per_gather": max(1, int(cpu_cores * 2)),
        "max_parallel_maintenance_workers": max(1, int(cpu_cores * 2)),
        "max_connections": 200,
        "max_wal_senders": 10,
        "max_replication_slots": 10,
        "commit_delay": 1000,
        "deadlock_timeout": 5000,
        "wal_writer_delay": 200,
        "bgwriter_delay": 1000,
        "from_collapse_limit": 32,
        "join_collapse_limit": 32,
        "geqo_threshold": 32,
    }

    if knob in worker_limits:
        try:
            val_int = int(float(val_str))
            if val_int > worker_limits[knob]:
                return str(worker_limits[knob])
        except ValueError:
            pass

    float_limits = {
        "autovacuum_vacuum_scale_factor": 0.4,
        "autovacuum_analyze_scale_factor": 0.4,
        "vacuum_cost_delay": 20.0,
    }

    if knob in float_limits:
        try:
            val_float = float(val_str)
            if val_float > float_limits[knob]:
                return str(float_limits[knob])
        except ValueError:
            pass

    # Dynamic memory limits proportional to the input memory_gb parameter
    mem_gb = float(memory_gb)
    mem_limits_mb = {
        "shared_buffers": int(mem_gb * 0.50 * 1024),          # e.g., 1024MB for 2GB RAM
        "effective_cache_size": int(mem_gb * 0.75 * 1024),    # e.g., 1536MB for 2GB RAM
        "maintenance_work_mem": int(mem_gb * 0.25 * 1024),    # e.g., 512MB for 2GB RAM
        "work_mem": max(4, int((mem_gb * 1024) / 32)),       # e.g., 64MB for 2GB RAM
        "wal_buffers": min(64, int(mem_gb * 32)),            # e.g., 64MB for 2GB RAM
        "autovacuum_work_mem": int(mem_gb * 0.15 * 1024),     # e.g., 300MB for 2GB RAM
        "logical_decoding_work_mem": int(mem_gb * 0.10 * 1024),# e.g., 200MB for 2GB RAM
        "temp_buffers": int(mem_gb * 0.10 * 1024),            # e.g., 200MB for 2GB RAM
    }

    if knob in mem_limits_mb:
        max_mb = mem_limits_mb[knob]
        min_mb = int(mem_gb * 0.25 * 1024) if knob == "effective_cache_size" else None
        try:
            if val_str.isdigit():
                num = int(val_str)
                max_blocks = max_mb * 128  # 8kB blocks
                if num > max_blocks:
                    return str(max_blocks)
                if min_mb is not None and num < min_mb * 128:
                    return str(min_mb * 128)
            else:
                m = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$', val_str)
                if m:
                    num = float(m.group(1))
                    unit = (m.group(2) or 'MB').upper()
                    if 'GB' in unit or 'G' in unit:
                        mb = num * 1024
                    elif 'KB' in unit or 'K' in unit:
                        mb = num / 1024
                    else:
                        mb = num
                    if mb > max_mb:
                        return f"{max_mb}MB"
                    if min_mb is not None and mb < min_mb:
                        return f"{min_mb}MB"
        except Exception:
            pass

    return val_str


class HarnessWorkloadRunner:
    """Drives harness Python workload clients (Smallbank / TPC-C) instead of BenchBase."""
    def __init__(self, dbms, test, exp_path, benchmark, dbms_name, target_path=None):
        self.dbms = dbms
        self.test = test
        self.exp_path = exp_path
        self.benchmark = benchmark  # 'smallbank' or 'tpcc'
        self.dbms_name = dbms_name
        self.target_path = target_path or os.path.join(gptuner_dir, "optimization_results/temp_results")
        self._stdout = ""
        self._stderr = ""
        self.process = None

    def clear_summary_dir(self):
        os.makedirs(self.target_path, exist_ok=True)

    def check_sequence_in_file(self):
        return False

    def run_benchmark(self):
        host = getattr(self.dbms, "host", "127.0.0.1")
        port = str(getattr(self.dbms, "port", 5432))
        if self.benchmark == "smallbank":
            cmd = [
                sys.executable, "main.py", "run",
                "--driver", self.dbms_name,
                "--host", host,
                "--port", port,
                "--accounts", "100000",
                "--transactions", "2000"
            ]
            cwd = os.path.join(self.exp_path, "workload/apps/smallbank")
        elif self.benchmark == "tpcc":
            cfg_path = os.path.join(self.exp_path, "workload/apps/tpcc/db.config")
            cmd = [
                sys.executable, "tpcc.py", self.dbms_name,
                f"--config={cfg_path}",
                "--warehouses", os.environ.get("WAREHOUSES", "32"),
                "--clients", os.environ.get("CLIENTS", "8"),
                "--duration", "20",
                "--no-load"
            ]
            cwd = os.path.join(self.exp_path, "workload/apps/tpcc")
        elif self.benchmark == "tpch":
            self._run_tpch_queries(host, port)
            return
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")

        try:
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
            self._stdout = res.stdout
            self._stderr = res.stderr
        except subprocess.TimeoutExpired as e:
            print(f"Workload timed out: {e}")
            self._stdout = e.stdout if e.stdout else ""
            self._stderr = e.stderr if e.stderr else ""
        except Exception as e:
            print(f"Workload execution error: {e}")
            self._stdout = ""
            self._stderr = str(e)

    def _run_tpch_queries(self, host, port):
        """Run the 22 TPCH queries sequentially and emit a TOTAL summary line.

        TOTAL,<total_s>,<avg_ms_per_query>,<queries_per_s> matches the
        _parse_total_row format (latency at index 2, throughput at index 3).
        """
        qdir = os.path.join(self.exp_path, "workload/databases/PostgreSQL/tpch")
        files = [os.path.join(qdir, f"query{i}.sql") for i in range(1, 23)]
        files = [f for f in files if os.path.exists(f)]
        user = getattr(self.dbms, "user", "postgres")
        password = getattr(self.dbms, "password", "postgres")
        dbname = getattr(self.dbms, "db", self.benchmark)
        start = time.time()
        try:
            conn = psycopg2.connect(host=host, port=int(port), user=user,
                                    password=password, dbname=dbname, connect_timeout=5)
            try:
                conn.autocommit = True
                cur = conn.cursor()
                for f in files:
                    with open(f) as fh:
                        sql = fh.read()
                    try:
                        cur.execute(sql)
                        try:
                            cur.fetchall()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"TPCH query {os.path.basename(f)} failed: {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass
            finally:
                conn.close()
        except Exception as e:
            print(f"TPCH workload execution error: {e}")
        total = max(time.time() - start, 1e-3)
        n = max(len(files), 1)
        self._stdout = f"TOTAL,{total:.3f},{total / n * 1000.0:.2f},{n / total:.4f}\n"
        self._stderr = ""

    def _parse_total_row(self):
        for line in self._stdout.splitlines():
            line_clean = line.strip()
            if line_clean.upper().startswith("TOTAL,") or line_clean.lower().startswith("total,"):
                parts = line_clean.split(",")
                if len(parts) >= 4:
                    return parts
            elif line_clean.upper().startswith("TOTAL") and ("TXN/S" in line_clean.upper() or len(line_clean.split()) >= 4):
                tokens = line_clean.replace("txn/s", "").replace("TXN/S", "").split()
                if len(tokens) >= 4:
                    return tokens
        return None

    def get_throughput(self):
        parts = self._parse_total_row()
        if parts is not None:
            try:
                val = float(parts[3])
                if val > 0:
                    return val
            except (ValueError, IndexError):
                pass
        print(f"Warning: Could not parse throughput from stdout:\n{self._stdout[:500]}")
        return 1.0

    def get_latency(self):
        parts = self._parse_total_row()
        if parts is not None:
            try:
                val = float(parts[2])
                if val > 0:
                    return val
            except (ValueError, IndexError):
                pass
        return 1000.0


def _table_row_count(host, port, user, password, dbname, table):
    """Best-effort row count; returns -1 on any failure (caller loads)."""
    try:
        conn = psycopg2.connect(host=host, port=int(port), user=user,
                                password=password, dbname=dbname, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute(f'SELECT count(*) FROM "{table}"')
            row = cur.fetchone()
            return int(row[0]) if row else -1
        finally:
            conn.close()
    except Exception:
        return -1


def prepare_workload_database(exp_path, benchmark, dbms_name, host="127.0.0.1", port=5432,
                              user="postgres", password="postgres", dbname=None):
    """Ensure benchmark schema and initial data are populated before tuning iterations begin.

    Skips the load when representative data is already present (e.g. the
    harness restored its snapshot): tpcc needs >= 2 warehouses, smallbank
    needs >= 100000 accounts (this function's own load sizes).
    """
    print(f"Preparing and checking initial dataset for {benchmark}...")
    if benchmark == "smallbank":
        if _table_row_count(host, port, user, password, dbname or "smallbank", "accounts") >= 100000:
            print(f"Dataset already present for {benchmark}; skipping load.")
            return
        cwd = os.path.join(exp_path, "workload/apps/smallbank")
        cmd = [
            sys.executable, "main.py", "load",
            "--driver", dbms_name,
            "--host", host,
            "--port", str(port),
            "--accounts", "100000",
            "--threads", "8",
            "--reset"
        ]
        subprocess.run(cmd, cwd=cwd, check=False)
    elif benchmark == "tpcc":
        if _table_row_count(host, port, user, password, dbname or "tpcc", "warehouse") >= 2:
            print(f"Dataset already present for {benchmark}; skipping load.")
            return
        cwd = os.path.join(exp_path, "workload/apps/tpcc")
        cfg_path = os.path.join(exp_path, "workload/apps/tpcc/db.config")
        cmd = [
            sys.executable, "tpcc.py", dbms_name,
            f"--config={cfg_path}",
            "--warehouses", "2",
            "--clients", "4",
            "--duration", "5",
            "--reset"
        ]
        subprocess.run(cmd, cwd=cwd, check=False)
    elif benchmark == "tpch":
        # No Python loader for tpch; data comes from the harness snapshot
        # (bench_fresh_db). Skip when representative data is already present.
        if _table_row_count(host, port, user, password, dbname or "tpch", "lineitem") >= 100000:
            print(f"Dataset already present for {benchmark}; skipping load.")
            return
        print(f"TPCH dataset not detected; expecting the harness snapshot to provide it.")
    print(f"Dataset preparation for {benchmark} complete.")


class HarnessPgDBMS(PgDBMS):
    """PgDBMS subclass supporting custom host, port, parameterized memory safety, and Docker recovery."""
    def __init__(self, db, user, password, host="127.0.0.1", port=5432,
                 memory_gb=2.0, cpu_cores=2, restart_cmd="docker restart adcoexp-db",
                 recover_script="./scripts/recover_postgres.sh",
                 knob_info_path="./knowledge_collection/postgres/knob_info/system_view.json"):
        super().__init__(db, user, password, restart_cmd, recover_script, knob_info_path)
        self.host = host
        self.port = int(port)
        self.memory_gb = float(memory_gb)
        self.cpu_cores = int(cpu_cores)

    def _connect(self, db=None):
        self.failed_times = 0
        target_db = db if db is not None else self.db
        for _ in range(15):
            try:
                self.connection = psycopg2.connect(
                    database=target_db,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port,
                    connect_timeout=5
                )
                self.connection.autocommit = True
                return True
            except Exception as e:
                self.failed_times += 1
                time.sleep(2)
        print(f"Failed to connect to {target_db} after retries")
        return False

    def reset_config(self):
        try:
            if not self.connection:
                self._connect()
            super().reset_config()
        except Exception as e:
            print(f"reset_config error: {e}")
            self.recover_dbms()

    def set_knob(self, knob, knob_value):
        clean_val = sanitize_knob_value(knob, knob_value, self.memory_gb, self.cpu_cores)
        if clean_val is None or clean_val == "":
            return True
        return super().set_knob(knob, clean_val)

    def recover_dbms(self):
        print("Recovering PostgreSQL: clearing postgresql.auto.conf and restarting container...")
        self._disconnect()
        container = os.environ.get("PG_CONTAINER_NAME", "adcoexp-db")
        subprocess.run(["docker", "exec", container, "bash", "-c", "rm -f /var/lib/postgresql/data/postgresql.auto.conf"], check=False)
        subprocess.run(["docker", "restart", container], check=False)
        time.sleep(5)
        return self._connect()

    def reconfigure(self):
        self._disconnect()
        if self.restart_cmd:
            os.system(self.restart_cmd)
        time.sleep(3)
        if self._connect():
            return True
        print("Reconfiguration failed to start PostgreSQL. Attempting container recovery...")
        return self.recover_dbms()


def patch_default_space(exp_path, benchmark, dbms_name):
    """Monkey-patch DefaultSpace to use HarnessWorkloadRunner and skip Benchbase table resetting."""
    def custom_get_default_result(self):
        print(f"Testing default performance for {self.test}...")
        self.dbms.reset_config()
        self.dbms.reconfigure()
        runner = HarnessWorkloadRunner(self.dbms, self.test, exp_path, benchmark, dbms_name, self.summary_path)
        runner.clear_summary_dir()
        runner.run_benchmark()
        throughput = runner.get_throughput()
        print(f"Default throughput: {throughput:.2f} txn/s")
        return throughput

    def custom_set_and_replay_ori(self, config, seed=0):
        self.round += 1
        print(f"Tuning round {self.round} ...")
        self.dbms.reset_config()
        self.dbms.reconfigure()

        for knob in self.target_knobs:
            try:
                control_para = config.get(f"control_{knob}")
                if control_para == "0":
                    value = config[knob]
                elif control_para == "1":
                    value = config[f"special_{knob}"]
                else:
                    value = config.get(knob, None)
            except Exception:
                value = config.get(knob, None)
            if value is not None:
                self.dbms.set_knob(knob, value)

        success = self.dbms.reconfigure()
        if not success or self.dbms.failed_times >= 4:
            print(f"Round {self.round} failed to start DB with configuration. Applying penalty.")
            self.dbms.recover_dbms()
            return -int(self.penalty) / 2

        runner = HarnessWorkloadRunner(self.dbms, self.test, exp_path, benchmark, dbms_name, self.summary_path)
        runner.clear_summary_dir()
        runner.run_benchmark()
        throughput = runner.get_throughput()
        print(f"Round {self.round} Throughput: {throughput:.2f} txn/s")

        if throughput > self.penalty:
            self.penalty = throughput

        return -float(throughput)

    _ds.DefaultSpace.get_default_result = custom_get_default_result
    _ds.DefaultSpace.set_and_replay_ori = custom_set_and_replay_ori


def main():
    parser = argparse.ArgumentParser(description="GPTuner harness runner with parameterized hardware input")
    parser.add_argument("--db", default="postgres", help="DBMS type (postgres/mysql)")
    parser.add_argument("--benchmark", required=True, choices=["smallbank", "tpcc", "tpch"], help="Target benchmark")
    parser.add_argument("--exp-path", required=True, help="Path to adco-experiments root")
    parser.add_argument("--memory-gb", type=float, default=None, help="Target DBMS memory limit in GB")
    parser.add_argument("--cpu-cores", type=int, default=None, help="Target DBMS CPU cores")
    parser.add_argument("--storage-gb", type=float, default=None, help="Target DBMS storage size in GB")
    parser.add_argument("--disk-type", type=str, default=None, choices=["SSD", "HDD"], help="Target DBMS disk type")
    parser.add_argument("--coarse-trials", type=int, default=30, help="Number of coarse stage trials")
    parser.add_argument("--fine-trials", type=int, default=110, help="Total trials (coarse + fine)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--timeout", type=int, default=120, help="Workload timeout per trial")
    parser.add_argument("--skip-kg", action="store_true", help="Skip LLM knowledge preparation if already cached")
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg_path = os.path.join(gptuner_dir, "configs", f"{args.db}.ini")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg.read(cfg_path)

    db_sec = cfg["DATABASE"]
    llm_sec = cfg["LLM"] if "LLM" in cfg else {}
    hw_sec = cfg["HARDWARE"] if "HARDWARE" in cfg else {}

    # Resolve hardware parameters from CLI flags -> config file -> defaults
    memory_gb = args.memory_gb if args.memory_gb is not None else float(hw_sec.get("memory_gb", 2.0))
    cpu_cores = args.cpu_cores if args.cpu_cores is not None else int(hw_sec.get("cpu_cores", 2))
    storage_gb = args.storage_gb if args.storage_gb is not None else float(hw_sec.get("storage_gb", 10.0))
    disk_type = args.disk_type if args.disk_type is not None else hw_sec.get("disk_type", "SSD")

    print(f"Target Hardware Configuration: {cpu_cores} CPU cores, {memory_gb:.1f} GB RAM, {storage_gb:.1f} GB {disk_type}")
    set_hardware_override(cpu_cores=cpu_cores, memory_gb=memory_gb, disk_gb=storage_gb, disk_type=disk_type)

    db_host = db_sec.get("host", "127.0.0.1")
    # Resolve host: if 'pgdb' doesn't resolve locally on host, fallback to 127.0.0.1
    try:
        import socket
        socket.gethostbyname(db_host)
    except socket.gaierror:
        db_host = "127.0.0.1"

    db_port = int(db_sec.get("port", 5432))
    db_user = db_sec.get("user", "postgres")
    db_password = db_sec.get("password", "postgres")
    db_name = args.benchmark
    restart_cmd = db_sec.get("restart_cmd", "docker restart adcoexp-db")
    recover_script = os.path.join(gptuner_dir, db_sec.get("recover_script", "./scripts/recover_postgres.sh"))
    knob_info_path = os.path.join(gptuner_dir, db_sec.get("knob_info_path", "./knowledge_collection/postgres/knob_info/system_view.json"))

    os.chdir(gptuner_dir)
    for stale_dir in [
        os.path.join(gptuner_dir, "smac3_output", "optimization_results", args.db, "coarse"),
        os.path.join(gptuner_dir, "smac3_output", "optimization_results", args.db, "fine"),
        os.path.join(gptuner_dir, "optimization_results", args.db, "coarse"),
        os.path.join(gptuner_dir, "optimization_results", args.db, "fine"),
        os.path.join(gptuner_dir, "optimization_results", args.db, "log"),
    ]:
        shutil.rmtree(stale_dir, ignore_errors=True)
    os.makedirs(os.path.join(gptuner_dir, "optimization_results", args.db, "log"), exist_ok=True)
    os.makedirs(os.path.join(gptuner_dir, "optimization_results", args.db, "coarse"), exist_ok=True)
    os.makedirs(os.path.join(gptuner_dir, "optimization_results", args.db, "fine"), exist_ok=True)
    os.makedirs(os.path.join(gptuner_dir, "optimization_results", "temp_results"), exist_ok=True)

    # Pre-populate benchmark data if not yet loaded
    prepare_workload_database(args.exp_path, args.benchmark, args.db, host=db_host, port=db_port,
                              user=db_user, password=db_password, dbname=db_name)

    dbms = HarnessPgDBMS(
        db=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        memory_gb=memory_gb,
        cpu_cores=cpu_cores,
        restart_cmd=restart_cmd,
        recover_script=recover_script,
        knob_info_path=knob_info_path
    )

    patch_default_space(args.exp_path, args.benchmark, args.db)

    target_knobs_path = os.path.join(gptuner_dir, f"knowledge_collection/{args.db}/target_knobs.txt")
    if not os.path.exists(target_knobs_path):
        candidate_path = os.path.join(gptuner_dir, f"knowledge_collection/{args.db}/candidate_knobs.txt")
        if os.path.exists(candidate_path):
            with open(candidate_path, "r") as f:
                cdata = json.load(f)
            with open(target_knobs_path, "w") as f:
                for k in list(cdata.keys())[:30]:
                    f.write(f"{k}\n")

    # Optional LLM knowledge collection phase
    if not args.skip_kg and llm_sec.get("api_key"):
        api_base = llm_sec.get("api_base", "https://generativelanguage.googleapis.com/v1beta/openai/")
        api_key = llm_sec.get("api_key")
        model = llm_sec.get("model", "gemini-3.5-flash-lite")
        print(f"Running Knowledge Handler with LLM ({model}) for {cpu_cores} CPUs / {memory_gb}GB RAM...")
        try:
            with open(target_knobs_path, "r") as f:
                target_knobs = [line.strip() for line in f if line.strip()]
            kg_pre = KGPre(db=args.db, api_base=api_base, api_key=api_key, model=model)
            kg_trans = KGTrans(db=args.db, api_base=api_base, api_key=api_key, model=model)
            kg_update = KGUpdate(db=args.db, api_base=api_base, api_key=api_key, model=model)
            for knob in target_knobs[:10]:
                try:
                    kg_pre.pipeline(knob)
                    kg_trans.pipeline(knob)
                    kg_update.pipeline(knob)
                except Exception as e:
                    print(f"Warning during KG processing for {knob}: {e}")
        except Exception as e:
            print(f"KG pipeline skipped due to: {e}")

    print("==================================================")
    print(f" Starting GPTuner Coarse Stage for {args.benchmark}")
    print("==================================================")
    coarse = CoarseStage(
        dbms=dbms,
        test=args.benchmark,
        timeout=args.timeout,
        target_knobs_path=target_knobs_path,
        seed=args.seed
    )
    coarse.optimize(
        name=f"./optimization_results/{args.db}/coarse/",
        trials_number=args.coarse_trials,
        initial_config_number=min(10, args.coarse_trials)
    )

    # Ensure coarse runhistory.json is available in optimization_results directory
    coarse_src = os.path.join(gptuner_dir, f"smac3_output/optimization_results/{args.db}/coarse/{args.seed}/runhistory.json")
    coarse_dst = os.path.join(gptuner_dir, f"optimization_results/{args.db}/coarse/{args.seed}/runhistory.json")
    if os.path.exists(coarse_src):
        os.makedirs(os.path.dirname(coarse_dst), exist_ok=True)
        shutil.copyfile(coarse_src, coarse_dst)

    print("==================================================")
    print(f" Starting GPTuner Fine Stage for {args.benchmark}")
    print("==================================================")
    fine = FineStage(
        dbms=dbms,
        test=args.benchmark,
        timeout=args.timeout,
        target_knobs_path=target_knobs_path,
        seed=args.seed
    )
    fine.optimize(
        name=f"./optimization_results/{args.db}/fine/",
        trials_number=args.fine_trials
    )

    # Read the best configuration from Fine Stage runhistory
    fine_history_candidates = [
        os.path.join(gptuner_dir, f"smac3_output/optimization_results/{args.db}/fine/{args.seed}/runhistory.json"),
        os.path.join(gptuner_dir, f"optimization_results/{args.db}/fine/{args.seed}/runhistory.json"),
        os.path.join(gptuner_dir, f"smac3_output/fine/{args.seed}/runhistory.json"),
    ]
    best_config = {}
    for fine_history_path in fine_history_candidates:
        if os.path.exists(fine_history_path):
            with open(fine_history_path, "r") as f:
                hdata = json.load(f)
            data_entries = hdata.get("data", [])
            configs = hdata.get("configs", {})
            if data_entries:
                best_entry = min(data_entries, key=lambda x: x[4])
                best_config_id = str(best_entry[0])
                best_config = configs.get(best_config_id, {})
                print(f"Best configuration (cost={best_entry[4]}): {best_config}")
                break

    out_res_dir = os.path.join(args.exp_path, "results/db_layer/gptuner")
    os.makedirs(out_res_dir, exist_ok=True)
    best_conf_file = os.path.join(out_res_dir, f"best_configuration_{args.benchmark}.json")
    with open(best_conf_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best configuration to {best_conf_file}")

    # Apply best configuration to PostgreSQL
    if best_config:
        print("Applying best configuration to database...")
        dbms.reset_config()
        for knob in fine.target_knobs:
            try:
                control_para = best_config.get(f"control_{knob}")
                if control_para == "0":
                    value = best_config[knob]
                elif control_para == "1":
                    value = best_config[f"special_{knob}"]
                else:
                    value = best_config.get(knob, None)
            except Exception:
                value = best_config.get(knob, None)
            if value is not None:
                dbms.set_knob(knob, value)
        dbms.reconfigure()
        print("Best configuration applied.")


if __name__ == "__main__":
    main()
