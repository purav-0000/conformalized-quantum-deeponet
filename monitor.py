#!/usr/bin/env python
"""Read-only live monitor for the August 2026 quantum DeepONet fleet runs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REMOTE_PYTHON = "/home/pmatlia/qiskit-fleet/micromamba/envs/qiskit/bin/python"
ADV = "/home/pmatlia/qiskit-fleet/runs/advection-train-bc86248"
ADV_HELIX = "/home/pmatlia/qiskit-fleet/runs/advection-fef5c1e-bc86248"
VOLT = "/home/pmatlia/qiskit-fleet/runs/voltage-00f5d82"


def _job(host, gpu, experiment, member, ensemble, log, completion, label="primary"):
    return {
        "host": host, "gpu": gpu, "experiment": experiment, "member": member,
        "ensemble": ensemble, "log": log, "completion": completion, "label": label,
    }


JOBS = []
for member, host, gpu, root in (
    (0, "apex", 1, ADV), (1, "chroma", 0, ADV), (2, "chroma", 1, ADV),
    (3, "helix", 0, ADV_HELIX), (4, "helix", 1, ADV_HELIX),
    (5, "vector", 1, ADV), (6, "vector", 2, ADV), (7, "chroma", 0, ADV),
):
    JOBS.append(_job(host, gpu, "advection", member, "advection_trajectory_v2",
        f"{root}/member_{member}.log",
        f"{root}/worktree/models/ensembles/advection_trajectory_v2/model_{member}/timing.json"))
JOBS += [
    _job("chroma", 1, "advection", 5, "advection_trajectory_v2",
         f"{ADV}/member_5_duplicate.log", f"{ADV}/worktree/models/ensembles/advection_trajectory_v2/model_5/timing.json", "duplicate"),
    _job("chroma", 0, "advection", 6, "advection_trajectory_v2",
         f"{ADV}/member_6_duplicate.log", f"{ADV}/worktree/models/ensembles/advection_trajectory_v2/model_6/timing.json", "duplicate"),
]
for member, host, gpu in ((0,"chroma",1),(1,"helix",0),(2,"helix",1),(3,"vector",1),
                          (4,"vector",2),(5,"apex",1),(6,"chroma",0),(7,"chroma",1)):
    JOBS.append(_job(host, gpu, "offline_voltage", member, "offline_voltage_trajectory_v2",
        f"{VOLT}/offline_member_{member}.log",
        f"{VOLT}/source/models/ensembles/offline_voltage_trajectory_v2/model_{member}/timing.json"))
JOBS += [
    _job("helix", 0, "offline_voltage", 3, "offline_voltage_trajectory_v2",
         f"{VOLT}/offline_member_3_duplicate.log", f"{VOLT}/source/models/ensembles/offline_voltage_trajectory_v2/model_3/timing.json", "duplicate"),
    _job("helix", 1, "offline_voltage", 4, "offline_voltage_trajectory_v2",
         f"{VOLT}/offline_member_4_duplicate.log", f"{VOLT}/source/models/ensembles/offline_voltage_trajectory_v2/model_4/timing.json", "duplicate"),
]
for member, host, gpu in ((0,"helix",0),(1,"helix",1),(2,"vector",1),(3,"vector",2)):
    JOBS.append(_job(host, gpu, "online_voltage", member, "online_voltage_trajectory_v2",
        f"{VOLT}/online_member_{member}.log",
        f"{VOLT}/source/models/ensembles/online_voltage_trajectory_v2/model_{member}/timing.json"))

REMOTE_CODE = r'''
import json, os, re, subprocess, sys
jobs=json.loads(sys.argv[2])
processes=subprocess.run(["pgrep","-af","src.runners.training|launch_training_after|chain_training_member"],text=True,capture_output=True).stdout.splitlines()
rows=[]
for job in jobs:
    matching=[line for line in processes if job["ensemble"] in line and (f"ensemble_member={job['member']}" in line or re.search(rf"\s{job['member']}\s+\d+\s+[^ ]+", line))]
    running=any("src.runners.training" in line for line in matching)
    queued=any("launch_training_after" in line or "chain_training_member" in line for line in matching)
    text=""
    if os.path.isfile(job["log"]):
        with open(job["log"], "r", encoding="utf-8", errors="replace") as handle: text=handle.read()
    steps=re.findall(r"^\s*(\d+)\s+\[", text, re.M)
    if os.path.isfile(job["completion"]): state="completed"
    elif running: state="running"
    elif "Traceback (most recent call last)" in text or "did not complete successfully" in text: state="failed"
    elif queued: state="queued"
    elif text: state="stopped"
    else: state="queued"
    job.update(state=state, step=int(steps[-1]) if steps else None)
    rows.append(job)
print(json.dumps(rows))
'''


def _probe(host: str, jobs: list[dict]) -> list[dict]:
    payload = json.dumps(jobs, separators=(",", ":"))
    encoded = base64.b64encode(REMOTE_CODE.encode()).decode()
    bootstrap = "import base64,sys;exec(base64.b64decode(sys.argv[1]))"
    remote = " ".join(shlex.quote(value) for value in (
        REMOTE_PYTHON, "-c", bootstrap, encoded, payload
    ))
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ssh exited {result.returncode}")
    return json.loads(result.stdout)


def query() -> tuple[list[dict], list[str]]:
    grouped = {host: [j for j in JOBS if j["host"] == host] for host in {j["host"] for j in JOBS}}
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_probe, host, jobs): host for host, jobs in grouped.items()}
        for future in as_completed(futures):
            try: rows.extend(future.result())
            except Exception as exc: errors.append(f"{futures[future]}: {exc}")
    order = {"running":0,"queued":1,"stopped":2,"failed":3,"completed":4}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r["experiment"], r["member"], r["label"]))
    return rows, errors


def render(rows: list[dict], errors: list[str]) -> None:
    headers = ["STATE","SLOT","EXPERIMENT","MEMBER","STEP","COPY"]
    table = [[r["state"], f'{r["host"]}:{r["gpu"]}', r["experiment"], str(r["member"]),
              f'{r["step"] or 0}/40000' if r["experiment"] != "online_voltage" else f'{r["step"] or 0}/15000', r["label"]] for r in rows]
    widths=[max([len(headers[i])] + [len(row[i]) for row in table]) for i in range(len(headers))]
    print(f"Fleet status at {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-"*w for w in widths))
    for row in table: print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    states=sorted({r["state"] for r in rows})
    print("Summary: "+", ".join(f'{s}={sum(r["state"]==s for r in rows)}' for s in states))
    for error in errors: print(f"Host error: {error}", file=sys.stderr)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", nargs="?", const=30.0, type=float, metavar="SECONDS")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args=parser.parse_args()
    while True:
        rows, errors=query()
        if args.as_json: print(json.dumps({"jobs":rows,"host_errors":errors}, indent=2))
        else: render(rows, errors)
        if not args.watch or (rows and all(r["state"] in {"completed","failed"} for r in rows)): return 0
        time.sleep(max(5.0,args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
