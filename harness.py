#!/usr/bin/env python3
"""
gpuqual supervisor -- owns all persistent state, restarts the CUDA worker,
monitors Xid / PCIe AER / telemetry, and enforces the total test budget.

Multi-GPU: every card nvidia-smi reports is monitored. Temperature, HBM
temperature and power-limit ceilings are evaluated per card, and any single card
breaching a halt threshold stops the whole run -- the cards share a chassis and
an airflow path, so one overheating is never a per-card problem.

Restart failsafe: the budget is consumed in accumulated TEST seconds, not wall
clock, and is checkpointed atomically. Killing this process (or the machine)
and starting it again resumes toward the same 48h target instead of restarting
the clock. Coverage is credited only for phases that completed, so an
interrupted phase under-counts rather than inflating the qualification math.

NEVER writes any power management setting.
"""

import argparse, json, os, re, signal, subprocess, sys, threading, time, uuid
from pathlib import Path

DEF_DIR = Path("/home/smith/gpuqual")
SESSION_MAX = 6 * 3600          # cap one worker session; forces periodic re-alloc
HANG_TIMEOUT = 420              # no worker event for this long => assume hung


def gpu_pci_addrs():
    """Map nvidia-smi GPU index -> sysfs PCI address, for every card present.

    This used to be a single hardcoded address, which silently left every other
    card unmonitored for PCIe AER -- exactly the class of failure this harness
    exists to catch. Enumerating instead means adding or moving a card needs no
    edit here.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    addrs = {}
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # nvidia-smi prints an 8-digit domain (00000000:01:00.0); sysfs uses 4
        # (0000:01:00.0). Normalise or every path lookup silently misses.
        bus = parts[1].lower()
        if bus.count(":") == 2:
            dom, rest = bus.split(":", 1)
            try:
                bus = f"{int(dom, 16):04x}:{rest}"
            except ValueError:
                pass
        addrs[idx] = bus
    return addrs

XID_MEANING = {
    13: "Graphics engine exception / illegal address",
    31: "GPU memory page fault (MMU)",
    43: "GPU stopped processing (channel error)",
    45: "Preemptive cleanup / robust channel",
    48: "Double-bit ECC error",
    62: "Internal micro-controller halt",
    63: "ECC row remapping event",
    64: "ECC row remapping failure",
    74: "NVLink error",
    79: "GPU has fallen off the bus",
    92: "High single-bit ECC error rate",
    94: "Contained ECC error",
    95: "Uncontained ECC error",
    119: "GSP RPC timeout",
    120: "GSP error",
}
# Any of these is an automatic NO-GO.
XID_FATAL = {48, 63, 64, 79, 92, 94, 95}


def now():
    return time.time()


class State:
    def __init__(self, path):
        self.path = Path(path)
        self.d = {
            "run_id": str(uuid.uuid4())[:8],
            "created": now(),
            "elapsed_test_s": 0.0,
            "cycle": 0,
            "seed": 0x1234ABCD,
            "coverage": {"bitreads": 0, "alu_ops": 0, "retention_bit_hours": 0.0,
                         "retention_max_hold_s": 0.0, "cycles_completed": 0},
            "tally": {"mismatch": 0, "hard": 0, "soft": 0, "transient": 0, "sdc": 0,
                      "xid": 0, "xid_fatal": 0, "context_loss": 0, "worker_hang": 0,
                      "thermal_drift": 0, "critical": 0, "pcie_correctable": 0,
                      "pcie_fatal": 0, "backoff_s": 0.0},
            "known_bad": [],
            "restarts": 0,
            "reboots": 0,
            "boot_id": boot_id(),
            "gib": 0,
            "power_limit_w": 0.0,
            "power_limit_history": [],
            "target_s": 0.0,
            "deadline": 0.0,
            # Worst case across every card, kept flat for report.py. Authoritative
            # per-card values live in "per_gpu".
            "peak_gpu_c": 0,
            "peak_mem_c": 0,
            "per_gpu": {},
            "halted": False,
            "halt_reason": "",
            "finished": False,
        }
        if self.path.exists():
            try:
                self.d.update(json.loads(self.path.read_text()))
            except Exception as e:
                print(f"[warn] state unreadable ({e}); starting fresh", file=sys.stderr)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        os.replace(tmp, self.path)   # atomic


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        return "unknown"


class Harness:
    def __init__(self, a):
        self.dir = Path(a.dir)
        self.logs = self.dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.a = a
        self.state = State(self.logs / "state.json")
        self.events = open(self.logs / "events.jsonl", "a", buffering=1)
        self.tele_f = open(self.logs / "telemetry.csv", "a", buffering=1)
        self.control = self.logs / "control"
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.abort = False
        self.graceful = False
        self.proc = None
        self.last_event = now()
        self.lock = threading.Lock()
        # credit_mark is the instant up to which test time has already been
        # credited. Time is credited CONTINUOUSLY, not at session end, so an
        # abrupt kill loses at most one credit interval instead of the whole
        # session.
        self.credit_mark = None
        self.credit_lock = threading.Lock()
        self.halted = bool(self.state.d.get("halted"))
        self.last_tele = now()
        # Resolved once: telemetry and AER both need it, and hotplugging a GPU
        # mid-run is not a supported scenario.
        self.pci = gpu_pci_addrs()
        # With --device this supervisor owns exactly one card. Narrowing self.pci
        # scopes AER, telemetry and halting to that card, so two supervisors can
        # qualify two cards side by side without halting each other's run.
        if a.device is not None:
            self.pci = {a.device: self.pci[a.device]} if a.device in self.pci else {}

    def credit(self):
        """Fold elapsed wall time since the last credit into the test budget."""
        with self.credit_lock:
            if self.credit_mark is None or self.paused.is_set():
                return
            t = now()
            self.state.d["elapsed_test_s"] += max(0.0, t - self.credit_mark)
            self.credit_mark = t

    def worker_cmdline(self, pid):
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf8", "ignore").split("\0")
        except Exception:
            return []

    def kill_stale_workers(self):
        """A supervisor crash can orphan a worker still holding 60 GiB of VRAM,
        which would make the next worker fail to allocate. Clear them first."""
        try:
            out = subprocess.run(["pgrep", "-f", str(self.dir / "gpuqual")],
                                 capture_output=True, text=True).stdout.split()
            mine = str(os.getpid())
            stale = [p for p in out if p != mine]
            # When this supervisor owns one card, kill ONLY workers on that card.
            # Two supervisors running side by side would otherwise SIGKILL each
            # other's worker on every restart -- the path pattern alone is not
            # proof of ownership.
            if self.a.device is not None:
                want = f"--device={self.a.device}"
                stale = [p for p in stale if want in self.worker_cmdline(p)]
            for p in stale:
                try:
                    os.kill(int(p), signal.SIGKILL)
                except Exception:
                    pass
            if stale:
                self.ev(ev="stale_worker_killed", pids=stale,
                        note="orphaned worker(s) from a previous supervisor were holding VRAM")
                time.sleep(3)
        except Exception as e:
            self.ev(ev="stale_worker_check_failed", msg=str(e))

    # ---------------- logging ----------------
    def ev(self, **kw):
        kw.setdefault("t", now())
        kw.setdefault("src", "supervisor")
        line = json.dumps(kw, separators=(",", ":"))
        with self.lock:
            self.events.write(line + "\n")
        if kw.get("ev") in ("CRITICAL", "finding", "xid", "context_loss", "worker_hang",
                            "run_start", "run_end", "resume", "control"):
            print(line, flush=True)

    # ---------------- telemetry ----------------
    # "index" leads so every sample is attributable to a card. With more than one
    # GPU nvidia-smi emits one line per card per interval; without the index the
    # two cards' samples are indistinguishable once written to the CSV.
    FIELDS = ("index,timestamp,temperature.gpu,temperature.memory,power.draw,power.limit,"
              "clocks.sm,clocks.mem,utilization.gpu,utilization.memory,pstate,"
              "clocks_event_reasons.active,memory.used,memory.total")

    def telemetry_thread(self):
        cmd = ["nvidia-smi", f"--query-gpu={self.FIELDS}",
               "--format=csv,noheader,nounits", "-l", "5"]
        # Scoped to this supervisor's card, so a neighbour's temperature can
        # neither halt this run nor pollute its peaks.
        if self.a.device is not None:
            cmd += ["-i", str(self.a.device)]
        if self.tele_f.tell() == 0:
            self.tele_f.write(self.FIELDS + "\n")
        while not self.stop.is_set():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    if self.stop.is_set():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    self.tele_f.write(line + "\n")
                    self.check_telemetry(line)
                p.terminate()
            except Exception as e:
                self.ev(ev="telemetry_error", msg=str(e))
            if not self.stop.is_set():
                time.sleep(5)

    def check_telemetry(self, line):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 14:
            return
        try:
            gpu = int(f[0])
            gpu_c, mem_c = int(f[2]), int(f[3])
            pwr, plim = float(f[4]), float(f[5])
        except ValueError:
            return
        st = self.state.d
        # Every threshold below is evaluated PER CARD. Tracking one shared peak
        # across two cards lets a cool card mask a hot one in the report, and
        # makes the power-limit comparison ping-pong between them.
        pg = st.setdefault("per_gpu", {}).setdefault(str(gpu), {
            "pci": self.pci.get(gpu, "unknown"),
            "peak_gpu_c": 0, "peak_mem_c": 0, "power_limit_w": 0.0})
        pg["peak_gpu_c"] = max(pg["peak_gpu_c"], gpu_c)
        pg["peak_mem_c"] = max(pg["peak_mem_c"], mem_c)
        # Flat fields stay as the worst case over all cards, so report.py keeps
        # reading a meaningful number without knowing about per_gpu.
        st["peak_gpu_c"] = max(st["peak_gpu_c"], gpu_c)
        st["peak_mem_c"] = max(st["peak_mem_c"], mem_c)
        # HARD RULE guard: the limit must never exceed the safe ceiling. Setting
        # it lower is an operator choice and is safe -- record it as a change in
        # test conditions, don't alarm, and only emit on an actual change.
        last = pg.get("power_limit_w", 0.0)
        if abs(plim - last) > 0.5:
            if plim > self.a.plimit + 0.5:
                st["tally"]["critical"] += 1
                self.ev(ev="CRITICAL", kind="power_limit_above_ceiling", gpu=gpu,
                        pci=pg["pci"], ceiling_w=self.a.plimit, observed_w=plim)
            else:
                self.ev(ev="config_change", kind="power_limit_set", gpu=gpu,
                        pci=pg["pci"], observed_w=plim, ceiling_w=self.a.plimit,
                        note="at or below safe ceiling; recorded as a test condition")
            st.setdefault("power_limit_history", []).append(
                {"t": now(), "gpu": gpu, "w": plim,
                 "elapsed_test_s": round(st["elapsed_test_s"], 1)})
            pg["power_limit_w"] = plim
            st["power_limit_w"] = plim   # most recent sample, any card
        # INDEPENDENT emergency stop. The worker governs itself via NVML; this
        # path uses nvidia-smi, so a failure of one cannot blind the other.
        # Critically this ACTS -- the previous version only logged.
        # ANY card over the limit halts the WHOLE run: the cards share a chassis
        # and an airflow path, so one overheating is not a per-card problem.
        self.last_tele = now()
        if gpu_c >= self.a.t_halt or (mem_c and mem_c >= self.a.t_mem_halt):
            st["tally"]["critical"] += 1
            self.halt_run("thermal", f"GPU{gpu} ({pg['pci']}) {gpu_c}C / HBM {mem_c}C at or "
                                     f"above halt limit ({self.a.t_halt}/{self.a.t_mem_halt}C)",
                          gpu=gpu, gpu_c=gpu_c, mem_c=mem_c, pwr_w=pwr)
        elif gpu_c >= self.a.t_critical:
            # Rate-limit the alarm PER CARD, otherwise a card sitting at 81C
            # suppresses the alarm for a second card crossing the same line.
            key = f"_last_thermal_alarm_{gpu}"
            if now() - getattr(self, key, 0) > 60:
                setattr(self, key, now())
                st["tally"]["critical"] += 1
                self.ev(ev="CRITICAL", kind="thermal_supervisor", gpu=gpu, pci=pg["pci"],
                        gpu_c=gpu_c, mem_c=mem_c, limit_c=self.a.t_critical, pwr_w=pwr)

    def halt_run(self, reason, detail, **kw):
        """Stop everything and leave the card idle. Never auto-resumes."""
        if self.halted:
            return
        self.halted = True
        st = self.state.d
        st["halted"] = True
        st["halt_reason"] = f"{reason}: {detail}"
        self.ev(ev="CRITICAL", kind="RUN_HALTED", reason=reason, detail=detail, **kw)
        self.state.save()
        self.graceful = True
        self.kill_worker(f"halt:{reason}")
        # Escalate to SIGKILL -- a wedged worker must not keep the card loaded.
        def _reap():
            time.sleep(20)
            p = self.proc
            if p and p.poll() is None:
                try:
                    p.kill()
                    self.ev(ev="worker_sigkill", note="did not exit after halt; killed")
                except Exception:
                    pass
        threading.Thread(target=_reap, daemon=True).start()
        self.stop.set()

    def tele_stall_thread(self):
        """If telemetry stops arriving we are blind. Blind means stop, not carry on."""
        while not self.stop.is_set():
            time.sleep(10)
            if self.proc is None or self.proc.poll() is not None or self.paused.is_set():
                continue
            gap = now() - self.last_tele
            if gap > 180:
                self.halt_run("telemetry_blind",
                              f"no GPU telemetry for {gap:.0f}s; cannot verify temperature")
            elif gap > 60 and now() - getattr(self, "_last_stall_warn", 0) > 60:
                self._last_stall_warn = now()
                self.ev(ev="CRITICAL", kind="telemetry_stalled", gap_s=round(gap))

    # ---------------- Xid monitor ----------------
    XID_RE = re.compile(r"NVRM:\s*Xid\s*\(PCI:([0-9a-fA-F:.]+)\):\s*(\d+),?\s*(.*)")

    def xid_thread(self):
        cmd = ["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso", "--no-pager"]
        while not self.stop.is_set():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    if self.stop.is_set():
                        break
                    m = self.XID_RE.search(line)
                    if m:
                        xid = int(m.group(2))
                        fatal = xid in XID_FATAL
                        self.state.d["tally"]["xid"] += 1
                        if fatal:
                            self.state.d["tally"]["xid_fatal"] += 1
                        self.ev(ev="xid", xid=xid, fatal=fatal,
                                meaning=XID_MEANING.get(xid, "unknown"),
                                pci=m.group(1), detail=m.group(3)[:300], raw=line.strip()[:400])
                    elif "NVRM:" in line and ("error" in line.lower() or "fail" in line.lower()):
                        self.ev(ev="nvrm_message", raw=line.strip()[:400])
                    elif "pcieport" in line and "AER" in line:
                        self.ev(ev="pcie_aer_kernel", raw=line.strip()[:400])
                p.terminate()
            except Exception as e:
                self.ev(ev="xid_monitor_error", msg=str(e))
            if not self.stop.is_set():
                time.sleep(5)

    # ---------------- PCIe AER counters ----------------
    def aer_snapshot(self):
        """AER counters for EVERY GPU present, keyed (index, pci addr, counter)."""
        out = {}
        for gpu, addr in sorted(self.pci.items()):
            base = Path("/sys/bus/pci/devices") / addr
            for name in ("aer_dev_correctable", "aer_dev_fatal", "aer_dev_nonfatal"):
                f = base / name
                if not f.exists():
                    continue
                try:
                    tot = 0
                    for ln in f.read_text().splitlines():
                        parts = ln.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            tot += int(parts[1])
                    out[(gpu, addr, name)] = tot
                except Exception:
                    pass
        return out

    def aer_thread(self):
        prev = self.aer_snapshot()
        while not self.stop.is_set():
            time.sleep(30)
            cur = self.aer_snapshot()
            for k, v in cur.items():
                if k in prev and v > prev[k]:
                    gpu, addr, name = k
                    delta = v - prev[k]
                    # endswith, not "in": "aer_dev_nonfatal" contains "fatal", so a
                    # substring test escalated every non-fatal AER error to CRITICAL.
                    is_fatal = name.endswith("_fatal")
                    key = "pcie_fatal" if is_fatal else "pcie_correctable"
                    self.state.d["tally"][key] += delta
                    self.ev(ev="CRITICAL" if is_fatal else "finding",
                            kind="PCIE_AER", gpu=gpu, pci=addr, counter=name,
                            delta=delta, total=v)
            prev = cur

    # ---------------- control file ----------------
    def control_thread(self):
        while not self.stop.is_set():
            time.sleep(2)
            if not self.control.exists():
                continue
            try:
                cmd = self.control.read_text().strip().lower()
                self.control.unlink()
            except Exception:
                continue
            if not cmd:
                continue
            self.ev(ev="control", cmd=cmd)
            if cmd == "pause":
                self.paused.set(); self.kill_worker("pause")
            elif cmd == "resume":
                self.paused.clear()
            elif cmd == "stop":
                self.graceful = True; self.stop.set(); self.kill_worker("stop")
            elif cmd == "abort":
                self.abort = True; self.graceful = True; self.stop.set(); self.kill_worker("abort")
            elif cmd == "restart":
                self.kill_worker("restart")

    def kill_worker(self, why):
        p = self.proc
        if p and p.poll() is None:
            self.ev(ev="worker_signal", why=why, pid=p.pid)
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass

    # ---------------- watchdog ----------------
    def watchdog_thread(self):
        while not self.stop.is_set():
            time.sleep(15)
            # Credit + checkpoint continuously so a kill at any moment loses at
            # most ~15s of test credit rather than the whole worker session.
            if self.proc is not None and self.proc.poll() is None and not self.paused.is_set():
                self.credit()
                self.state.save()
            # The main thread blocks reading worker stdout, so the deadline has
            # to be enforced from here.
            if self.a.deadline and now() >= self.a.deadline and self.proc is not None \
                    and self.proc.poll() is None:
                self.ev(ev="deadline_reached", deadline=self.a.deadline,
                        note="stopping worker at wall-clock deadline")
                self.kill_worker("deadline")
                continue
            if self.paused.is_set() or self.proc is None or self.proc.poll() is not None:
                continue
            idle = now() - self.last_event
            if idle > HANG_TIMEOUT:
                self.state.d["tally"]["worker_hang"] += 1
                self.ev(ev="worker_hang", kind="WORKER_HANG", idle_s=round(idle, 1),
                        note="no worker event within timeout; killing (possible kernel hang / Xid)")
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.last_event = now()

    # ---------------- worker event ingestion ----------------
    def ingest(self, obj):
        self.last_event = now()
        st = self.state.d
        e = obj.get("ev")
        obj["src"] = "worker"
        with self.lock:
            self.events.write(json.dumps(obj, separators=(",", ":")) + "\n")

        if e == "alloc":
            st["gib"] = obj.get("gib", 0)
        elif e == "triage":
            c = obj.get("class")
            if c == "HARD_FAULT":
                st["tally"]["hard"] += 1
                g = obj.get("gidx")
                if g is not None and g not in st["known_bad"]:
                    st["known_bad"].append(g)
            elif c == "SOFT_FLIP":
                st["tally"]["soft"] += 1
            elif c == "TRANSIENT_READ":
                st["tally"]["transient"] += 1
            print(json.dumps(obj, separators=(",", ":")), flush=True)
        elif e == "mismatch_batch":
            st["tally"]["mismatch"] += obj.get("count", 0)
        elif e == "finding":
            k = obj.get("kind")
            if k == "COMPUTE_SDC":
                st["tally"]["sdc"] += obj.get("threads_bad", 1)
            elif k == "THERMAL_DRIFT":
                st["tally"]["thermal_drift"] += 1
            print(json.dumps(obj, separators=(",", ":")), flush=True)
        elif e == "CRITICAL":
            st["tally"]["critical"] += 1
            print(json.dumps(obj, separators=(",", ":")), flush=True)
        elif e == "phase_end":
            # coverage is credited ONLY for completed phases
            if obj.get("phase") == "P5_retention":
                st["coverage"]["retention_bit_hours"] += obj.get("bit_hours", 0.0)
                st["coverage"]["retention_max_hold_s"] = max(
                    st["coverage"]["retention_max_hold_s"], obj.get("held_s", 0.0))
        elif e == "cycle_end":
            st["cycle"] = obj.get("cycle", st["cycle"])
            st["coverage"]["bitreads"] = obj.get("bitreads", 0) + st["coverage"].get("_base_bitreads", 0)
            st["coverage"]["alu_ops"] = obj.get("alu_ops", 0) + st["coverage"].get("_base_alu", 0)
            # Only credit cycles that ran to completion; an interrupted cycle
            # still emits cycle_end but must not inflate the coverage claim.
            if obj.get("complete", 1):
                st["coverage"]["cycles_completed"] += 1
            st["tally"]["backoff_s"] = obj.get("backoff_s", 0.0) + st["coverage"].get("_base_backoff", 0.0)
            self.credit()
            self.state.save()
        elif e == "session_end":
            # fold this session's counters into the persistent base
            cov = st["coverage"]
            cov["_base_bitreads"] = cov.get("bitreads", 0)
            cov["_base_alu"] = cov.get("alu_ops", 0)
            cov["_base_backoff"] = st["tally"].get("backoff_s", 0.0)
            self.state.save()

    # ---------------- main loop ----------------
    def run(self):
        st = self.state.d
        cur_boot = boot_id()
        if st["boot_id"] != cur_boot and st["elapsed_test_s"] > 0:
            st["reboots"] += 1
            st["boot_id"] = cur_boot
            self.ev(ev="resume", kind="RESUME_AFTER_REBOOT", reboots=st["reboots"],
                    elapsed_test_s=round(st["elapsed_test_s"], 1),
                    note="machine rebooted mid-run; retention exposure for the interrupted cycle is not credited")
        elif st["elapsed_test_s"] > 0:
            self.ev(ev="resume", kind="RESUME", elapsed_test_s=round(st["elapsed_test_s"], 1),
                    remaining_s=round(self.a.duration - st["elapsed_test_s"], 1))

        st["target_s"] = self.a.duration
        st["deadline"] = self.a.deadline
        if st.get("halted") and not self.a.clear_halt:
            self.ev(ev="CRITICAL", kind="refusing_start", reason=st.get("halt_reason"),
                    note="run was halted for safety; inspect cooling, then "
                         "'gpuqual-ctl clear-halt' to allow restart")
            self.stop.set()
            return
        if self.a.clear_halt and st.get("halted"):
            st["halted"] = False
            self.ev(ev="halt_cleared", previous_reason=st.get("halt_reason"))
            self.halted = False
        self.ev(ev="run_start", run_id=st["run_id"], duration_s=self.a.duration,
                deadline=self.a.deadline,
                deadline_utc=(time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(self.a.deadline))
                              if self.a.deadline else None),
                elapsed_test_s=round(st["elapsed_test_s"], 1), cycle=st["cycle"],
                restarts=st["restarts"], reboots=st["reboots"], pid=os.getpid(),
                gpus={str(i): a for i, a in sorted(self.pci.items())})
        # Guarding zero cards is indistinguishable from guarding a healthy one
        # unless it is said out loud.
        if not self.pci:
            self.ev(ev="CRITICAL", kind="no_gpus_enumerated",
                    note="nvidia-smi returned no GPUs; PCIe AER monitoring is inactive")
        self.state.save()

        for fn in (self.telemetry_thread, self.xid_thread, self.aer_thread,
                   self.control_thread, self.watchdog_thread, self.tele_stall_thread):
            threading.Thread(target=fn, daemon=True).start()

        while not self.stop.is_set():
            remaining = self.a.duration - st["elapsed_test_s"]
            # A wall-clock deadline ends the run regardless of accumulated test
            # time (test seconds alone cannot hit a calendar time).
            if self.a.deadline and now() >= self.a.deadline:
                self.graceful = True
                self.ev(ev="deadline_reached", deadline=self.a.deadline,
                        elapsed_test_s=round(st["elapsed_test_s"], 1))
                break
            if remaining <= 0:
                self.graceful = True
                break
            if self.paused.is_set():
                time.sleep(2)
                continue

            budget = min(remaining, SESSION_MAX)
            if self.a.deadline:
                budget = min(budget, max(60.0, self.a.deadline - now()))
            st["seed"] = (st["seed"] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            cmd = [str(self.dir / "gpuqual"),
                   f"--budget={budget:.0f}", f"--scale={self.a.scale}",
                   f"--seed={st['seed']}", f"--cycle={st['cycle']}",
                   f"--t-backoff={self.a.t_backoff}", f"--t-drift={self.a.t_drift}",
                   f"--t-critical={self.a.t_critical}", f"--plimit={self.a.plimit}",
                   f"--t-pause={self.a.t_pause}", f"--t-halt={self.a.t_halt}",
                   f"--t-resume={self.a.t_resume}",
                   f"--t-mem-pause={self.a.t_mem_pause}", f"--t-mem-halt={self.a.t_mem_halt}",
                   f"--tag=run{st['run_id']}"]
            if self.a.device is not None:
                cmd.append(f"--device={self.a.device}")
            if self.a.size_gib:
                cmd.append(f"--size-gib={self.a.size_gib}")

            self.ev(ev="worker_start", budget_s=round(budget), cmd=" ".join(cmd))
            self.kill_stale_workers()
            t0 = now()
            self.last_event = t0
            with self.credit_lock:
                self.credit_mark = t0
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1)
            except Exception as ex:
                self.ev(ev="CRITICAL", kind="worker_spawn_failed", msg=str(ex))
                time.sleep(30)
                continue

            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.ingest(json.loads(line))
                except json.JSONDecodeError:
                    self.ev(ev="worker_stderr", raw=line[:500])
            rc = self.proc.wait()
            self.credit()
            with self.credit_lock:
                self.credit_mark = None
            spent = now() - t0
            self.state.save()

            if self.stop.is_set():
                break
            if rc == 20:      # EXIT_THERMAL_HALT
                self.halt_run("thermal_worker",
                              "worker halted itself on temperature; card left idle")
                break
            if rc != 0:
                st["restarts"] += 1
                st["tally"]["context_loss"] += 1
                self.ev(ev="context_loss", kind="CONTEXT_LOSS", rc=rc,
                        ran_s=round(spent, 1), restarts=st["restarts"],
                        note="worker exited non-zero; CUDA context likely poisoned. Restarting.")
                self.state.save()
                time.sleep(10)
            else:
                self.ev(ev="worker_exit", rc=rc, ran_s=round(spent, 1),
                        elapsed_test_s=round(st["elapsed_test_s"], 1))
                if spent < 30 and not self.paused.is_set():
                    self.ev(ev="CRITICAL", kind="worker_flapping", ran_s=round(spent, 1))
                    time.sleep(30)

        st["finished"] = not self.abort and (
            st["elapsed_test_s"] >= self.a.duration
            or bool(self.a.deadline and now() >= self.a.deadline))
        self.state.save()
        self.ev(ev="run_end", elapsed_test_s=round(st["elapsed_test_s"], 1),
                target_s=self.a.duration, finished=st["finished"], aborted=self.abort,
                tally=st["tally"], coverage=st["coverage"])
        self.stop.set()
        time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(DEF_DIR))
    p.add_argument("--device", type=int, default=None,
                   help="test and monitor ONLY this GPU index. Omit to monitor every "
                        "card (the single-card default). Run one supervisor per card, "
                        "each with its own --dir, to qualify both at once.")
    p.add_argument("--duration", type=float, default=48 * 3600)
    p.add_argument("--deadline", type=float, default=0,
                   help="absolute epoch seconds; run ends at this wall-clock time "
                        "regardless of accumulated test time (0 = no deadline)")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--size-gib", type=int, default=0)
    p.add_argument("--plimit", type=float, default=140.0,
                   help="SAFE CEILING in watts. A limit above this is CRITICAL; "
                        "a limit at or below it is recorded as a test condition.")
    p.add_argument("--t-drift", type=float, default=76.0)
    p.add_argument("--t-backoff", type=float, default=78.0)
    p.add_argument("--t-critical", type=float, default=82.0)
    p.add_argument("--t-pause", type=float, default=80.0)
    p.add_argument("--t-halt", type=float, default=84.0)
    p.add_argument("--t-resume", type=float, default=70.0)
    p.add_argument("--t-mem-pause", type=float, default=84.0)
    p.add_argument("--t-mem-halt", type=float, default=88.0)
    p.add_argument("--clear-halt", action="store_true",
                   help="allow starting a run that was previously halted for safety")
    p.add_argument("--fresh", action="store_true",
                   help="discard existing state and restart the clock (requires --force)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    sp = Path(a.dir) / "logs" / "state.json"
    if a.fresh:
        if not a.force:
            print("--fresh discards accumulated test time; pass --force to confirm", file=sys.stderr)
            return 2
        if sp.exists():
            sp.rename(sp.with_name(f"state.{int(now())}.bak.json"))

    h = Harness(a)
    def sigh(sig, frm):
        h.graceful = True
        h.stop.set()
        h.kill_worker("signal")
    signal.signal(signal.SIGTERM, sigh)
    signal.signal(signal.SIGINT, sigh)
    h.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
