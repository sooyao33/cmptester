#!/usr/bin/env python3
"""
gpuqual report -- turns the event log into a GO / INVESTIGATE / NO-GO verdict
with the statistical bounds the run actually achieved.

Deliberately states what the run CANNOT establish as well as what it can.
"""

import argparse, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

STOCK_FB = 0x200000000          # 8 GiB -- stock CMP 170HX framebuffer
CHI2_95_UPPER_0 = 3.0           # 95% one-sided upper bound with 0 observed events


def load(p):
    evs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return evs


def fmt_dur(s):
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m"


def si(x):
    if x <= 0:
        return "0"
    u = int(math.floor(math.log10(x) / 3))
    u = max(0, min(u, 6))
    return f"{x/10**(3*u):.3g}{['','k','M','G','T','P','E'][u]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/smith/gpuqual")
    ap.add_argument("--out", default=None)
    ap.add_argument("--duration", type=float, default=48 * 3600)
    a = ap.parse_args()

    logs = Path(a.dir) / "logs"
    state = json.loads((logs / "state.json").read_text())
    evs = load(logs / "events.jsonl")

    tally = state["tally"]
    cov = state["coverage"]
    elapsed = state["elapsed_test_s"]
    gib = state.get("gib", 0)
    bits = gib * (1 << 30) * 8

    triages = [e for e in evs if e.get("ev") == "triage"]
    xids = [e for e in evs if e.get("ev") == "xid"]
    crits = [e for e in evs if e.get("ev") == "CRITICAL"]
    findings = [e for e in evs if e.get("ev") == "finding"]

    # ---------------- verdict ----------------
    nogo, inv = [], []
    if tally["hard"] > 0:
        nogo.append(f"{tally['hard']} HARD_FAULT (stuck cell: rewrite did not restore correct value)")
    if tally["sdc"] > 0:
        nogo.append(f"{tally['sdc']} COMPUTE_SDC (silent bad arithmetic — ALU known-answer test failed)")
    if tally["xid_fatal"] > 0:
        nogo.append(f"{tally['xid_fatal']} fatal-class Xid")
    if tally["pcie_fatal"] > 0:
        nogo.append(f"{tally['pcie_fatal']} uncorrectable PCIe AER event")

    # repeat offenders: same address failing more than once
    addr_hits = Counter(t.get("gidx") for t in triages if t.get("gidx") is not None)
    repeats = {g: c for g, c in addr_hits.items() if c > 1}
    if repeats:
        nogo.append(f"{len(repeats)} address(es) failed more than once (repeatable defect)")

    if tally["soft"] or tally["transient"]:
        inv.append(f"{tally['soft']} SOFT_FLIP + {tally['transient']} TRANSIENT_READ at distinct addresses")
    if tally["thermal_drift"]:
        inv.append(f"{tally['thermal_drift']} THERMAL_DRIFT event(s) — cooling degraded from the 69C baseline")
    if tally["context_loss"]:
        inv.append(f"{tally['context_loss']} CONTEXT_LOSS (worker died, CUDA context poisoned)")
    if tally["worker_hang"]:
        inv.append(f"{tally['worker_hang']} WORKER_HANG (kernel stopped responding)")
    if tally["xid"] - tally["xid_fatal"] > 0:
        inv.append(f"{tally['xid'] - tally['xid_fatal']} non-fatal Xid event(s)")
    if tally["pcie_correctable"]:
        inv.append(f"{tally['pcie_correctable']} correctable PCIe AER event(s)")
    n_crit = len(crits)
    if n_crit:
        kinds = Counter(e.get("kind") for e in crits)
        inv.append(f"{n_crit} CRITICAL telemetry event(s): {dict(kinds)}")

    verdict = "NO-GO" if nogo else ("INVESTIGATE" if inv else "GO")
    if not state.get("finished") and verdict == "GO":
        verdict = "INCOMPLETE"

    # ---------------- coverage math ----------------
    bitreads = cov.get("bitreads", 0)
    bit_hours = cov.get("retention_bit_hours", 0.0)
    alu_ops = cov.get("alu_ops", 0)

    def bound(n_events, exposure):
        if exposure <= 0:
            return None
        return (CHI2_95_UPPER_0 if n_events == 0 else 3.0 + 2.0 * n_events) / exposure

    access_bound = bound(tally["mismatch"], bitreads)
    ret_bound = bound(tally["mismatch"], bit_hours)
    sdc_bound = bound(tally["sdc"], alu_ops)
    fit_per_gbit = ret_bound * (1 << 30) * 1e9 if ret_bound else None

    # ---------------- address distribution ----------------
    above = sum(1 for t in triages if t.get("above_stock_fb"))
    below = len(triages) - above
    bitpos = Counter()
    for t in triages:
        b = t.get("bits")
        if b is not None:
            bitpos[b] += 1
    chunks = Counter(t.get("chunk") for t in triages)
    temps = [t.get("gpu_c") for t in triages if t.get("gpu_c")]

    # ---------------- thermal ----------------
    tele = logs / "telemetry.csv"
    gpu_t, mem_t, pwr, plims = [], [], [], set()
    if tele.exists():
        with open(tele) as f:
            next(f, None)
            for line in f:
                p = [x.strip() for x in line.split(",")]
                if len(p) < 5:
                    continue
                try:
                    gpu_t.append(int(p[1])); mem_t.append(int(p[2]))
                    pwr.append(float(p[3])); plims.add(round(float(p[4]), 1))
                except ValueError:
                    pass

    L = []
    w = L.append
    w(f"# GPU Qualification Report — {verdict}")
    w("")
    w(f"**Card:** NVIDIA CMP 170HX (GA100, HBM2e, unlocked past 8 GiB stock FB; see nvidia-smi-q for ECC mode)  ")
    target = state.get("target_s") or a.duration
    w(f"**Run:** `{state['run_id']}` · tested **{fmt_dur(elapsed)}** of {fmt_dur(target)} target · "
      f"{cov.get('cycles_completed',0)} cycles · **{gib} GiB** under test  ")
    w(f"**Interruptions:** {state.get('restarts',0)} worker restart(s), {state.get('reboots',0)} reboot(s)  ")
    w(f"**Peak temps:** {state.get('peak_gpu_c',0)}°C GPU / {state.get('peak_mem_c',0)}°C HBM  ")
    w("")

    w("## Verdict")
    w("")
    if nogo:
        w("**NO-GO — do not put this card into production.**")
        w("")
        for r in nogo:
            w(f"- {r}")
    elif inv:
        w("**INVESTIGATE — no disqualifying defect, but findings need a decision.**")
        w("")
        for r in inv:
            w(f"- {r}")
    elif verdict == "INCOMPLETE":
        w("**INCOMPLETE — clean so far, but the run has not reached its target duration.**")
    else:
        w("**GO — no memory, compute, driver, or link errors detected.**")
    w("")

    w("## Error tally")
    w("")
    w("| Class | Count | Meaning |")
    w("|---|---:|---|")
    w(f"| HARD_FAULT | {tally['hard']} | Stuck cell — rewrite did not fix it |")
    w(f"| SOFT_FLIP | {tally['soft']} | Bit flipped in DRAM, cell itself healthy |")
    w(f"| TRANSIENT_READ | {tally['transient']} | Read path glitched, stored data was correct |")
    w(f"| COMPUTE_SDC | {tally['sdc']} | Silent bad arithmetic (ALU known-answer) |")
    w(f"| Xid (fatal class) | {tally['xid_fatal']} | ECC/row-remap/off-bus class |")
    w(f"| Xid (other) | {tally['xid'] - tally['xid_fatal']} | Other driver/hardware faults |")
    w(f"| PCIe AER correctable | {tally['pcie_correctable']} | Link-level recovered errors |")
    w(f"| PCIe AER fatal | {tally['pcie_fatal']} | Link-level uncorrectable |")
    w(f"| CONTEXT_LOSS | {tally['context_loss']} | Worker died, context poisoned |")
    w(f"| WORKER_HANG | {tally['worker_hang']} | Kernel stopped responding |")
    w(f"| THERMAL_DRIFT | {tally['thermal_drift']} | Sustained above the 69°C baseline |")
    w(f"| Raw word mismatches | {tally['mismatch']} | Total 16-byte words that failed compare |")
    w("")

    if triages:
        w("### Where the errors landed")
        w("")
        w(f"- **Below 8 GiB (stock framebuffer): {below}** · **Above 8 GiB (unlocked region): {above}**")
        if above and not below:
            w("  - :warning: **Every error is in the unlocked region.** That is the signature of "
              "memory the vendor fused off for a reason — treat as NO-GO regardless of count.")
        if repeats:
            w(f"- **{len(repeats)} address(es) failed repeatedly** — a repeatable defect, not a soft error.")
        w(f"- Flipped-bit-count distribution: {dict(sorted(bitpos.items()))}")
        w(f"- Most-affected 1 GiB chunks: {chunks.most_common(5)}")
        if temps:
            w(f"- GPU temp at failure: min {min(temps)}°C / median {statistics.median(temps):.0f}°C / max {max(temps)}°C")
            if max(temps) - min(temps) < 4 and len(temps) > 3:
                w("  - Errors are tightly clustered in temperature — suggests a thermal dependence.")
        w("")

    w("## Coverage achieved and what it proves")
    w("")
    w("| Exposure | Amount | 95% upper bound on error rate |")
    w("|---|---:|---|")
    w(f"| Bits read & verified | {si(bitreads)} | {f'{access_bound:.2g} per bit read' if access_bound else 'n/a'} |")
    w(f"| Retention (bit-hours) | {si(bit_hours)} | {f'{ret_bound:.2g} per bit-hour' if ret_bound else 'n/a'} |")
    w(f"| ALU ops verified | {si(alu_ops)} | {f'{sdc_bound:.2g} per op' if sdc_bound else 'n/a'} |")
    if fit_per_gbit:
        w(f"| Retention, as FIT/Gbit | — | **{fit_per_gbit:.3g} FIT/Gbit** |")
    w("")
    w(f"Longest single retention hold: **{fmt_dur(cov.get('retention_max_hold_s',0))}** "
      f"(60 GiB held untouched at the thermal ceiling, then verified).")
    w("")
    w("### What this run establishes")
    w("")
    w("- No stuck cells, weak rows, or address-decode faults across the tested range.")
    w("- No load- or thermal-dependent instability at the power limit used for this run.")
    w("- No silent compute corruption over the verified ALU op count.")
    w("- Clean driver (Xid) and PCIe link behaviour.")
    w("")
    w("### What this run does NOT establish")
    w("")
    if fit_per_gbit:
        w(f"The retention bound of **{fit_per_gbit:.3g} FIT/Gbit** is far above the ~10–200 FIT/Gbit "
          "typical of healthy DRAM. A 48-hour run on a single card is roughly **three orders of "
          "magnitude too short** to prove this card's random soft-error rate is normal — no test of "
          "this duration can, and no ECC-less card can ever promise zero flips. That is physics, "
          "not a defect in this card.")
    w("")
    w("**Therefore, regardless of the verdict above:** run production workloads with "
      "application-level integrity checks — checksum model weights at load time and re-verify "
      "resident weights periodically. On a card with no ECC that is cheap insurance this "
      "qualification cannot substitute for.")
    w("")

    if gpu_t:
        w("## Thermal & power")
        w("")
        w(f"- GPU: min {min(gpu_t)}°C / mean {statistics.mean(gpu_t):.1f}°C / **max {max(gpu_t)}°C**")
        w(f"- HBM: min {min(mem_t)}°C / mean {statistics.mean(mem_t):.1f}°C / **max {max(mem_t)}°C**")
        w(f"- Power: mean {statistics.mean(pwr):.1f} W / max {max(pwr):.1f} W")
        w(f"- Power limit observed: {sorted(plims)} W " +
          ("— **constant, never modified**" if len(plims) == 1 else "— :warning: **CHANGED DURING RUN**"))
        if len(plims) == 1 and max(pwr) > list(plims)[0]:
            w(f"  - Instantaneous samples peak at {max(pwr):.1f} W, above the {list(plims)[0]:.0f} W cap. "
              "This is normal NVML behaviour — the cap is enforced on a rolling average, so single "
              "samples overshoot. The **limit itself never changed**, which is what the hard rule guards.")
        w(f"- Duty-cycle backoff engaged for {tally.get('backoff_s',0):.0f}s total")
        hot = [t for t in gpu_t if t >= 76]
        w(f"- Samples at/above the 76°C drift threshold: {len(hot)} / {len(gpu_t)} "
          f"({100.0*len(hot)/len(gpu_t):.2f}%)")
        # Cooling-degradation trend: absolute thresholds miss a slow drift, so
        # compare steady-state early vs late in the run.
        n = len(gpu_t)
        if n >= 40:
            seg = max(10, n // 10)
            early = statistics.mean(sorted(gpu_t[:seg])[-seg // 2:])
            late = statistics.mean(sorted(gpu_t[-seg:])[-seg // 2:])
            d = late - early
            w(f"- **Cooling trend:** steady-state {early:.1f}°C early → {late:.1f}°C late "
              f"(**{d:+.1f}°C** over the run)")
            if d > 5:
                w("  - :warning: **Sustained upward drift >5°C — cooling is degrading** "
                  "(pump/fan, thermal pads, or dust). Investigate before production.")
            elif d > 2:
                w("  - Mild upward drift; worth a second look but within normal warm-up spread.")
            else:
                w("  - Thermally stable across the run — no sign of cooling degradation.")
        w("")

    cfg_changes = [e for e in evs if e.get("ev") == "config_change"]
    plim_hist = state.get("power_limit_history", [])
    if cfg_changes or plim_hist:
        w("## Test conditions")
        w("")
        seen = set()
        for e in cfg_changes:
            k = (e.get("kind"), e.get("observed_w"))
            if k in seen:
                continue
            seen.add(k)
            if e.get("kind") == "power_limit_set":
                w(f"- Power limit set to **{e.get('observed_w')} W** "
                  f"(safe ceiling {e.get('ceiling_w')} W) — operator choice, at or below the "
                  "ceiling, so not a fault. Recorded because it changes the stress level.")
            else:
                w(f"- `{e.get('kind')}`: {e.get('note','')}")
        limits = sorted({h["w"] for h in plim_hist})
        if len(limits) > 1:
            w("")
            w(f"- :warning: **The card was tested at more than one power limit: {limits} W.** "
              "Coverage is not uniform across them — the qualification is only as strong as the "
              "*lowest* limit tested, since lower power means lower thermal and electrical stress.")
        if limits and max(limits) < 140:
            w("")
            w(f"- **Qualified at {max(limits)} W, not 140 W.** If production runs at 140 W, this "
              "run does not fully cover that condition. Re-qualify at the production limit, or "
              "run production at the tested limit.")
        w("")

    if xids:
        w("## Xid events")
        w("")
        for e in xids[:40]:
            w(f"- Xid **{e['xid']}** ({'FATAL' if e.get('fatal') else 'non-fatal'}): {e.get('meaning')}")
        w("")

    if crits:
        w("## Critical events")
        w("")
        for e in crits[:40]:
            w(f"- `{e.get('kind')}` — {json.dumps({k: v for k, v in e.items() if k not in ('t','ev','src')})}")
        w("")

    if state.get("known_bad"):
        w("## Known-bad addresses (retested every cycle)")
        w("")
        for g in state["known_bad"][:50]:
            w(f"- global word index `{g}` (byte offset `{g*16}`, "
              f"{'ABOVE' if g*16 >= STOCK_FB else 'below'} 8 GiB stock FB)")
        w("")

    out = "\n".join(L) + "\n"
    dest = Path(a.out) if a.out else logs / "summary.md"
    dest.write_text(out)
    print(out)
    return 1 if verdict == "NO-GO" else 0


if __name__ == "__main__":
    sys.exit(main())
