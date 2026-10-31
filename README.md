# CMP 170HX unlock + stability test

Unlocks NVIDIA CMP 170HX cards with [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
(latest version) and then proves, with measurements and logs, that the unlock works and the cards
are stable. Ubuntu/Debian, x86-64, run as root. Needs internet. Secure Boot must be OFF.

## How to run (seller)

```bash
git clone https://github.com/sooyao33/cmptester.git
cd cmptester
sudo ./run.sh 4          # 4 = soak hours per card (all cards run in parallel); default 4
```

1. It installs the build tools, the NVIDIA open driver and the latest cmpunlocker (10-20 min).
   Stop any miner or other GPU program first.
2. **The machine powers itself OFF** (the unlock needs a cold boot). **Turn it back on with the power
   button.** If it has not switched off after 3 minutes, hold the power button for 10 seconds.
3. After boot everything continues by itself: unlock check (~5 min), error-detector self-test
   (~5 min), stability soak (the hours you chose), report.
4. When finished, send back the file **`~/cmp-test-results-<hostname>.tar.gz`** (also in `/root`).

Check progress any time: `sudo /opt/cmp-test/run.sh status` (or `... log` to follow live).
Closing the terminal does not stop it (it runs as the `cmp-test` systemd service).
If a reboot is needed first (e.g. the old `nouveau` driver is in use) it reboots and continues by itself.

---

## Details

Unlock the seller's CMP 170HX cards and prove, with measurements and logs, that the unlock works
and the cards are stable. Every check was validated on two known-good unlocked cards (2026-10-31),
including deliberate fault injection: the detectors must catch planted faults before a clean
result means anything. Reference numbers are in `qwenian-ref/`.

Binaries are prebuilt for sm_80 with
CUDA 13.3 (static runtime; needs the 610.x driver, which stage 1 installs). Sources are in `src/`.

| Stage | Command | Changes the box? | Time |
|---|---|---|---|
| 0 recon | `sudo ./00-recon.sh` | no (read-only) | 10 s |
| 1 unlock | `sudo ./01-unlock.sh` | yes: build tools, nvidia-open 610.43.03 (or the newest cmpunlocker supports), latest cmpunlocker (`--no-passthrough --no-iommu`) | 10-20 min |
| -- | **COLD power cycle** (`shutdown -h now`, then the seller powers it on) | | |
| 2 verify | `sudo ./02-verify-unlock.sh` | no | ~5 min |
| 3 self-test | `sudo ./03-selftest.sh` | no | ~5 min |
| 4 soak | `sudo ./04-soak.sh start HOURS` / `status` / `stop` | no | HOURS |
| 5 evidence | `sudo ./05-collect.sh` -> `out/PROOF.md` + tarball with SHA256SUMS | no | 1 min |

## What is proven, and how

| Claim | Evidence | Pass criterion (from qwenian) |
|---|---|---|
| Patched driver is the one running | `/sys/module/nvidia/srcversion` = built module; SEC2_DEBUG boot log; no RmInitAdapter failure | exact match |
| Full memory is real, not just reported | cmpprobe allocates every 1 GiB chunk, writes an address-derived pattern, verifies with uncached reads, 2 passes, inverted data | >= 60 GiB, 0 bad words (qwenian: 61 GiB) |
| Checker can see faults | `--inject` plants flips below and above 8 GiB | exactly 2 caught |
| All SMs unlocked | CUDA SM count, FP32 FMA throughput | 74 SMs, >= 12.0 TFLOPS (qwenian 13.1) |
| HBM at full speed | copy kernel bandwidth | >= 1100 GB/s (qwenian 1195-1221) |
| Tensor cores correct | FP16 WMMA matmul on exact integers vs independent CUDA-core matmul, all 16.7M elements x 5 runs, + CPU spot checks | bit-exact (tested on qwenian; see qwenian-ref) |
| Detectors work on THIS box | gpuqual `--self-test` (soft flip, stuck cell, compute SDC every cycle), vramcheck `--inject` | every planted fault classified correctly |
| Stable over time | gpuqual harness: golden recompute, moving inversions, RMW hammer, random scatter, ALU known-answer soak, retention hold at thermal ceiling; Xid + PCIe AER watch; 30 s health log incl. ECC counters | report verdict not NO-GO; 0 mismatches; 0 Xid; ECC counters unchanged |

ECC note: the new unlock turns ECC on. ECC silently corrects single-bit flips, so a weak card can
look clean to the pattern tests. That is why the sampler logs ECC counters and `05-collect.sh`
compares them before/after: a rising corrected count is a finding even when gpuqual is clean.

Soak length: one full gpuqual cycle is 3.3 h at scale 1. Shorter runs are time-compressed so every
phase (including the retention hold) still runs; longer is stronger evidence. Thermal halt is 85 C
GPU / 92 C HBM and the harness never touches power limits (it records the limit it found).

Safety: stage 1 refuses on qwenian, refuses with Secure Boot on, and stops for a human decision if a
different NVIDIA driver is installed (`REPLACE_DRIVER=1` to replace it). It never edits GRUB.
After a failed driver load, never warm-reboot: power-cycle (a warm reboot wedged qwenian).

Undo on the seller box: `cd /opt/cmpunlocker && sudo ./remove.sh --yes`, then
`sudo nvidia-uninstall` if they want the driver gone too, then reboot.
