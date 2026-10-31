#!/usr/bin/env bash
# Stage 4 - the long soak: gpuqual supervisor (harness.py) on every card in parallel, plus a
# 30 s health sampler (temps, power, clocks, PCIe link, ECC counters) and a kernel-log capture.
#
#   sudo ./04-soak.sh start HOURS     e.g. 4  (one full cycle = 3.3 h at scale 1; shorter runs are
#                                            time-compressed so every phase incl. retention still runs)
#   sudo ./04-soak.sh status
#   sudo ./04-soak.sh stop
# Detached (setsid): survives the ssh session dropping. Thermal: backoff 74, critical 80, pause 82,
# HALT 85 C (GPU) / pause 88, HALT 92 C (HBM). The harness never changes power limits.
source "$(dirname "$0")/lib.sh"
need_root
D="$OUT/soak"; mkdir -p "$D"
cmd="${1:-status}"

case "$cmd" in
start)
    H="${2:?usage: $0 start HOURS}"
    pgrep -f '[h]arness.py --dir .*/out/soak/gpu' >/dev/null && die "a soak is already running ($0 status)"
    # one full cycle = ~3.25 h at scale 1 (measured); scale so ONE cycle just exceeds the target - the harness
    # always finishes the cycle it is in, so a cycle slightly shorter than the target would run a 2nd one
    scale=$(awk -v h="$H" 'BEGIN{s = h / 3.15; if (s > 1) s = 1; if (s < 0.05) s = 0.05; printf "%.3f", s}')
    dur=$(awk -v h="$H" 'BEGIN{printf "%d", h * 3600}')
    say "== soak start $(stamp): ${H} h per card, scale $scale"
    if [[ ! -f "$D/xid-before.txt" ]]; then   # first start only; a restart after a reboot resumes the same run
        nvidia-smi -q > "$D/nvidia-smi-q-before.txt"
        nvidia-smi --query-gpu=index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv > "$D/ecc-before.csv" 2>&1
        xid_count > "$D/xid-before.txt"
    fi
    for i in $(gpu_indices); do
        G="$D/gpu$i"; mkdir -p "$G/logs"; ln -sf "$BIN/gpuqual" "$G/gpuqual"
        pl=$(nvidia-smi -i "$i" --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
        echo "$pl" > "$G/plimit.txt"
        setsid nohup python3 -I "$KIT/harness.py" --dir "$G" --device "$i" --duration "$dur" --scale "$scale" \
            --plimit "$pl" --t-backoff 74 --t-drift 70 --t-critical 80 --t-pause 82 --t-halt 85 --t-resume 68 \
            --t-mem-pause 88 --t-mem-halt 92 > "$G/logs/supervisor.log" 2>&1 < /dev/null &
        say "GPU $i: harness pid $! (power limit ${pl} W, $dur s)"
    done
    setsid nohup bash -c "while :; do nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,clocks_event_reasons.active --format=csv,noheader >> '$D/health.csv' 2>&1; sleep 30; done" > /dev/null 2>&1 < /dev/null &
    echo $! > "$D/sampler.pid"
    setsid nohup bash -c "journalctl -k -f -o short-iso --no-pager >> '$D/kernel.log' 2>&1" > /dev/null 2>&1 < /dev/null &
    echo $! > "$D/journal.pid"
    [[ -f "$D/started" ]] || stamp > "$D/started"; echo "$H" > "$D/hours"
    say "running. check with: $0 status"
    ;;
running)   # exit 0 while any harness is alive
    pgrep -f '[h]arness.py --dir .*/out/soak/gpu' >/dev/null ;;
finished)  # exit 0 when every card's run reached its target
    for G in "$D"/gpu*; do python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' "$G/logs/state.json" 2>/dev/null || exit 1; done ;;
status)
    [[ -f "$D/started" ]] || die "no soak started"
    say "== soak started $(cat "$D/started") for $(cat "$D/hours") h; now $(stamp)"
    for G in "$D"/gpu*; do
        i=${G##*gpu}; alive=$(pgrep -f "[h]arness.py --dir $G " >/dev/null && echo running || echo stopped)
        st="$G/logs/state.json"
        python3 -I - "$st" "$alive" "$i" <<'PY'
import json, sys
p, alive, i = sys.argv[1:4]
try: s = json.load(open(p))
except Exception as e: print(f"GPU {i}: {alive}, no state yet ({e})"); sys.exit()
t = s.get("tally", {}); c = s.get("coverage", {})
bad = {k: v for k, v in t.items() if v and k != "backoff_s"}
print(f"GPU {i}: {alive}  tested {s.get('elapsed_test_s', 0)/3600:.2f} h  cycles {c.get('cycles_completed', 0)}  "
      f"bits verified {c.get('bitreads', 0):.3g}  findings: {bad or 'NONE'}")
PY
    done
    tail -n "$(gpu_indices | wc -l)" "$D/health.csv" 2>/dev/null
    say "new Xid since start: $(( $(xid_count) - $(cat "$D/xid-before.txt") ))"
    ;;
stop)
    pkill -TERM -f '[h]arness.py --dir .*/out/soak/gpu' || true
    for f in sampler journal; do [[ -f "$D/$f.pid" ]] && kill "$(cat "$D/$f.pid")" 2>/dev/null; done
    pkill -f "[j]ournalctl -k -f -o short-iso" || true
    say "stop requested $(stamp) - harnesses finish their current phase and exit (up to ~2 min)"
    ;;
*) die "usage: $0 start HOURS | status | stop" ;;
esac
