#!/usr/bin/env bash
# Stage 3 - prove the DETECTORS work on this hardware before trusting a clean soak.
#   gpuqual --self-test : plants a soft flip, a stuck cell and a compute error every cycle;
#                         each must be classified SOFT_FLIP / HARD_FAULT / COMPUTE_SDC
#   vramcheck --inject  : plants 1 corrupted element; must exit 1 (FAILED)
# ~3 min, all cards in parallel.
source "$(dirname "$0")/lib.sh"
need_root
D="$OUT/selftest"; mkdir -p "$D"; FAILS=0
exec > >(tee "$D/summary.txt") 2>&1
say "== detector self-test $(stamp)"
X0=$(xid_count); PIDS=()
for i in $(gpu_indices); do
    pl=$(nvidia-smi -i "$i" --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
    timeout 900 "$BIN/gpuqual" --self-test --size-gib=4 --scale=0.004 --budget=120 --device="$i" --plimit="$pl" \
        --t-backoff=74 --t-critical=80 --t-pause=82 --t-halt=85 --t-resume=68 --tag="selftest$i" \
        > "$D/gpuqual-selftest-gpu$i.jsonl" 2>&1 &
    PIDS+=($!)
done
wait "${PIDS[@]}"   # not a bare wait: that also waits on the tee process substitution and hangs
for i in $(gpu_indices); do
    f="$D/gpuqual-selftest-gpu$i.jsonl"
    cyc=$(grep -c '"ev":"cycle_end"' "$f"); h=$(grep -c '"class":"HARD_FAULT"' "$f"); s=$(grep -c '"class":"SOFT_FLIP"' "$f")
    c=$(grep -c '"kind":"COMPUTE_SDC"' "$f"); inj=$(grep -c '"ev":"self_test_inject"' "$f")
    if [[ $cyc -ge 1 && $h -ge $cyc && $s -ge $cyc && $c -ge $cyc ]]; then
        pass "GPU $i gpuqual: $cyc cycles, planted $inj x (soft+hard) + SDC -> caught $s SOFT_FLIP, $h HARD_FAULT, $c COMPUTE_SDC"
    else fail "GPU $i gpuqual self-test: cycles=$cyc soft=$s hard=$h sdc=$c (each must be >= cycles)"; fi
done
for i in $(gpu_indices); do
    CUDA_VISIBLE_DEVICES=$i timeout 1200 "$BIN/vramcheck" --inject 1 --rounds 1 --iters 64 > "$D/vramcheck-inject-gpu$i.txt" 2>&1; rc=$?
    [[ $rc -eq 1 ]] && grep -q 'RESULT: FAILED' "$D/vramcheck-inject-gpu$i.txt" \
        && pass "GPU $i vramcheck: planted corruption detected ($(grep -oE '[0-9]+ error' "$D/vramcheck-inject-gpu$i.txt" | head -1)s)" \
        || fail "GPU $i vramcheck did not flag the planted corruption (rc=$rc)"
done
X1=$(xid_count); [[ $X1 -eq $X0 ]] && pass "no Xid during self-test" || fail "$((X1-X0)) Xid line(s) during self-test"
say "== self-test result: $([[ $FAILS -eq 0 ]] && echo ALL PASS || echo "$FAILS FAILURE(S)")"
exit $(( FAILS > 0 ))
