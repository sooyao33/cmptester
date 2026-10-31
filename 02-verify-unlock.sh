#!/usr/bin/env bash
# Stage 2 - prove the unlock after the cold boot. Measured, not just what nvidia-smi says.
# Pass criteria come from qwenian's two known-good cards (qwenian-ref/).
#   per card: patched module running, full memory reported, >= 60 GiB (64 GiB card) allocated and
#   verified twice, injected flips caught, 74 SMs, tensor cores bit-exact, HBM >= 1100 GB/s, FP32 >= 12.0 TFLOPS, no Xid
source "$(dirname "$0")/lib.sh"
need_root
D="$OUT/verify"; mkdir -p "$D"; FAILS=0
exec > >(tee "$D/summary.txt") 2>&1
say "== verify unlock $(stamp) on $(hostname)"
for t in $(seq 1 24); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 5; done   # driver may still be coming up at boot
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
[[ -z "$others" ]] && pass "no other programs using the GPUs" || fail "other programs are using the GPUs (stop miners etc.): $others"
X0=$(xid_count)

want=$(cat "$OUT/unlock/patched-srcversion.txt" 2>/dev/null || modinfo -F srcversion "/lib/modules/$(uname -r)/updates/cmpunlocker/nvidia.ko" 2>/dev/null)
have=$(cat /sys/module/nvidia/srcversion 2>/dev/null)
[[ -n "$have" && "$have" == "$want" ]] && pass "patched nvidia module running ($have)" || fail "running module '$have' != patched '$want'"
n_init_fail=$(dmesg | grep -c 'RmInitAdapter failed')
[[ $n_init_fail -eq 0 ]] && pass "no RmInitAdapter failures this boot" || fail "$n_init_fail RmInitAdapter failure(s) this boot (cards did not initialise)"
dmesg | grep -E 'SEC2_DEBUG|NVRM' > "$D/dmesg-nvrm.txt"
grep -qE 'SEC2_DEBUG' "$D/dmesg-nvrm.txt" && pass "unlock boot log present ($(grep -c SEC2_DEBUG "$D/dmesg-nvrm.txt") SEC2_DEBUG lines)" || warn "no SEC2_DEBUG lines in dmesg"

nvidia-smi -q > "$D/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,pci.bus_id,serial,uuid,vbios_version,memory.total,ecc.mode.current,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,clocks.max.sm,temperature.gpu --format=csv > "$D/cards.csv"
cat "$D/cards.csv"

for i in $(gpu_indices); do
    bus=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader); bus=${bus#0000}
    exp=$(expect_mib_for_bus "$bus"); tot=$(nvidia-smi -i "$i" --query-gpu=memory.total --format=csv,noheader,nounits)
    ecc=$(nvidia-smi -i "$i" --query-gpu=ecc.mode.current --format=csv,noheader)
    say "-- GPU $i ($bus): expect ${exp} MiB"
    [[ $exp -gt 0 && $tot -ge $exp ]] && pass "GPU $i reports ${tot} MiB" || fail "GPU $i reports ${tot} MiB (expected ${exp})"
    [[ "$ecc" == "Enabled" ]] && pass "GPU $i ECC enabled" || warn "GPU $i ECC is '$ecc' (new unlock enables it; soak still valid without)"

    "$BIN/cmpprobe" --device "$i" --inject > "$D/cmpprobe-inject-gpu$i.jsonl" 2>&1
    if grep -q '"bad_words":2,"bad_below_8gib":1,"bad_above_8gib":1,"injected":1' "$D/cmpprobe-inject-gpu$i.jsonl"; then
        pass "GPU $i fault injection: both planted bit flips caught (below and above 8 GiB)"
    else fail "GPU $i fault injection NOT caught exactly - checker untrustworthy here"; fi
    grep '"ev":"tensor"' "$D/cmpprobe-inject-gpu$i.jsonl" | grep -q '"bad_vs_cudacore":1,' \
        && pass "GPU $i tensor-core fault injection: planted wrong element caught" \
        || fail "GPU $i tensor-core checker missed the planted element"

    "$BIN/cmpprobe" --device "$i" > "$D/cmpprobe-gpu$i.jsonl" 2>&1; rc=$?
    j="$D/cmpprobe-gpu$i.jsonl"; num() { grep "\"ev\":\"$1\"" "$j" | grep -oE "\"$2\":[0-9.]+" | cut -d: -f2; }
    sms=$(num props sms); gib=$(num alloc_verify gib_verified); bad=$(num alloc_verify bad_words)
    bw=$(num hbm_bw gb_s); tf=$(num fp32 tflops); h2d=$(num pcie_bw h2d_gb_s); d2h=$(num pcie_bw d2h_gb_s)
    min_gib=$(( exp >= 65536 ? 60 : (exp >= 40960 ? 37 : 999) ))
    [[ $rc -eq 0 && "${bad:-x}" == 0 && ${gib:-0} -ge $min_gib ]] && pass "GPU $i wrote+verified ${gib} GiB twice, 0 bad words" || fail "GPU $i memory verify: ${gib:-?} GiB, ${bad:-?} bad words, rc=$rc"
    [[ ${sms:-0} -ge 74 ]] && pass "GPU $i ${sms} SMs" || { [[ ${sms:-0} -ge 70 ]] && warn "GPU $i ${sms} SMs (old unlock level)" || fail "GPU $i only ${sms:-?} SMs"; }
    awk "BEGIN{exit !(${bw:-0} >= 1100)}" && pass "GPU $i HBM ${bw} GB/s" || fail "GPU $i HBM ${bw:-?} GB/s (< 1100; qwenian 1195-1221)"
    awk "BEGIN{exit !(${tf:-0} >= 12.0)}" && pass "GPU $i FP32 ${tf} TFLOPS" || fail "GPU $i FP32 ${tf:-?} TFLOPS (< 12.0; qwenian 13.1)"
    tb=$(grep '"ev":"tensor"' "$j" | grep -oE '"bad_vs_cudacore":[0-9]+,"bad_vs_cpu":[0-9]+'); ttf=$(num tensor tflops_fp16)
    awk "BEGIN{exit !(${ttf:-0} >= 10.0)}" || fail "GPU $i tensor fp16 ${ttf:-?} TFLOPS (< 10; qwenian 12.3 with this simple kernel)"
    grep '"ev":"tensor"' "$j" | grep -q '"result":"CLEAN"' && pass "GPU $i tensor cores bit-exact over 5 x 16.7M elements ($tb), ${ttf} TFLOPS fp16" || fail "GPU $i tensor-core mismatch: $tb"
    say "INFO  GPU $i PCIe host->dev ${h2d} GB/s, dev->host ${d2h} GB/s (qwenian Gen2 x4: 0.88 / 1.68)"
done

X1=$(xid_count); [[ $X1 -eq $X0 ]] && pass "no new Xid during verification" || fail "$((X1-X0)) new Xid line(s) during verification"
say "== verify result: $([[ $FAILS -eq 0 ]] && echo ALL PASS || echo "$FAILS FAILURE(S)")"
exit $(( FAILS > 0 ))
