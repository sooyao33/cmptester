#!/usr/bin/env bash
# Stage 5 - turn everything into evidence: per-card gpuqual reports, before/after ECC + Xid,
# one PROOF.md with every PASS/FAIL line, and a tarball with SHA256SUMS.
#   sudo ./05-collect.sh        (run after the soak finished, or after ./04-soak.sh stop)
source "$(dirname "$0")/lib.sh"
need_root
D="$OUT/soak"
if pgrep -f '[h]arness.py --dir .*/out/soak/gpu' >/dev/null; then
    die "soak still running - wait for it, or ./04-soak.sh stop first"
fi
for f in sampler journal; do [[ -f "$D/$f.pid" ]] && kill "$(cat "$D/$f.pid")" 2>/dev/null; done
pkill -f "[j]ournalctl -k -f -o short-iso" 2>/dev/null

nvidia-smi -q > "$D/nvidia-smi-q-after.txt" 2>&1
nvidia-smi --query-gpu=index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv > "$D/ecc-after.csv" 2>&1
H=$(cat "$D/hours" 2>/dev/null || echo 0)
P="$OUT/PROOF.md"
{
echo "# CMP 170HX test evidence - $(hostname) - $(stamp)"
echo
echo '```'
grep -E '^(PRETTY_NAME|.*Linux )' "$OUT/recon/os.txt" 2>/dev/null | head -2
cat "$OUT/verify/cards.csv" 2>/dev/null
echo '```'
for s in verify selftest; do
    echo; echo "## Stage: $s"; echo '```'
    grep -E '^(PASS|FAIL|WARN|INFO|==)' "$OUT/$s/summary.txt" 2>/dev/null || echo "(stage not run)"
    echo '```'
done
echo; echo "## Stage: soak (${H} h per card target)"
for G in "$D"/gpu*; do
    i=${G##*gpu}
    python3 -I "$KIT/report.py" --dir "$G" --duration "$(awk -v h="$H" 'BEGIN{print h*3600}')" --out "$G/summary.md" >/dev/null 2>&1
    echo; echo "### GPU $i"; sed -n '1,/^## Coverage/p' "$G/summary.md" | sed '$d'
done
echo; echo "## ECC counters (aggregate) before -> after"; echo '```'; cat "$D/ecc-before.csv" "$D/ecc-after.csv" 2>/dev/null; echo '```'
xb=$(cat "$D/xid-before.txt" 2>/dev/null || echo 0); xa=$(xid_count)
echo; echo "## Kernel"; echo "- new \`NVRM: Xid\` lines during the soak: **$((xa - xb))**"
echo "- kernel log lines captured: $(wc -l < "$D/kernel.log" 2>/dev/null || echo 0); Xid/AER in it: $(grep -cE 'Xid|AER' "$D/kernel.log" 2>/dev/null || echo 0)"
echo; echo "## Performance and power summary"
echo
echo "| GPU | SMs | HBM GB/s | FP32 TFLOPS | FP16 tensor TFLOPS | PCIe H2D / D2H GB/s | avg / max power under load (limit) | peak / avg GPU C | peak / avg HBM C | SM MHz under load (min / avg) | throttle seen | samples |"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
for i in $(gpu_indices); do
    j="$OUT/verify/cmpprobe-gpu$i.jsonl"
    v() { grep "\"ev\":\"$1\"" "$j" 2>/dev/null | grep -oE "\"$2\":[0-9.]+" | cut -d: -f2; }
    h=$(awk -F', ' -v g="$i" '$2==g && ($9+0)>=90 {n++; p=$5+0; ps+=p; if(p>pm)pm=p; lim=$6+0; t=$3+0; ts+=t; if(t>tm)tm=t; m=$4+0; ms+=m; if(m>mm)mm=m;
            c=$7+0; cs+=c; if(cmin==""||c<cmin)cmin=c; if($15!="0x0000000000000000" && $15!="")th[$15]++}
         END{if(!n){print "| no load samples | | | | | |"; exit}
             r=""; for(k in th) r=r k "(" th[k] ") "; if(r=="") r="none";
             printf "| %.0f / %.0f W (%.0f W) | %d / %.0f | %d / %.0f | %d / %.0f | %s | %d |", ps/n, pm, lim, tm, ts/n, mm, ms/n, cmin, cs/n, r, n}' "$D/health.csv" 2>/dev/null)
    echo "| $i | $(v props sms) | $(v hbm_bw gb_s) | $(v fp32 tflops) | $(v tensor tflops_fp16) | $(v pcie_bw h2d_gb_s) / $(v pcie_bw d2h_gb_s) ${h:-| (no health log) | | | | | |}"
done
echo
echo "Throttle codes (nvidia-smi clocks_event_reasons): 0x4 = power cap (normal at full load), 0x20/0x40 = thermal slowdown, 0x8 = HW slowdown. Raw 30 s log: out/soak/health.csv"
} > "$P" 2>&1
cd "$OUT/.." && T="cmp-test-$(hostname)-$(date -u +%Y%m%dT%H%MZ).tar.gz"
find out -type f ! -name '*.run' ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > out/SHA256SUMS
tar --exclude='*.run' -czf "$T" out
say "evidence: $P"; say "bundle:   $KIT/$T  (sha256 $(sha256sum "$T" | cut -c1-16)...)"
grep -E '^\*\*.*(GO|INCOMPLETE|INVESTIGATE)' "$P" | head; grep -c '^FAIL' "$P" | xargs -I{} echo "FAIL lines: {}"
