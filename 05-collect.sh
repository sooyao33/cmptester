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
if [[ -s "$D/health.csv" ]]; then
    echo; echo "## Health sampler (30 s)"
    awk -F', ' '{g=$2; t=$3+0; m=$4+0; if(t>mt[g])mt[g]=t; if(m>mm[g])mm[g]=m; n[g]++; if($11!=""&&$11!=gen0[g]){if(gen0[g]!="")chg[g]++; gen0[g]=$11}}
         END{for(g in n) printf "- GPU %s: %d samples, peak GPU %d C, peak HBM %d C, PCIe gen changes %d\n", g, n[g], mt[g], mm[g], chg[g]+0}' "$D/health.csv"
fi
} > "$P" 2>&1
cd "$OUT/.." && T="cmp-test-$(hostname)-$(date -u +%Y%m%dT%H%MZ).tar.gz"
find out -type f ! -name '*.run' ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > out/SHA256SUMS
tar --exclude='*.run' -czf "$T" out
say "evidence: $P"; say "bundle:   $KIT/$T  (sha256 $(sha256sum "$T" | cut -c1-16)...)"
grep -E '^\*\*.*(GO|INCOMPLETE|INVESTIGATE)' "$P" | head; grep -c '^FAIL' "$P" | xargs -I{} echo "FAIL lines: {}"
