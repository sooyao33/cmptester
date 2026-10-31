#!/usr/bin/env bash
# Stage 0 - READ-ONLY inventory of the seller's box. Changes nothing.
# Output: out/recon/  (+ a summary on stdout)
source "$(dirname "$0")/lib.sh"
D="$OUT/recon"; mkdir -p "$D"
R() { local f="$1"; shift; { echo "\$ $*"; "$@"; } > "$D/$f" 2>&1 || true; }

R os.txt          bash -c 'hostnamectl; cat /etc/os-release; uname -a; uptime'
R cpu-mem.txt     bash -c 'lscpu; free -h; df -h'
R lspci-nvidia.txt bash -c 'lspci -Dnn | grep -i 10de'
for b in $(cmp_bdfs); do R "lspci-vvv-${b//:/_}.txt" lspci -vvv -s "$b"; done
R secureboot.txt  bash -c 'ls -d /sys/firmware/efi 2>&1; mokutil --sb-state 2>&1'
R cmdline.txt     cat /proc/cmdline
R modules.txt     bash -c 'lsmod | grep -E "^(nvidia|nouveau|vfio)"; cat /proc/driver/nvidia/version; cat /sys/module/nvidia/srcversion; modinfo -F filename nvidia'
R nvidia-smi-q.txt nvidia-smi -q
R nvidia-smi.txt  nvidia-smi
R packages.txt    bash -c 'dpkg -l 2>/dev/null | grep -iE "nvidia|cuda|linux-(image|headers)-$(uname -r)"; rpm -qa 2>/dev/null | grep -iE "nvidia|cuda"; dkms status 2>&1; ls /usr/src | grep -i nvidia; ls /lib/modules/$(uname -r)/updates 2>&1'
R headers.txt     bash -c 'ls -d /lib/modules/$(uname -r)/build && echo HEADERS_OK; gcc --version | head -1; make --version | head -1; python3 -c "import yaml; print(\"pyyaml ok\")"'
R kernel-nvrm.txt bash -c 'dmesg | grep -E "NVRM|nvidia|SEC2_DEBUG|AER" | tail -300'
R xid-history.txt bash -c 'journalctl -k --no-pager 2>/dev/null | grep -E "NVRM: Xid" | tail -100'
R sensors.txt     bash -c 'sensors 2>&1'
R net.txt         bash -c 'for u in https://github.com https://download.nvidia.com; do curl -sS -o /dev/null -m 10 -w "%{http_code} $u\n" $u; done'
R cmpunlocker.txt bash -c 'for d in /root/cmpunlocker /opt/cmpunlocker $(find / -maxdepth 4 -type d -name cmpunlocker 2>/dev/null | head -5); do [ -d $d/.git ] && git -C $d log --oneline -3 && echo "^ $d"; done; cat /lib/modules/$(uname -r)/updates/cmpunlocker/{driver_version,card_profile,unlock_geometry,gpu_inventory} 2>&1'

echo "=== RECON SUMMARY ($(stamp)) ==="
grep -hE '^PRETTY_NAME' "$D/os.txt"; echo "kernel: $(uname -r)"
echo "CMP cards: $(cmp_bdfs | wc -l)"; lspci -nn | grep -iE '10de:(20c2|2082|20b0)'
echo "driver: $(grep -oE 'Kernel Module.*' /proc/driver/nvidia/version 2>/dev/null || echo 'none loaded')"
echo "secure boot: $(mokutil --sb-state 2>/dev/null || echo 'unknown/BIOS')"
grep -q HEADERS_OK "$D/headers.txt" && echo "headers: ok" || echo "headers: MISSING for $(uname -r)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.total,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current,pcie.link.width.max,power.limit,temperature.gpu --format=csv 2>/dev/null
echo "past Xid lines: $(grep -c 'Xid' "$D/xid-history.txt")"
cat "$D/net.txt" | grep -v '^\$'
echo "full output: $D"
