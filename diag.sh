#!/usr/bin/env bash
# Quick diagnosis when nvidia-smi shows no cards after the unlock.
#   cd ~/cmptester && git pull && sudo ./diag.sh
# Prints a verdict and writes ~/cmp-diag.tar.gz (send that file back).
[[ $EUID -eq 0 ]] || { echo "run with sudo:  sudo $0"; exit 1; }
D=/tmp/cmpdiag; rm -rf $D; mkdir -p $D
K=$(uname -r)
dmesg > $D/dmesg.txt 2>&1
uname -a > $D/uname.txt
lspci -nnvv -d 10de: > $D/lspci.txt 2>&1
nvidia-smi > $D/smi.txt 2>&1
lsmod | grep -i -E 'nvidia|nouveau' > $D/lsmod.txt
cat /proc/cmdline > $D/cmdline.txt
ls -la /lib/modules/*/updates/cmpunlocker/ > $D/patched-modules.txt 2>&1
systemctl status gen2.service cmp-test.service --no-pager > $D/services.txt 2>&1

want=$(modinfo -F srcversion /lib/modules/$K/updates/cmpunlocker/nvidia.ko 2>/dev/null)
have=$(cat /sys/module/nvidia/srcversion 2>/dev/null)
ncard=$(lspci -nn | grep -ciE '10de:(20c2|2082|20b0)')
init_fail=$(grep -c 'RmInitAdapter failed' $D/dmesg.txt)
booter=$(grep -c 'Booter failed\|failed to execute Booter' $D/dmesg.txt)   # NORMAL in the hundreds on a working unlock (retries)
gsp_fail=$(grep -c 'Cannot initialize GSP firmware RM' $D/dmesg.txt)
smi_cards=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
xid=$(grep -c 'NVRM: Xid' $D/dmesg.txt)
gen2bad=$(grep -ciE 'retrain.*(fail|without Gen2)|FAILED to set' $D/dmesg.txt)
aer=$(grep -ci 'AER' $D/dmesg.txt)
built_for=$(ls -d /lib/modules/*/updates/cmpunlocker 2>/dev/null | cut -d/ -f4 | tr '\n' ' ')

echo "=============== CMP DIAGNOSIS ==============="
echo "running kernel:        $K"
echo "patched driver built:  ${built_for:-NONE}"
echo "nvidia module loaded:  ${have:-NOT LOADED}   (patched: ${want:-none for this kernel})"
echo "CMP cards on PCIe:     $ncard"
echo "cards nvidia-smi sees:  $smi_cards"
echo "RmInitAdapter failed:  $init_fail   GSP init failed: $gsp_fail   Xid: $xid"
echo "(info, normal in small/large numbers on a working unlock: Booter retries $booter, Gen2 retrain notes $gen2bad, AER lines $aer)"
grep -E 'SEC2_DEBUG.*(fbSize|heap_total|WPR2)' $D/dmesg.txt | head -3
nvidia-smi --query-gpu=index,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 | head -5
echo "---------------------------------------------"
if [[ $ncard -gt 0 && $smi_cards -eq $ncard && $init_fail -eq 0 ]]; then
    echo "VERDICT: OK - all $ncard card(s) are up on the driver. Memory per card is shown above."
elif [[ $ncard -eq 0 ]]; then
    echo "VERDICT: the cards are not visible on PCIe at all -> hardware/seating/riser/power problem"
elif [[ -z "$want" ]]; then
    echo "VERDICT: booted kernel $K has no patched driver (built for: ${built_for:-none}) -> reboot and choose plain 'Ubuntu' in GRUB"
elif [[ -z "$have" ]]; then
    echo "VERDICT: the nvidia module is not loaded -> see dmesg.txt in the bundle"
elif [[ "$have" != "$want" ]]; then
    echo "VERDICT: a different (unpatched) nvidia module is loaded -> send the bundle"
elif [[ $init_fail -gt 0 || $gsp_fail -gt 0 ]]; then
    echo "VERDICT: patched driver loaded but cannot start the cards (Booter/RmInitAdapter)."
    echo "         Try: shut down, UNPLUG power 60 s, start, choose 'Ubuntu'. If it repeats -> send the bundle"
else
    echo "VERDICT: no obvious cause found -> send the bundle"
fi
tar czf ~/cmp-diag.tar.gz -C /tmp cmpdiag /var/log/cmp-test.log /opt/cmp-test/out /var/log/nvidia-installer.log 2>/dev/null
U=${SUDO_USER:-root}; H=$(getent passwd "$U" | cut -d: -f6); [[ -n "$H" && "$H" != "$HOME" ]] && cp -f ~/cmp-diag.tar.gz "$H/" && chown "$U" "$H/cmp-diag.tar.gz"
echo "bundle: ${H:-$HOME}/cmp-diag.tar.gz   <- send this file"
