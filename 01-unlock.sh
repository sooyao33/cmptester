#!/usr/bin/env bash
# Stage 1 - unattended: build tools + NVIDIA open driver + latest cmpunlocker.
# Exit codes: 0 = installed, needs a COLD power cycle | 10 = needs a reboot, then rerun | 2 = fatal
# Never touches GRUB/kernel cmdline (--no-iommu) or VM passthrough (--no-passthrough).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
need_root; not_qwenian
L="$OUT/unlock"; mkdir -p "$L"

CMPU_URL="https://github.com/amoghmunikote/cmpunlocker.git"
CMPU_DIR=/opt/cmpunlocker
PREFERRED_DRV=610.43.03     # the version validated on our reference cards
K=$(uname -r)

[[ $(cmp_bdfs | wc -l) -gt 0 ]] || die "no CMP 170HX (10de:20c2 / 10de:2082) found on this machine"
say "== $(stamp) stage 1 on $(hostname), kernel $K, $(cmp_bdfs | wc -l) CMP card(s)"
lspci -nn | grep -iE '10de:(20c2|2082|20b0)'

if command -v mokutil >/dev/null && mokutil --sb-state 2>/dev/null | grep -qi 'enabled'; then
    die "Secure Boot is ENABLED. Disable Secure Boot in the BIOS/UEFI setup, then run again."
fi

# --- 1. packages --------------------------------------------------------------------
command -v apt-get >/dev/null || die "this script supports Ubuntu/Debian (apt) only"
export DEBIAN_FRONTEND=noninteractive
PK="build-essential linux-headers-$K pkg-config libglvnd-dev python3 python3-yaml curl git patch kmod pciutils mokutil"
if ! dpkg -s $PK >/dev/null 2>&1; then
    apt-get update || die "apt-get update failed (no internet?)"
    apt-get install -y $PK || die "installing prerequisites failed"
fi
[[ -d /lib/modules/$K/build ]] || die "kernel headers for $K missing (apt install linux-headers-$K)"
pass "prerequisites installed"

# --- 2. latest cmpunlocker (decides which driver versions are supported) ------------------
if [[ -d $CMPU_DIR/.git ]]; then
    git -C "$CMPU_DIR" fetch -q origin || git -C "$CMPU_DIR" -c protocol.version=1 fetch -q origin || die "git fetch failed"
    git -C "$CMPU_DIR" reset -q --hard origin/HEAD
else
    git clone -q "$CMPU_URL" "$CMPU_DIR" || git -c protocol.version=1 clone -q "$CMPU_URL" "$CMPU_DIR" || die "git clone of cmpunlocker failed"
fi
say "cmpunlocker: $(git -C "$CMPU_DIR" log --oneline -1)"
git -C "$CMPU_DIR" log -1 --format=%H > "$L/cmpunlocker-commit.txt"
mapfile -t SUP < <(grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' "$CMPU_DIR/driver/VERSION")
[[ ${#SUP[@]} -gt 0 ]] || die "cmpunlocker lists no supported driver versions"
DRV=$PREFERRED_DRV; [[ " ${SUP[*]} " == *" $DRV "* ]] || DRV=${SUP[0]}
say "supported drivers: ${SUP[*]} -> target $DRV"

# --- 3. nouveau out of the way ------------------------------------------------------------
if lsmod | grep -q '^nouveau'; then
    printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
    update-initramfs -u
    modprobe -r nouveau 2>/dev/null && say "nouveau unloaded" || { say "nouveau is in use: reboot needed"; exit 10; }
fi

# --- 4. NVIDIA driver -----------------------------------------------------------------------
cur=$(grep -oE '[0-9]{3}\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
[[ -z "$cur" ]] && cur=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
is_open=0; grep -q 'Open Kernel Module' /proc/driver/nvidia/version 2>/dev/null && is_open=1
if [[ -n "$cur" && " ${SUP[*]} " == *" $cur "* && $is_open == 1 ]]; then
    DRV=$cur; pass "NVIDIA open driver $cur already installed and supported"
else
    if [[ -n "$cur" ]] || dpkg -l 2>/dev/null | awk '/^ii/{print $2}' | grep -qE '^(nvidia-(driver|kernel|dkms|utils)|libnvidia-compute)'; then
        say "removing the existing NVIDIA driver (${cur:-apt packages}) - unsupported by cmpunlocker"
        [[ -x /usr/bin/nvidia-uninstall ]] && /usr/bin/nvidia-uninstall --silent
        apt-get purge -y '^nvidia-.*' '^libnvidia-.*' >/dev/null 2>&1; apt-get autoremove -y >/dev/null 2>&1
        if lsmod | grep -q '^nvidia'; then
            for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia; do modprobe -r $m 2>/dev/null; done
            lsmod | grep -q '^nvidia' && { say "old NVIDIA module still loaded: reboot needed"; exit 10; }
        fi
    fi
    RUN="NVIDIA-Linux-x86_64-$DRV.run"
    cd "$L" || exit 2
    if [[ ! -f "$RUN" ]]; then
        curl -fL --retry 3 -o "$RUN.part" "https://download.nvidia.com/XFree86/Linux-x86_64/$DRV/$RUN" && mv -f "$RUN.part" "$RUN"
    fi
    [[ -f "$RUN" ]] || die "download of NVIDIA driver $DRV failed"
    sh "./$RUN" --check || die "NVIDIA installer integrity check failed"
    sh "./$RUN" --silent --kernel-module-type=open --no-x-check --no-nouveau-check \
        || die "NVIDIA driver install failed (see /var/log/nvidia-installer.log)"
    pass "installed NVIDIA open driver $DRV"
fi

# Keep the stock module busy (persistence mode) so cmpunlocker cannot hot-swap it: a hot-swapped
# patched module cannot boot cards the stock driver already initialised (needs a cold boot anyway)
# and on our reference box that state hung the following reboot.
modprobe nvidia 2>/dev/null; nvidia-smi -pm 1 >/dev/null 2>&1 || true

# --- 5. unlock --------------------------------------------------------------------------------
( cd "$CMPU_DIR" && ./install.sh --no-passthrough --no-iommu ) || die "cmpunlocker install.sh failed"
modinfo -F srcversion "/lib/modules/$K/updates/cmpunlocker/nvidia.ko" > "$L/patched-srcversion.txt" 2>/dev/null \
    || die "patched nvidia.ko not found after cmpunlocker install"
pass "cmpunlocker installed, patched module srcversion $(cat "$L/patched-srcversion.txt")"
exit 0
