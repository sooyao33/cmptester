#!/usr/bin/env bash
# CMP 170HX unlock + stability test - one command, fully automatic.
#
#   sudo ./run.sh [HOURS]     start (default soak 1 h per card, all cards in parallel)
#   sudo ./run.sh status      where it is / live progress
#   sudo ./run.sh log         follow the live log (Ctrl-C only stops watching)
#
# What happens:
#   1. installs build tools, the NVIDIA open driver and the latest cmpunlocker
#   2. POWERS THE MACHINE OFF (a cold boot is required for the unlock).
#      -> press the power button to turn it back on. Everything continues by itself.
#   3. after boot: verifies the unlock, self-tests the error detectors, runs the stability soak
#   4. writes the results bundle cmp-test-results-*.tar.gz into /root and your home directory
# Runs as the systemd service cmp-test, so closing the terminal does not stop it.
set -uo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
INST=/opt/cmp-test
STATE_DIR=/var/lib/cmp-test
LOG=/var/log/cmp-test.log
UNIT=/etc/systemd/system/cmp-test.service

[[ $EUID -eq 0 ]] || { echo "run with sudo:  sudo $0 $*"; exit 1; }
state() { cat "$STATE_DIR/state" 2>/dev/null || echo none; }
setstate() { echo "$1" > "$STATE_DIR/state"; echo "[$(date -u +%FT%TZ)] state -> $1"; }
# follow the live log until the service stops (finished, failed, or the machine powers off)
follow() {
    tail -n 50 -F "$LOG" & local tp=$!
    trap 'kill $tp 2>/dev/null; echo; echo "(stopped watching - the test keeps running. Check later: sudo $INST/run.sh status)"; exit 0' INT
    while systemctl is-active --quiet cmp-test; do sleep 3; done
    sleep 2; kill $tp 2>/dev/null; echo
    case "$(state)" in
        done)   echo "FINISHED. Send back the result file named above." ;;
        failed) echo "STOPPED WITH AN ERROR (see the lines above). Send us the file /var/log/cmp-test.log" ;;
        *)      echo "The test is continuing (state: $(state)). If the machine is about to power off, turn it back on afterwards." ;;
    esac
}
banner() { local m="$*"; echo; echo "########## $m"; wall "cmp-test: $m" 2>/dev/null || true; echo "cmp-test: $m" > /dev/console 2>/dev/null || true; }

case "${1:-start}" in
status)
    echo "state: $(state)   (log: $LOG)"
    [[ "$(state)" == soak ]] && "$INST/04-soak.sh" status
    [[ "$(state)" == done ]] && { cat "$STATE_DIR/result" 2>/dev/null; }
    tail -n 15 "$LOG" 2>/dev/null; exit 0 ;;
log) follow; exit 0 ;;
start|[0-9]*)
    H="${1:-1}"; [[ "$H" == start ]] && H="${2:-1}"
    [[ "$H" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "HOURS must be a number"; exit 1; }
    systemctl is-active --quiet cmp-test && { echo "already running - sudo $0 status"; exit 0; }
    mkdir -p "$STATE_DIR" "$INST"
    [[ "$SRC" != "$INST" ]] && cp -a "$SRC/." "$INST/"
    chmod +x "$INST"/*.sh "$INST"/bin/*
    echo "$H" > "$STATE_DIR/hours"
    echo "${SUDO_USER:-root}" > "$STATE_DIR/user"
    [[ "$(state)" == none || "$(state)" == failed ]] && setstate unlock > /dev/null
    cat > "$UNIT" <<EOF
[Unit]
Description=CMP 170HX unlock + stability test
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=$INST/run.sh --service
StandardOutput=append:$LOG
StandardError=append:$LOG
TimeoutStartSec=infinity
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload && systemctl enable --now cmp-test.service
    echo "started (soak ${H} h per card). Live log below - Ctrl-C only stops watching; it keeps running."
    echo "Later:  sudo $INST/run.sh status"
    sleep 1; follow; exit 0 ;;
--service) ;;   # fall through: the state machine below
*) echo "usage: sudo $0 [HOURS] | status | log"; exit 1 ;;
esac

# ------------------------------------------------------------------ state machine (as the service)
cd "$INST" || exit 2
H=$(cat "$STATE_DIR/hours" 2>/dev/null || echo 1)
echo "[$(date -u +%FT%TZ)] cmp-test service up, state $(state), boot $(cat /proc/sys/kernel/random/boot_id)"
while :; do
case "$(state)" in
unlock)
    banner "stage 1/4: installing driver + cmpunlocker (10-20 min)"
    [[ -f out/recon/os.txt ]] || ./00-recon.sh
    ./01-unlock.sh; rc=$?
    if [[ $rc -eq 10 ]]; then
        banner "a reboot is needed before the unlock can continue - rebooting in 20 s (it continues by itself)"
        sleep 20; systemctl reboot; exit 0
    elif [[ $rc -eq 0 ]]; then
        setstate verify
        banner "unlock installed. POWERING OFF in 60 s for the required cold boot. Turn the machine back ON with the power button - testing continues automatically. If it has not switched off after 3 min, hold the power button 10 s."
        sleep 60; systemctl poweroff; exit 0
    else
        setstate failed; banner "stage 1 FAILED - see $LOG"; systemctl disable cmp-test; exit 1
    fi ;;
verify)
    banner "stage 2/4: verifying the unlock"
    ./02-verify-unlock.sh; echo "verify rc=$?"
    setstate selftest ;;
selftest)
    banner "stage 3/4: self-testing the error detectors (fault injection)"
    ./03-selftest.sh; echo "selftest rc=$?"
    setstate soak ;;
soak)
    if ! ./04-soak.sh finished; then
        ./04-soak.sh running || { banner "stage 4/4: stability soak, ${H} h per card (all cards in parallel)"; ./04-soak.sh start "$H"; }
        while ./04-soak.sh running; do sleep 60; done
        sleep 5
    fi
    setstate collect ;;
collect)
    ./05-collect.sh > out/collect.txt 2>&1; cat out/collect.txt
    T=$(ls -t "$INST"/cmp-test-*.tar.gz 2>/dev/null | head -1)
    U=$(cat "$STATE_DIR/user" 2>/dev/null || echo root); UH=$(getent passwd "$U" | cut -d: -f6)
    for d in /root "$UH"; do [[ -n "$d" && -d "$d" ]] && cp -f "$T" "$d/cmp-test-results-$(hostname).tar.gz" 2>/dev/null && chown "$U" "$d/cmp-test-results-$(hostname).tar.gz" 2>/dev/null; done
    {
      echo "RESULT FILE: ${UH:-/root}/cmp-test-results-$(hostname).tar.gz   (send this file back)"
      for k in PASS FAIL WARN; do echo "  $k: $(grep -c "^$k" out/PROOF.md)"; done
      grep -E '^# GPU Qualification Report' out/soak/gpu*/summary.md 2>/dev/null
    } > "$STATE_DIR/result"
    setstate done
    banner "ALL DONE. Send back: ${UH:-/root}/cmp-test-results-$(hostname).tar.gz"
    cat "$STATE_DIR/result"
    systemctl disable cmp-test; exit 0 ;;
done|failed) systemctl disable cmp-test; exit 0 ;;
*) setstate unlock ;;
esac
done
