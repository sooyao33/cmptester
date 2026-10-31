# shared helpers for the CMP 170HX seller test kit (sourced, not run)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$KIT/out"
BIN="$KIT/bin"
mkdir -p "$OUT"

say()  { printf '%s\n' "$*"; }
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; FAILS=$((${FAILS:-0}+1)); }
warn() { printf 'WARN  %s\n' "$*"; }
die()  { printf 'ABORT %s\n' "$*" >&2; exit 2; }

need_root() { [[ $EUID -eq 0 ]] || die "run as root (sudo $0)"; }

# never run the destructive stages on our own server
not_qwenian() { [[ "$(hostname)" != "qwenian" || -n "${KIT_ON_QWENIAN:-}" ]] || die "this is qwenian - refusing"; }

cmp_bdfs() { lspci -Dnn 2>/dev/null | grep -iE '10de:(20c2|2082|20b0)' | awk '{print $1}'; }

# nvidia-smi index list (empty if driver not up)
gpu_indices() { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '; }

# expected unlocked MiB by PCI device id: 20c2 = 8 GB card -> 64 GiB, 2082 = 10 GB card -> 40 GiB
expect_mib_for_bus() {
    local bus="${1,,}" id
    id=$(lspci -n -s "${bus: -7}" 2>/dev/null | grep -oE '10de:[0-9a-f]{4}' | cut -d: -f2)   # bb:dd.f
    case "$id" in 20c2) echo 65536 ;; 2082) echo 40960 ;; *) echo 0 ;; esac
}

xid_count() { dmesg 2>/dev/null | grep -c 'NVRM: Xid' || true; }

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
