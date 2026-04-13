#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ADDR="${SWITCH_BT_ADDR:-94:58:CB:AE:06:C9}"
HCI_DEV="${SWITCH_BT_DEV:-hci0}"

log() {
    printf '[switch-bt-setup] %s\n' "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "Missing required command: $1"
        exit 1
    fi
}

require_cmd hciconfig

BDADDR_BIN="${SWITCH_BT_BDADDR_BIN:-}"
if [[ -z "${BDADDR_BIN}" ]]; then
    for candidate in "${SCRIPT_DIR}/bdaddr-switch" "${ROOT_DIR}/tools/bdaddr/bdaddr"; do
        if [[ -x "${candidate}" ]]; then
            BDADDR_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${BDADDR_BIN}" ]]; then
    log "Missing bdaddr utility. Checked ${SCRIPT_DIR}/bdaddr-switch and ${ROOT_DIR}/tools/bdaddr/bdaddr"
    exit 1
fi

if command -v rfkill >/dev/null 2>&1; then
    rfkill unblock bluetooth || true
fi

hciconfig "${HCI_DEV}" up || true

current_addr="$("${BDADDR_BIN}" -i "${HCI_DEV}" | awk '/Device address:/ {print $3; exit}')"
if [[ -z "${current_addr}" ]]; then
    log "Could not determine the current Bluetooth address for ${HCI_DEV}"
    exit 1
fi

if [[ "${current_addr}" != "${TARGET_ADDR}" ]]; then
    log "Changing BD_ADDR ${current_addr} -> ${TARGET_ADDR}"
    "${BDADDR_BIN}" -i "${HCI_DEV}" -r "${TARGET_ADDR}"
else
    log "BD_ADDR already set to ${TARGET_ADDR}"
fi

hciconfig "${HCI_DEV}" reset || true
hciconfig "${HCI_DEV}" up

log "Prepared ${HCI_DEV} identity for Switch controller emulation."
