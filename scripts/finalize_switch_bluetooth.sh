#!/usr/bin/env bash
set -euo pipefail

TARGET_NAME="${SWITCH_BT_NAME:-Pro Controller}"
TARGET_CLASS="${SWITCH_BT_CLASS:-0x002508}"
HCI_DEV="${SWITCH_BT_DEV:-hci0}"

log() {
    printf '[switch-bt-finalize] %s\n' "$1"
}

if ! command -v hciconfig >/dev/null 2>&1; then
    log "Missing required command: hciconfig"
    exit 1
fi

hciconfig "${HCI_DEV}" up
hciconfig "${HCI_DEV}" class "${TARGET_CLASS}"
hciconfig "${HCI_DEV}" name "${TARGET_NAME}"

log "Applied ${TARGET_CLASS} and ${TARGET_NAME} to ${HCI_DEV}."
