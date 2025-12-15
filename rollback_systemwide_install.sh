#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="/opt/rocm"
PROFILE_FILE="/etc/profile.d/rocm-therock.sh"
LDCONF_FILE="/etc/ld.so.conf.d/rocm-therock.conf"

if [ "$EUID" -ne 0 ]; then
  echo "ERROR: rollback_systemwide_install.sh requires root privileges."
  echo "Run it with sudo once you confirm the live install needs cleanup."
  exit 1
fi

echo "=========================================="
echo "TheRock System-wide Rollback Helper"
echo "=========================================="
echo ""

if [ -d "$INSTALL_DIR" ]; then
  echo "[1/4] Removing /opt/rocm"
  rm -rf "$INSTALL_DIR"
  echo "✓ Removed $INSTALL_DIR"
else
  echo "[1/4] No /opt/rocm directory found (nothing to remove)"
fi

BACKUP="$(ls -dt /opt/rocm.backup.* 2>/dev/null | head -n 1 || true)"
if [ -n "$BACKUP" ] && [ -d "$BACKUP" ]; then
  echo "[2/4] Restoring previous backup: $BACKUP"
  mv "$BACKUP" "$INSTALL_DIR"
  echo "✓ Restored $BACKUP → $INSTALL_DIR"
else
  echo "[2/4] No backup snapshot found; nothing to restore"
fi

if [ -f "$PROFILE_FILE" ]; then
  echo "[3/4] Removing TheRock profile ($PROFILE_FILE)"
  rm -f "$PROFILE_FILE"
  echo "✓ Profile cleaned"
else
  echo "[3/4] Profile file already absent; skipping"
fi

if [ -f "$LDCONF_FILE" ]; then
  echo "[4/4] Removing ldconfig entry ($LDCONF_FILE)"
  rm -f "$LDCONF_FILE"
  ldconfig
  echo "✓ ld.so cache refreshed"
else
  echo "[4/4] ldconfig entry not present; nothing to do"
fi

echo ""
echo "Rollback complete."
echo "If you reinstall later, rerun install_systemwide.sh and log in/out afterwards."
