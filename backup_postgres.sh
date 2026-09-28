#!/usr/bin/env bash
#
# backup_postgres.sh
# ─────────────────────────────────────────────────────────────────
# Takes an immediate snapshot of the 'asp' database to the network
# volume, so it survives a pod stop/restart. setup_postgres.sh also
# runs this automatically in the background on a timer — use this
# script when you want an up-to-the-second snapshot right before
# stopping the pod deliberately.
#
set -euo pipefail

PG_DB="asp"
BACKUP_DIR="/workspace/pg-backups"
BACKUP_FILE="${BACKUP_DIR}/latest.dump"

mkdir -p "$BACKUP_DIR"

# Root on RunPod usually has no `sudo`; see setup_postgres.sh.
if [ "$(id -u)" -eq 0 ]; then
    PGRUN="runuser -u postgres --"
else
    PGRUN="sudo -u postgres"
fi
${PGRUN} pg_dump -Fc "${PG_DB}" -f "${BACKUP_FILE}.tmp"
mv "${BACKUP_FILE}.tmp" "$BACKUP_FILE"

echo "✅ Backup written to ${BACKUP_FILE}"
