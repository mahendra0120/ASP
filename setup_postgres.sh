#!/usr/bin/env bash
#
# setup_postgres.sh
# ─────────────────────────────────────────────────────────────────
# Installs PostgreSQL natively (no Docker) on the pod's LOCAL disk
# (required — Postgres refuses to run with a root-owned data
# directory, and RunPod network volumes generally don't support
# chown to a non-root owner, so the live data directory can't live
# on /workspace).
#
# Persistence across pod restarts is instead handled via backup /
# restore against the network volume:
#   - On every start, if a prior dump exists at
#     /workspace/pg-backups/latest.dump and this is a freshly
#     initialized cluster, it's restored before anything else runs.
#   - A background loop dumps the database to that same path every
#     BACKUP_INTERVAL_SECONDS, so the network volume always has a
#     recent snapshot even if the pod is killed ungracefully.
#
# Credentials here match .env.example — update both together if you
# change them.
#
# Usage:
#   bash setup_postgres.sh
#
set -euo pipefail

PG_USER="asp_user"
PG_PASSWORD="asp_password"
PG_DB="asp"
BACKUP_DIR="/workspace/pg-backups"
BACKUP_FILE="${BACKUP_DIR}/latest.dump"
BACKUP_INTERVAL_SECONDS="${BACKUP_INTERVAL_SECONDS:-300}"   # 5 min default

if [ ! -d /workspace ]; then
    echo "!! /workspace not found — this script expects a RunPod network volume"
    echo "   mounted at /workspace. Aborting."
    exit 1
fi
mkdir -p "$BACKUP_DIR"

# ── Install PostgreSQL if it isn't already ────────────────────────
if ! command -v psql >/dev/null 2>&1; then
    echo ">> Installing PostgreSQL..."
    apt-get update -y
    apt-get install -y postgresql
fi

PG_VERSION="$(ls /etc/postgresql | sort -V | tail -n1)"
PG_CLUSTER="main"
echo ">> Detected PostgreSQL ${PG_VERSION}, cluster '${PG_CLUSTER}'"

# ── Start PostgreSQL on its normal LOCAL data directory ───────────
echo ">> Starting PostgreSQL..."
service postgresql start || pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" start

echo ">> Waiting for PostgreSQL to accept connections..."
for i in $(seq 1 30); do
    if sudo -u postgres pg_isready -q; then break; fi
    sleep 1
done

# ── Create role + database on first run only ──────────────────────
FRESH_CLUSTER=false
ROLE_EXISTS="$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${PG_USER}'")"
if [ "$ROLE_EXISTS" != "1" ]; then
    echo ">> Creating role '${PG_USER}' and database '${PG_DB}'..."
    sudo -u postgres psql -c "CREATE ROLE ${PG_USER} LOGIN PASSWORD '${PG_PASSWORD}';"
    sudo -u postgres createdb -O "${PG_USER}" "${PG_DB}"
    FRESH_CLUSTER=true
else
    echo ">> Role '${PG_USER}' already exists — skipping creation"
fi

# ── Restore the latest backup into a fresh cluster, if one exists ─
if [ "$FRESH_CLUSTER" = true ] && [ -f "$BACKUP_FILE" ]; then
    echo ">> Found existing backup at ${BACKUP_FILE} — restoring..."
    sudo -u postgres pg_restore -d "${PG_DB}" --clean --if-exists "$BACKUP_FILE" \
        && echo ">> Restore complete." \
        || echo "!! Restore reported errors — check output above."
else
    echo ">> No restore needed (either not a fresh cluster, or no backup found yet)."
fi

# ── Background auto-backup loop: dump to the network volume ──────
# Writes to a temp file and renames atomically so a killed pod never
# leaves a half-written dump behind.
if ! pgrep -f "pg_dump.*${PG_DB}.*autobackup-loop" >/dev/null 2>&1; then
    echo ">> Starting background auto-backup loop (every ${BACKUP_INTERVAL_SECONDS}s)..."
    nohup bash -c "
        # marker string 'autobackup-loop' below is just so pgrep can find this loop
        while true; do
            sleep ${BACKUP_INTERVAL_SECONDS}
            sudo -u postgres pg_dump -Fc '${PG_DB}' -f '${BACKUP_FILE}.tmp' \
                && mv '${BACKUP_FILE}.tmp' '${BACKUP_FILE}' # autobackup-loop
        done
    " > "${BACKUP_DIR}/autobackup.log" 2>&1 &
    disown
fi

echo ""
echo "✅ PostgreSQL is running (local data dir) with backups persisted at ${BACKUP_FILE}"
echo "   DATABASE_URL=postgresql://${PG_USER}:${PG_PASSWORD}@localhost:5432/${PG_DB}"
echo ""
echo "Add that DATABASE_URL to your .env file (see .env.example)."
echo "To take a manual backup right now, run: bash backup_postgres.sh"
