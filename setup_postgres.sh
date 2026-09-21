#!/usr/bin/env bash
#
# setup_postgres.sh
# ─────────────────────────────────────────────────────────────────
# Installs PostgreSQL natively (no Docker) and relocates its data
# directory onto the RunPod network volume at /workspace, so the
# database survives pod restarts/terminations.
#
# Safe to re-run: if the data directory has already been moved to
# /workspace, the script just makes sure the cluster is started and
# exits.
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
VOLUME_DATA_DIR="/workspace/pg-data"

if [ ! -d /workspace ]; then
    echo "!! /workspace not found — this script expects a RunPod network volume"
    echo "   mounted at /workspace. Aborting."
    exit 1
fi

# ── Install PostgreSQL if it isn't already ────────────────────────
if ! command -v psql >/dev/null 2>&1; then
    echo ">> Installing PostgreSQL..."
    apt-get update -y
    apt-get install -y postgresql
fi

# ── Discover the installed cluster (version/name), e.g. 16 main ──
PG_VERSION="$(ls /etc/postgresql | sort -V | tail -n1)"
PG_CLUSTER="main"
NATIVE_DATA_DIR="/var/lib/postgresql/${PG_VERSION}/${PG_CLUSTER}"

echo ">> Detected PostgreSQL ${PG_VERSION}, cluster '${PG_CLUSTER}'"

# ── Relocate the data directory to the network volume (once) ─────
if [ -L "$NATIVE_DATA_DIR" ] && [ "$(readlink -f "$NATIVE_DATA_DIR")" = "$VOLUME_DATA_DIR" ]; then
    echo ">> Data directory already relocated to ${VOLUME_DATA_DIR}"
elif [ -d "$VOLUME_DATA_DIR" ]; then
    echo ">> ${VOLUME_DATA_DIR} already exists on the volume — relinking (no data copy)"
    pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" stop || true
    rm -rf "$NATIVE_DATA_DIR"
    ln -s "$VOLUME_DATA_DIR" "$NATIVE_DATA_DIR"
else
    echo ">> Moving PostgreSQL data directory to ${VOLUME_DATA_DIR}..."
    pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" stop || true
    mv "$NATIVE_DATA_DIR" "$VOLUME_DATA_DIR"
    ln -s "$VOLUME_DATA_DIR" "$NATIVE_DATA_DIR"
fi

chown -R postgres:postgres "$VOLUME_DATA_DIR"

# ── Start the cluster ──────────────────────────────────────────────
echo ">> Starting PostgreSQL..."
service postgresql start || pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" start

# ── Create role + database on first run only ──────────────────────
ROLE_EXISTS="$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${PG_USER}'")"
if [ "$ROLE_EXISTS" != "1" ]; then
    echo ">> Creating role '${PG_USER}' and database '${PG_DB}'..."
    sudo -u postgres psql -c "CREATE ROLE ${PG_USER} LOGIN PASSWORD '${PG_PASSWORD}';"
    sudo -u postgres createdb -O "${PG_USER}" "${PG_DB}"
else
    echo ">> Role '${PG_USER}' already exists — skipping creation"
fi

echo ""
echo "✅ PostgreSQL is running with data persisted at ${VOLUME_DATA_DIR}"
echo "   DATABASE_URL=postgresql://${PG_USER}:${PG_PASSWORD}@localhost:5432/${PG_DB}"
echo ""
echo "Add that DATABASE_URL to your .env file (see .env.example)."
