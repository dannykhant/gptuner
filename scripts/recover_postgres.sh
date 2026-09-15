#!/bin/bash
CONTAINER="${PG_CONTAINER_NAME:-adcoexp-db}"
docker exec "${CONTAINER}" bash -c 'rm -f /var/lib/postgresql/data/postgresql.auto.conf' 2>/dev/null || true
docker restart "${CONTAINER}" 2>/dev/null || true
sleep 3