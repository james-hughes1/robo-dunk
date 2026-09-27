#!/bin/bash
# Build and deploy the sandbox to Google Cloud Run.
# Requires: gcloud CLI, logged in (`gcloud auth login`) with a project set.
# The trained model(s) in models/ are uploaded with the source (see .gcloudignore).
set -euo pipefail

SERVICE="${SERVICE:-robo-dunk}"
REGION="${REGION:-europe-west2}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if ! ls models/*.zip > /dev/null 2>&1; then
  echo "No model found in models/ - the app needs at least one .zip model." >&2
  exit 1
fi

echo "Deploying $SERVICE to $PROJECT ($REGION)..."

# Scales to zero when idle; capped at one instance to bound cost.
# Session affinity keeps Gradio's queue requests on the same instance.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 300 \
  --session-affinity \
  --cpu-boost

# Source deploys push a new image every time; keep only the latest two
# so Artifact Registry storage stays small.
POLICY=$(mktemp)
cat > "$POLICY" <<'JSON'
[
  {"name": "keep-latest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-rest", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --project "$PROJECT" \
  --location "$REGION" \
  --policy "$POLICY" \
  --no-dry-run
rm -f "$POLICY"

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
