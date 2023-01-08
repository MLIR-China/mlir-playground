#!/bin/bash

# Script for publishing to Cloudflare pages.
# Used by the github actions that deploy the site.

set -e

# Command line arguments
CLOUDFLARE_API_TOKEN=${1}
CLOUDFLARE_ACCOUNT_ID=${2}
BRANCH_NAME=${3}
OUTPUT_FILE=${4}

# Install wrangler
npm i -g wrangler@2.2.1

# Publish to CF Pages
export API_CREDENTIALS="API Token"
export CLOUDFLARE_API_TOKEN=$CLOUDFLARE_API_TOKEN
export CLOUDFLARE_ACCOUNT_ID=$CLOUDFLARE_ACCOUNT_ID
OUTPUT_URL=$(wrangler pages publish out --project-name=mlir-playground-direct --branch=$BRANCH_NAME | grep -o "https://.*pages\.dev")

# Output
echo "Published to ${OUTPUT_URL}"
echo "PUBLISHED_URL=${OUTPUT_URL}" >> $OUTPUT_FILE
