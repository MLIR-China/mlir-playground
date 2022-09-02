#!/bin/bash

# Script for uploading the exported files to B2.
# Used by the github actions that deploy the site.

set -e

# Command line arguments
B2_BUCKET=${1}
B2_APP_KEY_ID=${2}
B2_APP_KEY=${3}
LOCAL_DIR=${4}
REMOTE_DIR=${5}

cd $LOCAL_DIR

# Install b2
pip install b2

# Authenticate with b2
b2 authorize-account $B2_APP_KEY_ID $B2_APP_KEY

# Upload data files using `b2 sync` to save uploading the same file.
b2 sync --excludeRegex '.*\.wasm' --replaceNewer --noProgress ./ b2://$B2_BUCKET/$REMOTE_DIR

# Upload wasm files individually to set custom contentType.
for wasmfile in *.wasm; do
    b2 upload-file $B2_BUCKET $wasmfile $REMOTE_DIR/$wasmfile --contentType application/wasm
done
