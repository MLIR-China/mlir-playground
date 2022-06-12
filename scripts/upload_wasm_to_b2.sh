#!/bin/bash

# Script for uploading the exported wasm files to B2.
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

# Upload data files in public/wasm
for datafile in *.data; do
    b2 upload-file $B2_BUCKET $datafile $REMOTE_DIR/$datafile
done

# Upload wasm files in public/wasm
for wasmfile in *.wasm; do
    b2 upload-file $B2_BUCKET $wasmfile $REMOTE_DIR/$wasmfile --contentType application/wasm
done
