#!/bin/bash

# Script for removing files from B2.
# Used by the github actions to clean up files used for preview deployments.

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

# Create an empty dir and use `b2 sync` to delete the remote dir.
mkdir tmp
b2 sync --allowEmptySource --delete --noProgress ./tmp b2://$B2_BUCKET/$REMOTE_DIR
rm -r tmp
