#!/bin/bash
set -e
# Usage: ./scripts/upload_env.sh <ec2-host> [ssh-key-path] [remote-path]

EC2_HOST="$1"
SSH_KEY="${2:-~/.ssh/id_rsa}"
REMOTE_PATH="${3:-~/polymarket-mm/.env}"
LOCAL_ENV="$(dirname "$0")/../.env"

if [ -z "$EC2_HOST" ]; then
    echo "Usage: $0 <ec2-host> [ssh-key-path] [remote-path]"
    echo "  ec2-host: SSH destination (e.g., ubuntu@ec2-1-2-3-4.compute.amazonaws.com)"
    echo "  ssh-key-path: Path to SSH key (default: ~/.ssh/id_rsa)"
    echo "  remote-path: Remote .env path (default: ~/polymarket-mm/.env)"
    exit 1
fi

if [ ! -f "$LOCAL_ENV" ]; then
    echo "Error: Local .env file not found at $LOCAL_ENV"
    exit 1
fi

# Check if remote .env exists
if ssh -i "$SSH_KEY" "$EC2_HOST" "test -f \"$REMOTE_PATH\""; then
    echo "Config already exists on server. Skipping upload."
    exit 0
fi

# Upload .env file
scp -i "$SSH_KEY" "$LOCAL_ENV" "$EC2_HOST:$REMOTE_PATH" || { echo "Upload failed"; exit 1; }
echo "Successfully uploaded .env to $EC2_HOST:$REMOTE_PATH"
