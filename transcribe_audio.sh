#!/bin/bash
# Koe Audio Transcriber for Moltbot/Ronin
# Sends audio file to Koe server for transcription + AI cleanup.
# All processing (ffmpeg, transcription, post-processing, AI cleanup) happens server-side.
#
# Usage: bash transcribe_audio.sh <audio_file>
# Returns: Clean transcription text to stdout
#
# Requires: curl (only dependency - everything else is server-side)
# Server: Koe running on host (host.docker.internal:9876 or WHISPER_SERVER_URL)

set -euo pipefail

AUDIO_FILE="${1:-}"
SERVER="${WHISPER_SERVER_URL:-http://host.docker.internal:9876}"
MAX_RETRIES=2
CONNECT_TIMEOUT=5
REQUEST_TIMEOUT=180

# --- Validation ---

if [ -z "$AUDIO_FILE" ]; then
    echo "Usage: bash transcribe_audio.sh <audio_file>" >&2
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: File not found: $AUDIO_FILE" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl not found in PATH" >&2
    exit 1
fi

# --- Health Check ---

HEALTH=$(curl -s --connect-timeout "$CONNECT_TIMEOUT" --max-time 10 "$SERVER/health" 2>/dev/null) || true
if ! echo "$HEALTH" | grep -q '"ok"' 2>/dev/null; then
    echo "Error: Koe server not reachable at $SERVER" >&2
    echo "Hint: Make sure Koe is running on the desktop and the server is started." >&2
    exit 1
fi

# --- Send file to server (with retry) ---

TMPRESP=$(mktemp)
trap "rm -f $TMPRESP" EXIT

attempt=0
HTTP_CODE="000"
while [ $attempt -lt $MAX_RETRIES ]; do
    HTTP_CODE=$(curl -s -o "$TMPRESP" -w "%{http_code}" \
        --connect-timeout "$CONNECT_TIMEOUT" \
        --max-time "$REQUEST_TIMEOUT" \
        -X POST "$SERVER/transcribe_file" \
        -F "file=@$AUDIO_FILE" \
        -F "vad_filter=true" \
        -F "apply_post_processing=true" \
        -F "apply_ai_cleanup=true" \
        2>/dev/null) || HTTP_CODE="000"

    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -lt $MAX_RETRIES ]; then
        sleep 2
    fi
done

# --- Handle Response ---

if [ "$HTTP_CODE" != "200" ]; then
    if [ "$HTTP_CODE" = "000" ]; then
        echo "Error: Could not connect to Koe server at $SERVER (network error or timeout)" >&2
    elif [ "$HTTP_CODE" = "503" ]; then
        echo "Error: Koe server not ready (engine still loading). Try again in 30 seconds." >&2
    elif [ "$HTTP_CODE" = "413" ]; then
        echo "Error: Audio file too large (max 50MB)" >&2
    else
        echo "Error: Koe server returned HTTP $HTTP_CODE" >&2
        # Show server error detail if available
        cat "$TMPRESP" >&2 2>/dev/null || true
    fi
    exit 1
fi

# Extract text from JSON response - use python3 if available, fall back to grep
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import sys, json
with open('$TMPRESP', 'r') as f:
    r = json.load(f)
text = r.get('text', '')
if text:
    print(text)
else:
    print('Error: Empty transcription', file=sys.stderr)
    sys.exit(1)
"
else
    # Fallback: extract text field with sed (handles simple JSON)
    TEXT=$(sed -n 's/.*"text":"\([^"]*\)".*/\1/p' "$TMPRESP")
    if [ -n "$TEXT" ]; then
        echo "$TEXT"
    else
        echo "Error: Could not parse server response" >&2
        exit 1
    fi
fi
