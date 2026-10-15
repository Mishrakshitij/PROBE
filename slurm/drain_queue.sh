#!/bin/bash
# Submit queued sbatch commands as SLURM QOS slots free up.
#
# The cscc QOS caps this user at 4 submitted jobs, so the rebuttal experiment set
# cannot all be submitted at once. This reads one shell command per line from a
# queue file and submits the next whenever the user is below MAX_QUEUED, then
# removes it from the file. Blank lines and #-comments are skipped.
#
# Usage: bash slurm/drain_queue.sh <queue-file> [max_queued] [poll_seconds]
set -uo pipefail

QUEUE="${1:?Usage: drain_queue.sh <queue-file> [max_queued] [poll_seconds]}"
MAX_QUEUED="${2:-4}"
POLL="${3:-120}"
DEADLINE=$(( $(date +%s) + 86400 ))   # give up after 24h rather than loop forever

cd /home/salem.lahlou/kmishra/aapo

while [ -s "$QUEUE" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "drain: 24h deadline reached, stopping"; break; }

    n=$(squeue -u "$USER" -h | wc -l)
    if [ "$n" -lt "$MAX_QUEUED" ]; then
        # First non-blank, non-comment line.
        cmd=$(grep -vE '^\s*(#|$)' "$QUEUE" | head -1)
        if [ -z "$cmd" ]; then echo "drain: queue holds only comments, done"; break; fi
        echo "drain: [$n/$MAX_QUEUED in queue] submitting: ${cmd:0:100}..."
        if eval "$cmd"; then
            # Drop just that line, keeping the rest of the file intact.
            grep -vFx "$cmd" "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
        else
            echo "drain: submission FAILED, retrying after $POLL s"
        fi
    fi
    sleep "$POLL"
done
echo "drain: queue empty, exiting $(date)"
