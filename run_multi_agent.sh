#!/bin/bash

echo "Starting multi-agent training session..."
echo "Shared Q-table: qtable.pkl"
echo ""

# Read numAgents from server config
NUM_AGENTS=$(grep -o '"numAgents":\s*[0-9]*' ../openfront_io/bot-server.config.json | grep -o '[0-9]*')
if [ -z "$NUM_AGENTS" ]; then
    echo "Warning: numAgents not found in config, defaulting to 2"
    NUM_AGENTS=2
fi

echo "Starting $NUM_AGENTS agents..."

PIDS=()
for i in $(seq 1 $NUM_AGENTS); do
    AGENT_ID=$(printf "agent%03d" $i)
    echo "Starting agent $i with ID: $AGENT_ID"
    AGENT_CLIENT_ID=$AGENT_ID uv run main.py &
    PIDS+=($!)
    sleep 1
done

echo "All $NUM_AGENTS agents started: ${PIDS[*]}"
echo "Sharing qtable.pkl for accelerated learning"
echo "Press Ctrl+C to stop all agents"

trap "kill ${PIDS[*]} 2>/dev/null; exit" INT TERM

wait
