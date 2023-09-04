#!/bin/bash

# Remove the output file if it exists
rm -f memory_usage.txt

# Function to monitor memory usage
monitor_memory() {
  while [[ -e "/proc/$1" ]]; do
    echo "Timestamp: $(date)" >> memory_usage.txt
    free -h >> memory_usage.txt
    echo " " >> memory_usage.txt
    sleep 1
  done
}

# Start memory monitoring in the background
monitor_memory $$ &
MONITOR_PID=$!

# Run the DeepSpeed command
deepspeed train.py --gradient_checkpointing 1

# Kill the background memory monitoring process
kill $MONITOR_PID
