#!/bin/bash

while true; do
    # Get the current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Run nvidia-smi and parse the output
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while IFS=, read -r GPU_ID NAME GPU_UTIL MEMORY_TOTAL MEMORY_USED MEMORY_FREE; do
        # Format the log message
        LOG_MESSAGE="$TIMESTAMP GPU_ID=$GPU_ID NAME=$NAME GPU_UTIL=$GPU_UTIL% MEMORY_TOTAL=${MEMORY_TOTAL}MiB MEMORY_USED=${MEMORY_USED}MiB MEMORY_FREE=${MEMORY_FREE}MiB"
        echo $LOG_MESSAGE
        echo $LOG_MESSAGE>>"gpu.log"
    done

    sleep 1
done
