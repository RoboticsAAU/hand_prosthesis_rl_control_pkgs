#!/bin/bash

while true; do
    echo "Starting ROS launch..."
    roslaunch rl_env mia_hand_rl_env.launch &> /tmp/roslaunch_output &
    pid=$!  # Get the process ID of the roslaunch command
    tail --pid=$pid -f /tmp/roslaunch_output | while read line; do
        # Check if the output contains the specific error messages
        echo "$line" | grep -qE "free\(\): corrupted unsorted chunks|malloc\(\): unsorted double linked list corrupted|Aborted \(core dumped\)"
        if [ $? -eq 0 ]; then
            echo "Error detected. Terminating ROS launch..."
            kill $pid  # Terminate the roslaunch command
        fi
    done
    echo "ROS launch exited with code $?. Restarting..."
    sleep 5  # You may want to adjust the sleep time between restarts
done
