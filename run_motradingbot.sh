#!/bin/bash

# Navigate to the directory where your script is located
cd /Library/Python/3.9/site-packages/motradingbot/

# Infinite loop to keep restarting the script in case of failure
while true; do
    # Run the Python script
    python3 motradingbot.py

    # If the script exits, wait for a few seconds before restarting
    echo "Script crashed or terminated. Restarting in 10 seconds..."
    sleep 10
done
