#!/bin/bash

env | grep -q "WEBDIP_URL"

if [ $? -eq 0 ]; then
	echo "Found WEBDIP_URL"
else
	echo "WEBDIP_URL not set, running .profile"
	ls ~
	. ~/.profile
fi

echo "Loading environment"

eval "$(conda shell.bash hook)"

echo "Activating"
conda activate diplomacy_cicero

#cd ~kestasjk/fair/

# Function to check if the Python process is running
is_python_process_running() {
    kill -0 $PYTHON_PID 2>/dev/null
}

# Function to cleanup and terminate the Python process
cleanup() {
    if is_python_process_running; then
        echo "Terminating the Python process..."
        kill $PYTHON_PID
    fi
    exit
}
# Trap interrupt signals (e.g., Ctrl+C) to ensure cleanup
trap cleanup INT
trap cleanup TERM

echo "Starting script"
python fairdiplomacy_external/run.py --adhoc -c conf/c07_play_webdip/cicero_live.prototxt api_key=$DIPGPT_APIKEY account_name='dipgpt' allow_dialogue=true log_dir=/home/kestasjk/fair/logs/ is_backup=false retry_exception_attempts=1 reset_bad_games=1 I.agent=agents/cicero.prototxt require_message_approval=False only_bump_msg_reviews_for_same_power=True
PYTHON_PID=$!

echo "Deactivating"
conda deactivate

