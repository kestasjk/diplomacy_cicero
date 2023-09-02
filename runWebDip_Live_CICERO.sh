#!/bin/bash

pushd $(dirname "$0")

# Activate conda within a script:
eval "$(conda shell.bash hook)"

echo "Activating"
conda activate diplomacy_cicero

# Use the 2nd GPU as the primary:
export CUDA_VISIBLE_DEVICES="1"
export FIRST_GPU="1"
export SECOND_GPU="0"

python fairdiplomacy_external/run.py --adhoc \
    -c conf/c07_play_webdip/play.prototxt \
    api_key=$DIPGPT_APIKEY account_name='dipgpt' \
    allow_dialogue=true \
    log_dir="`pwd`/logs/" \
    is_backup=false \
    retry_exception_attempts=0 \
    reset_bad_games=1 \
    require_message_approval=False \
    only_bump_msg_reviews_for_same_power=True

popd

