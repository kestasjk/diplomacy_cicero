#!/bin/bash

pushd $(dirname "$0")

# Activate conda within a script:
eval "$(conda shell.bash hook)"

echo "Activating"
conda activate diplomacy_cicero

python fairdiplomacy_external/run.py --adhoc \
    -c conf/c07_play_webdip/play.prototxt \
    api_key=$DIPGPT_APIKEY account_name='dipgpt' \
    allow_dialogue=true \
    log_dir=logs/ \
    is_backup=false \
    retry_exception_attempts=0 \
    reset_bad_games=1 \
    I.agent=agents/cicero.prototxt \
    require_message_approval=False \
    only_bump_msg_reviews_for_same_power=True

popd

