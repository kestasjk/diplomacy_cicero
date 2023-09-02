#!/bin/bash

pushd $(dirname "$0")

# Activate conda within a script:
eval "$(conda shell.bash hook)"

echo "Activating"
conda activate diplomacy_cicero

python fairdiplomacy_external/run.py \
	--adhoc -c conf/c07_play_webdip/play.prototxt \
	api_key=$FAIRBOT_APIKEY \
	account_name=FairBot2 \
	allow_dialogue=false \
	log_dir=/home/kestasjk/fair/logs/ \
	is_backup=false \
	retry_exception_attempts=1 \
	reset_bad_games=1 \
	I.agent=agents/searchbot_neurips21_fva_dora.prototxt

popd
