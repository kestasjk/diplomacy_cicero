#!/bin/bash


# Activate conda within a script:
eval "$(conda shell.bash hook)"

echo "Activating"
conda activate diplomacy_cicero

pushd $(dirname "$0")

# Use the 2nd GPU as the primary:
export CUDA_VISIBLE_DEVICES="1"

python fairdiplomacy_external/run.py \
	--adhoc -c conf/c07_play_webdip/play_dora_fva.prototxt \
	api_key=$FAIRBOT_APIKEY \
	account_name=FairBot \
	allow_dialogue=false \
	log_dir="`pwd`/logs/" \
	is_backup=false \
	retry_exception_attempts=0 \
	reset_bad_games=1 \
	I.agent=agents/searchbot_neurips21_fva_dora.prototxt

popd
