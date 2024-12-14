#!/bin/bash
# This script is used to process the amass data
python uhc/data_process/process_amass_raw.py

python uhc/data_process/process_amass_db.py

python uhc/data_process/convert_amass_isaac.py