#!/bin/bash

python -u app_v3.py &
python -u video_worker.py &

wait