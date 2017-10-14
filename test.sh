#!/bin/bash
python3 weighted_flow_main.py --mode test --model_name my_model  --data_path /home/fei/Data/fei/flow/FlyingChairs_release/data_png/ --filenames_file flyingchairs_test.txt --log_directory ./log/ --checkpoint_path ./log/my_model/model-40000   
