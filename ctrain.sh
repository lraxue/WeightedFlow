#!/bin/bash
python3 weighted_flow_main.py --mode train --model_name my_model --data_path /home/fei/Data/fei/flow/FlyingChairs_release/data_png/ --filenames_file flyingchairs_train.txt --log_directory ./log/ --checkpoint_path ./log/my_model/model-210000  
