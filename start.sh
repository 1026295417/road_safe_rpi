#!/bin/bash
##source /home/pi/envs/ROAD/bin/activate

source /opt/intel/openvino/bin/setupvars.sh
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
cd /home/pi/works/road_safe_rpi/rpi
/home/pi/envs/ROAD/bin/python main.py
