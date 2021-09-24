#!/bin/bash

bash $(dirname "$0")/setup.sh || exit 1

help_string="-v - provide list of video_stream provider\n-c - provide path to connector chart config. You can find the usb outlets of the webcams by running $v4l2-ctl --list-devices\n
By default the script will publish the /dev/video0 device stream at /camera0/image_raw topic"
video_stream_provider_string=''
camera_connector_chart=''
python_node_string='python_node:=false'
camera_string='realsense:=false'

while getopts ":hv:c:" opt; do
  case $opt in
    h ) printf "Help: \n${help_string}\n"
        exit 0;;
    c ) if [[ -n $video_stream_provider ]];then
        echo "You cannot provide both -v and -c arguments"
        exit 1
        fi
        camera_connector_chart=$OPTARG;;
    v ) if [[ -n $camera_connector_chart ]];then
        echo "You cannot provide both -v and -c arguments"
        exit 1
        fi
        video_stream_provider=$OPTARG;;
    ? ) printf "Usage: cmd [-v] [-c]\n${help_string}";exit 1;;
  esac
done

if [ -n "$video_stream_provider" ];then
  video_stream_provider_string="video_stream_provider:=${video_stream_provider}"
fi


roslaunch robonetv2 launch.launch ${video_stream_provider_string} camera_connector_chart:=${camera_connector_chart} ${python_node_string} ${camera_string}

if [[ ! -e ~/interbotix_ws/src/robonetv2/widowx_envs/global_config.yml ]]; then
    touch ~/interbotix_ws/src/robonetv2/widowx_envs/global_config.yml
fi