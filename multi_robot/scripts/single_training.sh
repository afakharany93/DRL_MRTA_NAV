#!/bin/bash
# echo "${@:1:2}"
roscore &
rosrun multi_robot run.py "${@:3}" &
sleep 5
roslaunch multi_robot main.launch "${@:1:2}" &
