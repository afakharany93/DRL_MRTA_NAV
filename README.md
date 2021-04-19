# An implementation of a single environment for the paper "End-to-End Deep Reinforcement Learning for Decentralized Task Allocation and Navigation for a Multi-Robot System"

This is an implementation of a single environment for the paper mentioned above.

## Commands

1. Install [ROS Kinetic Kame](http://wiki.ros.org/kinetic/Installation)
2. Install the requirements in the file `requirements.txt`
2. Git clone this repo inside the folder  `/catkin_ws` and compile
3. For training:
  1. To launch the simulator, open a terminal and use the command:
          roslaunch multi_robot main.launch gui_on:=true stage:=1
  2. To start training, Open a second terminal and use the command:
          rosrun multi_robot run.py --HZ=5.0 --batch_size=1000 --batch_size_inc=0.1 --entropy_factor=0.01 --episode_duration=120 --episodes_respawn_goal=50 --epochs=10 --gae --gamma=0.98 --learning_rate=1e-05 --learning_rate_decay=0.95 --learning_rate_decay_steps=10000 --max_batch_size=1000 --mini_batch_size=256 --model_path='None' --run_time=96.0 --seconds_past_for_stacking=1.0 --threads=2 --value_factor=5.0 --vis --red_state
  3. To check the training results, navigate to the experiment folder in the folder `\multi_robot\experiments` in the terminal and use the command:
          rosrun multi_robot results_report.py
4. For testing:
  1. Get the absolute path of the weights of the policy, it's inside the `\checkpoint` inside the experiment folder.
  2. To launch the simulator, open a terminal and use the command:
          roslaunch multi_robot main.launch gui_on:=true stage:=1
  3. To start the test use the following command, replace the weights path in the command
          rosrun multi_robot run.py --HZ=5.0 --batch_size=1000 --batch_size_inc=0.1 --entropy_factor=0.01 --episode_duration=120 --episodes_respawn_goal=50 --epochs=10 --gae --gamma=0.98 --learning_rate=1e-05 --learning_rate_decay=0.95 --learning_rate_decay_steps=10000 --max_batch_size=1000 --mini_batch_size=256 --model_path='None' --run_time=2.0 --seconds_past_for_stacking=1.0 --threads=2 --value_factor=5.0 --vis --red_state --test --model_weights='weights path here'

## Citation
@Article{app11072895, <br>
AUTHOR = {Elfakharany, Ahmed and Ismail, Zool Hilmi}, <br>
TITLE = {End-to-End Deep Reinforcement Learning for Decentralized Task Allocation and Navigation for a Multi-Robot System }, <br>
JOURNAL = {Applied Sciences}, <br>
VOLUME = {11}, <br>
YEAR = {2021}, <br>
NUMBER = {7}, <br>
ARTICLE-NUMBER = {2895}, <br>
URL = {https://www.mdpi.com/2076-3417/11/7/2895}, <br>
ISSN = {2076-3417}, <br>
ABSTRACT = {In this paper, we present a novel deep reinforcement learning (DRL) based method that is used to perform multi-robot task allocation (MRTA) and navigation in an end-to-end fashion. The policy operates in a decentralized manner mapping raw sensor measurements to the robot’s steering commands without the need to construct a map of the environment. We also present a new metric called the Task Allocation Index (TAI), which measures the performance of a method that performs MRTA and navigation from end-to-end in performing MRTA. The policy was trained on a simulated gazebo environment. The centralized learning and decentralized execution paradigm was used for training the policy. The policy was evaluated quantitatively and visually. The simulation results showed the effectiveness of the proposed method deployed on multiple Turtlebot3 robots.}, <br>
DOI = {10.3390/app11072895} <br>
}
