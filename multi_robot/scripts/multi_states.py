#!/usr/bin/env python3
import numpy as np
from collections import deque


class MultiStates():
    def __init__(self, red_state = True, n_robots=2, hz=20, sec_past=0.5):
        self.row = 2 if red_state else 4
        self.n_robots = n_robots
        self.hz= hz
        self.sec_past = sec_past
        self.n_elements = int(self.hz*self.sec_past)
        self.l_current_robot_rel_dist = deque([np.zeros((self.row,self.n_robots)) for i in range(self.n_elements)],
                                            self.n_elements )
        self.l_other_robots_rel_dist = deque([np.zeros((self.n_robots-1,self.row,self.n_robots)) for i in range(self.n_elements)],
                                        self.n_elements )
        self.l_laser = deque([np.zeros((self.row//2,360)) for i in range(self.n_elements)],
                                        self.n_elements )

    def reset(self):
        for i in range(self.n_elements):
            self.l_current_robot_rel_dist.appendleft(np.zeros((self.row,self.n_robots)))
            self.l_other_robots_rel_dist.appendleft(np.zeros((self.n_robots-1,self.row,self.n_robots)))
            self.l_laser.appendleft(np.zeros((self.row//2,360)))

    def insert(self, s):
        self.l_current_robot_rel_dist.appendleft(s[0])
        self.l_other_robots_rel_dist.appendleft(s[1])
        self.l_laser.appendleft(s[2])
        return (np.array(self.l_current_robot_rel_dist), np.array(self.l_other_robots_rel_dist), np.array(self.l_laser), s[3])
