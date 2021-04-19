#!/usr/bin/env python3
from collections import deque

class TrainQ():
    def __init__(self, max_batch_size):
        self.q = deque([deque([], max_batch_size*10) for i in range(12)])

    def empty_q(self):
        for i in range(len(self.q)):
            self.q[i].clear()

    def train_push(self, s, a, r, s_, v, done):
        # with self.lock_queue:
        s1 = s[0]
        s2 = s[1]
        s3 = s[2]
        s4 = s[3]
        s1_ = s_[0]
        s2_ = s_[1]
        s3_ = s_[2]
        s4_ = s_[3]
        self.q[0].append(s1)
        self.q[1].append(s2)
        self.q[2].append(s3)
        self.q[3].append(s4)
        self.q[4].append(a)
        self.q[5].append(r)
        self.q[6].append(s1_)
        self.q[7].append(s2_)
        self.q[8].append(s3_)
        self.q[9].append(s4_)
        self.q[10].append(v)
        self.q[11].append(done)

    def not_empty(self):
        return len(self.q[-1]) > 0

    def has_more_elments_than(self, n=0):
        return len(self.q[-1]) > n

    def make_last_done_true(self):
        self.q[-1][-1] = True

    def push_to_other_list(self, func, name):
        if self.not_empty() :
            self.make_last_done_true()
            func(self.q, name)
            self.empty_q()




#
# def empty_q(self):
#     self.q = [ [], [], [], [], [], [], [], [], [], [], [], [] ]
#     rospy.loginfo('emptied train queue for {}'.format(self.env.robot_name))
