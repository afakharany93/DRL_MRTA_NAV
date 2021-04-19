#!/usr/bin/env python3
from collections import deque
import threading
# import numpy as np
# brain_q

class BrainTrainQ():
    def __init__(self, threads=4):
        self.train_queue = deque([deque([]) for i in range(12)])    # s1, s2, s3, a, r, s1', s2', s3', s', v, terminal mask
        self.names_queue = []
        self.lock_queue = threading.Lock()
        self.threads = threads

    def set_threads(self, threads):
        with self.lock_queue:
            self.threads = threads

    def check_memory_size(self, min_batch=10):
        if len(self.train_queue[0]) < min_batch or len(self.names_queue) < self.threads:
            return True
        return False

    def get_train_data(self):
        return self.train_queue

    def empty_train_queue(self):
        self.names_queue = []
        # self.train_queue = [ [], [], [], [], [], [], [], [], [], [], [], [] ] # s1, s2, s3, a, r, s1', s2', s3', s', v, terminal mask
        for i in range(len(self.train_queue)):
            self.train_queue[i].clear()

    def train_push(self, s, a, r, s_, v,done):
        with self.lock_queue:
            s1 = s[0]
            s2 = s[1]
            s3 = s[2]
            s4 = s[3]
            s1_ = s_[0]
            s2_ = s_[1]
            s3_ = s_[2]
            s4_ = s_[3]
            self.train_queue[0].append(s1)
            self.train_queue[1].append(s2)
            self.train_queue[2].append(s3)
            self.train_queue[3].append(s4)
            self.train_queue[4].append(a)
            self.train_queue[5].append(r)
            self.train_queue[6].append(s1_)
            self.train_queue[7].append(s2_)
            self.train_queue[8].append(s3_)
            self.train_queue[9].append(s4_)
            self.train_queue[10].append(v)
            self.train_queue[11].append(done)

    def train_push_list(self, queue_data, name):
        with self.lock_queue:
            s1, s2, s3, s4, a, r, s1_, s2_, s3_, s4_, v, done = queue_data
            # print(len(s2))
            # print(type(s2[0]))
            # print(s2[0].shape)
            self.train_queue[0].extend(s1)
            self.train_queue[1].extend(s2)
            self.train_queue[2].extend(s3)
            self.train_queue[3].extend(s4)
            self.train_queue[4].extend(a)
            self.train_queue[5].extend(r)
            self.train_queue[6].extend(s1_)
            self.train_queue[7].extend(s2_)
            self.train_queue[8].extend(s3_)
            self.train_queue[8].extend(s4_)
            self.train_queue[10].extend(v)
            self.train_queue[11].extend(done)
            if name not in self.names_queue and name != 'her':
                self.names_queue.append(name)
            # print(np.array(s2).shape)
