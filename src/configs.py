# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:51
# @Author  : LIU YI

class Configs(object):
    def __init__(self):
        self.data = 'cifar'
        self.user_num = 5
        self.gpu = 0
        self.rounds = 10
        self.local_ep = 1
        self.iid = 0
        self.unequal = 1
        self.frac = 1
        self.lr = 0.01
        self.model = 'cnn'
        if self.data == 'cifar':
            self.batch_size = 50
        else:
            self.batch_size = 10

        self.chaos = True

        import numpy as np
        np.random.seed(0)
        import pandas as pd
        if self.chaos:
            self.client_will = np.random.rand(self.user_num*self.rounds).round().reshape(self.rounds, self.user_num)
            pd.DataFrame(self.client_will).to_csv('100client_will.csv', columns=None, index=None, header=None)

        else:
            self.client_will = np.ones(self.user_num*self.rounds).reshape(self.rounds, self.user_num)

        self.neighbors = {0: [1, 4], 1: [0, 2, 3], 2: [1, 3], 3: [1, 2, 4], 4: [0, 3]}
        self.lamda = 0.1
        self.noniidlevel = 0.8

if __name__ == '__main__':
    config = Configs()