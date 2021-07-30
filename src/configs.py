# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:51
# @Author  : LIU YI

class Configs(object):
    def __init__(self):
        self.data = 'cifar'
        self.user_num = 15
        self.gpu = 0
        self.rounds = 150
        self.local_ep = 1
        self.personalized = True
        self.tau = 0.5
        self.aggregation = False
        self.self_attention = 0.33
        self.one_hop = False
        self.common_set = True

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
            self.client_will = np.random.rand(self.user_num * self.rounds).round().reshape(self.rounds, self.user_num)
            pd.DataFrame(self.client_will).to_csv('100client_will.csv', columns=None, index=None, header=None)

        else:
            self.client_will = np.ones(self.user_num * self.rounds).reshape(self.rounds, self.user_num)

        # self.neighbors = {0: [1, 5], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 0]}
        # self.neighbors = {0: [1, 5], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5, 6], 5: [4, 0], 6: [4]}
        self.neighbors = {0: [14], 1: [2], 2: [1, 3, 13], 3: [2, 5], 4: [5], 5: [3, 4, 6, 7], 6: [5], 7: [5, 8], 8: [7, 9], 9: [8, 10], 10: [9, 11], 11: [10, 12], 12: [11], 13: [2, 14], 14: [13, 0]}
        self.lamda = 0.1
        self.noniidlevel = 0.5


if __name__ == '__main__':
    config = Configs()
