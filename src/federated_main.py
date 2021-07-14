#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, googlenet, ResNet18, AlexNet, LeNet
from utils import get_dataset, average_weights, exp_details
import pandas as pd
import random
from cjltest.models import RNNModel
from cjltest.utils_model import MySGD

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def similarity_calculation(model_a, model_b): # TODO add temperature parameter tau

    param_a = torch.cat([params.view(-1) for layer, params in model_a.items()])
    param_b = torch.cat([params.view(-1) for layer, params in model_b.items()])

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = math.exp(cos(param_a, param_b))
    return similarity

# server_model = {1:odict([])}
# client_models = {2:Tensor([]), 5:Tensor([])}
def personalized_aggregation(server_model, client_models):

    aggregated_models = {}
    server_idx = list(server_model.keys())[0]
    server_param = server_model[server_idx]

    for idx, param in client_models.items():
        similarities = {}
        similarities[server_idx] = similarity_calculation(param, server_param)
        aggregated_models[idx] = copy.deepcopy(server_param)
        for layer, params in aggregated_models[idx].items():
            params *= similarities[server_idx]

        for neighbor, n_param in client_models.items():
            if neighbor == idx:
                continue
            similarities[neighbor] = similarity_calculation(param, n_param)
            to_be_aggregated = copy.deepcopy(n_param)
            for layer, params in to_be_aggregated.items():
                aggregated_models[idx][layer] += similarities[neighbor]*params

        total_similarity = sum(similarities.values())
        for layer, params in aggregated_models[idx].items():
            params /= total_similarity

    return aggregated_models

# if __name__ == '__main__':
#     server_model = {1: torch.Tensor([1, 2])}
#     client_models = {2: torch.Tensor([2, 3]), 5: torch.Tensor([-5, 1])}
#     personalized_aggregation(server_model, client_models)

if __name__ == '__main__':

    np.random.seed(0)
    torch.random.manual_seed(0)
    random.seed(0)
    start_time = time.time()
    acc_list = []
    loss_list = []
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)


    device = torch.device('cuda' if args.gpu else 'cpu')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            global_model = LeNet()
        elif args.dataset == 'PBT':
            global_model = RNNModel()

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.

    print(get_parameter_number(global_model))
    print('---------------------------------------------------------------------------------------')

    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    info = pd.DataFrame(columns=['test_acc', 'test_loss', 'train_acc', 'train_loss'])
    # client_will = args.client_will

    aggregated_models_all = {}
    clients_model = {}
    idxs_users = list(user_groups.keys())
    for idx in idxs_users:
        clients_model[idx] = global_model.state_dict()

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        aggregated_models_all = {0: [], 1: [], 2: [], 3: [], 4: []}

        for idx in idxs_users:

            neighbor_models = {}

            for one in args.neighbors[idx]:
                neighbor_models[one] = clients_model[one]
            aggregated_models = personalized_aggregation({idx: clients_model[idx]}, neighbor_models)

            for idx, model in aggregated_models.items():
                aggregated_models_all[idx].append(model)

        for idx in idxs_users:
            # if client_will[epoch][idx]:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            aggregated_models = aggregated_models_all[idx]

            w, train_loss = local_model.update_weights(
                copy.deepcopy(clients_model[idx]), aggregated_models, global_round=epoch)
            acc, loss = local_model.inference(model=global_model)

            clients_model[idx] = copy.deepcopy(w)
            # local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(train_loss))
            # local_acc.append(acc)
        #
        #
        print(epoch)
        # loss_avg = sum(local_losses) / len(local_losses)
        # acc_avg = sum(local_acc) / len(local_acc)
        #
        # test_acc, test_loss = test_inference(args, global_model, test_dataset)

        #
        #
        # info = info.append([{'test_acc': test_acc, 'test_loss': test_loss, 'train_acc': acc_avg, 'train_loss': loss_avg}])
        # info.to_csv(str(args.num_users)+'user_'+args.dataset + '_' + str(args.local_ep) +'_' + str(args.lr)+ '_with_free_will.csv', index=None)

        # print global training loss after every 'i' rounds

        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {loss_avg}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*acc_avg))



    # Saving the objects train_loss and train_accuracy:
    file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)




    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

