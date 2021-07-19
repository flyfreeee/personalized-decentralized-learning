#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
import time
import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, googlenet, ResNet18, AlexNet, LeNet
from utils import get_dataset, average_weights, exp_details

from cjltest.models import RNNModel
from cjltest.utils_model import MySGD


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def similarity_calculation(model_a, model_b, tau):
    param_a = torch.cat([params.view(-1) for layer, params in model_a.items()])
    param_b = torch.cat([params.view(-1) for layer, params in model_b.items()])

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = math.exp(cos(param_a, param_b) / tau)

    return similarity


''' input: 
    server_model = {1:model(odict([]))}
    client_models = {2:model(odict([])), 5:model(odict([]))}
    
    return aggregated_models = {2:odict([]), 5:odict([])}'''


def personalized_aggregation(server_model, client_models):
    aggregated_models = {}
    server_idx = list(server_model.keys())[0]
    server_param = server_model[server_idx].state_dict()

    for idx, i_model in client_models.items():
        similarities = {}
        param = i_model.state_dict()
        similarities[server_idx] = similarity_calculation(param, server_param, args.tau)
        aggregated_models[idx] = copy.deepcopy(server_param)
        for layer, params in aggregated_models[idx].items():
            params *= (1-args.self_attention) * similarities[server_idx]

        for neighbor, n_model in client_models.items():
            if neighbor == idx:
                continue
            n_param = n_model.state_dict()
            similarities[neighbor] = similarity_calculation(param, n_param, args.tau)
            for layer, params in n_param.items():
                aggregated_models[idx][layer] += (1-args.self_attention) * similarities[neighbor] * params

        total_similarity = sum(similarities.values())
        for layer, params in aggregated_models[idx].items():
            params /= total_similarity

        for neighbor, n_model in client_models.items():
            if neighbor == idx:
                n_param = n_model.state_dict()
                for layer, params in n_param.items():
                    aggregated_models[idx][layer] += args.self_attention * params
                break

    return aggregated_models


if __name__ == '__main__':

    seed = -1
    if seed < 0:
        seed = random.randint(0, 100000)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

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

    # build model
    if args.model == 'cnn':
        # Convolutional neural network
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

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # client_will = args.client_will

    aggregated_models_all = {}
    client_models = {}
    idx_users = list(user_groups.keys())
    info = pd.DataFrame(columns=['acc', 'loss'])
    info2 = pd.DataFrame(columns=['acc', 'loss'])
    info_personal = []
    info_global = []
    for idx in idx_users:
        info_personal.append(pd.DataFrame(columns=['acc', 'loss']))

    for idx in idx_users:
        info_global.append(pd.DataFrame(columns=['acc', 'loss']))

    for idx in idx_users:
        client_models[idx] = copy.deepcopy(global_model)

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        aggregated_models_all = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        # communication
        for idx in idx_users:

            # 'client' sends original models to 'server'
            neighbor_models = {}

            for neighbor in args.neighbors[idx]:
                neighbor_models[neighbor] = client_models[neighbor]

            # 'server' performs model aggregation
            if args.personalized:
                aggregated_models = personalized_aggregation({idx: client_models[idx]}, neighbor_models)
            else:
                aggregated_models = {}

                for neighbor in neighbor_models.keys():
                    aggregated_models[neighbor] = copy.deepcopy(client_models[idx].state_dict())

            # 'server' sends aggregated models to 'client'
            for receiver, model in aggregated_models.items():
                aggregated_models_all[receiver].append(model)

        if args.aggregation:
            if args.personalized:
                for idx in idx_users:
                    received_model_weights = [copy.deepcopy(m) for m in aggregated_models_all[idx]]
                    client_models[idx].load_state_dict(average_weights(received_model_weights))
            else:
                for idx in idx_users:
                    received_model_weights = [copy.deepcopy(m) for m in aggregated_models_all[idx]]
                    received_model_weights.append(client_models[idx])
                    client_models[idx].load_state_dict(average_weights(received_model_weights))

        # local update
        for idx in idx_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            my_aggregated_models = aggregated_models_all[idx]

            w, train_loss = local_model.update_weights(
                copy.deepcopy(client_models[idx]), my_aggregated_models, global_round=epoch)

            # test on personal test data
            acc, loss = local_model.inference(model=client_models[idx])

            client_models[idx].load_state_dict(copy.deepcopy(w))

            info_personal[idx] = info_personal[idx].append([{'acc': acc, 'loss': loss}])

            # local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(train_loss))
            local_acc.append(acc)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg = sum(local_acc) / len(local_acc)
        info = info.append([{'acc': acc_avg, 'loss': loss_avg}])

        print(epoch)

        # test on global test data
        global_losses = []
        global_acc = []
        for idx in idx_users:
            test_acc, test_loss = test_inference(args, client_models[idx], test_dataset)
            info_global[idx] = info_global[idx].append([{'acc': test_acc, 'loss': test_loss}])
            global_losses.append(test_loss)
            global_acc.append(test_acc)
        global_loss_avg = sum(global_losses) / len(global_losses)
        global_acc_avg = sum(global_acc) / len(global_acc)
        info2 = info2.append([{'acc': global_acc_avg, 'loss': global_loss_avg}])


        # print global training loss after every 'i' rounds

        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {loss_avg}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*acc_avg))

    # save records
    framework = 'personalized' if args.personalized else 'baseline'
    method = 'aggre' if args.aggregation else 'regul'

    for idx in idx_users:
        info_personal[idx].to_csv(
            '../records/' + framework + '_' + method + '_' + str(args.num_users) + 'user_' + args.dataset + '_' + str(args.local_ep) +
            '_' + str(args.lr) + '_' + str(idx) + '_' + datetime.now().strftime('%d-%H') + '_personal test.csv', index=None)
        info_global[idx].to_csv(
            '../records/' + framework + '_' + method + '_' + str(args.num_users) + 'user_' + args.dataset + '_' + str(args.local_ep) +
            '_' + str(args.lr) + '_' + str(idx) + '_' + datetime.now().strftime('%d-%H') + '_global test.csv', index=None)
    info.to_csv(
            '../records/' + framework + '_' + method + '_' + str(args.num_users) + 'user_' + args.dataset + '_' + str(args.local_ep) +
            '_' + str(args.lr) + '_avg_' + datetime.now().strftime('%d-%H') + '_personal test.csv', index=None)
    info2.to_csv(
        '../records/' + framework + '_' + method + '_' + str(args.num_users) + 'user_' + args.dataset + '_' + str(
            args.local_ep) +
        '_' + str(args.lr) + '_avg_' + datetime.now().strftime('%d-%H') + '_global test.csv', index=None)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    print(f'seed: {seed}')

    # Saving the objects train_loss and train_accuracy:
    # file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(info)), info['loss'], color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(framework, method, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(info)), info['acc'], color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(framework, method, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(info)), info['loss'], color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(framework, method, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Global Accuracy vs Communication rounds')
    plt.plot(range(len(info2)), info2['acc'], color='k')
    plt.ylabel('Average Global Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_global.png'.
                format(framework, method, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

# if __name__ == '__main__':
#     server_model = {1: torch.Tensor([1, 2])}
#     client_models = {2: torch.Tensor([2, 3]), 5: torch.Tensor([-5, 1])}
#     personalized_aggregation(server_model, client_models)
