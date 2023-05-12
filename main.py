#!/usr/bin/env python3
# from Bio import SeqIO
# import Levenshtein as L
from tqdm import tqdm

import os.path as osp
import os
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Transformer
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pprint
# from torch import summary
from sklearn.utils import shuffle
import time
import math

import sys
import argparse
from torch.nn import MSELoss, L1Loss

from model_shipyard.model_shipyard import M_transformer, M_convED_10, M_convED_5, M_GRU, M_RNN
# from model_shipyard_two.shipyard import ResNet, Bottlrneck, GoogLeNet, Inception, VGG15, VGG19
from model_shipyard_third.model_resnet import ResBlk, ResBlk2, ResBlk3, ResNet18, ResNet19, ResNet20, ResNet21, ResNet22, ResNet23, ResNet24, ResNet25, ResNet26, ResNet27, ResNet29, ResNet30, ResNet32
from sklearn.metrics import roc_curve, auc, accuracy_score

torch.manual_seed(3407)
np.random.seed(0)

    
class Twin(torch.nn.Module):
    def __init__(self, model):
        super(Twin, self).__init__()
        
        self.model = model
        self.scaling = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        
    def forward(self, x):
        
        x, y = torch.unbind(x, dim=1)
        xx = self.model(x) * self.scaling 
        yy = self.model(y) * self.scaling
        
        return torch.sum((xx-yy)**2,dim=-1)

def load_train_data():
    print('loading training data...')
#     with np.load('/root/DNA-experiments/half_blood_test_data_unbatch_dummy_2.0.npz') as f:
#     with np.load('/root/DNA-CLUSTER/half_blood_train_data_unbatch_balanced_2.0.npz') as f:
    with np.load('/root/00WeiXiang/CC/train_dummy_ten.npz') as f:
#     with np.load('half_blood_test_data_unbatch_dummy_2.0.npz') as f:
        data_unbatch = np.array(f['train_data'],dtype=np.float32)
        y_unbatch = np.array(f['target_data'],dtype=np.float32)
#         data_unbatch = np.array(f['data_unbatch'],dtype=np.float32)
#         y_unbatch = np.array(f['y_unbatch'],dtype=np.float32)
    print(y_unbatch.shape)
    data_unbatch = data_unbatch[y_unbatch != 0]
    y_unbatch = y_unbatch[y_unbatch != 0]
    print(y_unbatch.shape)
#     if flag == 'neg':
#         data_unbatch = data_unbatch[y_unbatch < K]
#         y_unbatch = y_unbatch[y_unbatch < K]
#     elif flag == 'pos':
#         data_unbatch = data_unbatch[y_unbatch >= K]
#         y_unbatch = y_unbatch[y_unbatch >= K]
    return data_unbatch, y_unbatch
def load_test_data():
    print('loading testing data...')
#     with np.load('/root/DNA-experiments/half_blood_test_data_unbatch_dummy_2.0.npz') as f:
#     with np.load('/root/DNA-CLUSTER/half_blood_test_data_unbatch_balanced_2.0.npz') as f:
    with np.load('/root/00WeiXiang/CC/test_dummy_ten.npz') as f:
#     with np.load('half_blood_test_data_unbatch_dummy_2.0.npz') as f:
        data_unbatch = np.array(f['train_data'],dtype=np.float32)
        y_unbatch = np.array(f['target_data'],dtype=np.float32)
#         data_unbatch = np.array(f['data_unbatch'],dtype=np.float32)
#         y_unbatch = np.array(f['y_unbatch'],dtype=np.float32)
    print(y_unbatch.shape)
    data_unbatch = data_unbatch[y_unbatch != 0]
    y_unbatch = y_unbatch[y_unbatch != 0]
    print(y_unbatch.shape)
#     if flag == 'neg':
#         data_unbatch = data_unbatch[y_unbatch < K]
#         y_unbatch = y_unbatch[y_unbatch < K]
#     elif flag == 'pos':
#         data_unbatch = data_unbatch[y_unbatch >= K]
#         y_unbatch = y_unbatch[y_unbatch >= K]
    return data_unbatch, y_unbatch

def main(seed, args, model, path, loss):
    np.random.seed(seed)
    args_dict = vars(args)
    
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    model_save_path = path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_save_path = os.path.join(model_save_path,'{}-{}'.format(seed, time.strftime('%Y-%m-%d', time.localtime())))
    
    for key in np.sort(list(args_dict.keys())):
        model_save_path += '-{}-{}'.format(key,args_dict[key])
        
#     m = model(num_layers = args.num_layers, conv_channels=args.conv_channels)
    m = model(output_dim=args.output_dim)
    
    tmodel = Twin(m)
    tmodel.to(device)
    
#     train_data_unbatch, y_unbatch = load_train_data()
#     train_data_unbatch, y_unbatch = shuffle(train_data_unbatch, y_unbatch)
    train_data = DataLoader(list(zip(train_data_unbatch, y_unbatch)), batch_size=args.batch_size)
    
    optimizer = torch.optim.Adam(tmodel.parameters(), lr=args.lr)
    
    loss_recorder_list = []
    for epoch in range(args.epochs):
        loss_value, loss_recorder = train(tmodel, train_data, epoch, args.batch_size, optimizer, total_epochs=args.epochs, loss=loss)
        print('Loss/scaling: {}/{}'.format(loss_value, tmodel.scaling.item()))
        loss_recorder_list.append(loss_recorder)
#         if (epoch+1) % 5 == 0:
#             torch.save(tmodel.state_dict(), model_save_path+'_epoch{}_tmodel.pth'.format(epoch+1))
        torch.save(tmodel.state_dict(), model_save_path+'_epoch{}_model.pth'.format(epoch+1))    
        
#     torch.save(tmodel.state_dict(), model_save_path+'_tmodel.pth')
#     torch.save(m.state_dict(), model_save_path+'_model.pth')
    np.save(model_save_path+'_loss_recorder.npy',loss_recorder_list)
    
#     test_data_unbatch, test_y_unbatch = load_test_data()
    test_data = DataLoader(list(zip(test_data_unbatch,test_y_unbatch)), batch_size=args.batch_size, shuffle=True)
    y_true, y_pred = test(tmodel, test_data, args.batch_size)
    np.savez(model_save_path+'_y_true_y_pred.npz', y_true=y_true, y_pred=y_pred)
    est_error = np.mean(np.abs(y_true-y_pred))
    est_error_relative = np.mean(np.abs(y_true-y_pred)*(y_true!=0)/(y_true+1e-10))
    K = 11
    accuracy = accuracy_score(y_true>K, y_pred>K)
    neg_y_pred = y_pred[y_true<K]
    neg_y_true = y_true[y_true<K]
    neg_meanabs = np.mean(np.abs(neg_y_pred-neg_y_true))
    
    print('\ntest loss: {}'.format(est_error))
    print('test loss_percent: {}'.format(est_error_relative))
    print('test accuracy: {}'.format(accuracy))
    print("test negloss: {}".format(neg_meanabs))
    print('tmodel scaling: {}'.format(tmodel.scaling))
    with open(model_save_path+'_test_results.txt','w') as f:
        f.write('test est_error: {}\n'.format(est_error))
        f.write('test est_error_percent: {}\n'.format(est_error_relative))
        f.write('test accuracy: {}\n'.format(accuracy))
        f.write('test neg_error: {}\n'.format(neg_meanabs))
        f.write('tmodel scaling: {}\n'.format(tmodel.scaling))
    

def test(tmodel, test_data, batch_size):
    tmodel.eval()
    y_true = []
    y_pred = []
    for data, y in tqdm(test_data):
        y_true.append(y.numpy())
        output = tmodel(data.to(device))
        y_pred.append(output.cpu().detach().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def fit_chi_sqare_loss(output,target):
#     neg_log = -(-target-1.442695*torch.lgamma(target)-output*1.442695 + (target-1)*torch.log2(2*output)) - 1
#     neg_log = -(-target/2-1.442695*torch.lgamma(target/2)-output/2*1.442695 + target/2*torch.log2(output))
    neg_log = output - (target-1/2) * torch.log(output)
#     neg_log = -(-target-1.442695*torch.lgamma(target)-output*1.442695 + (target-1)*torch.log2(2*output)) - 1
#     neg_log = (output>0) * neg_log
#     loss_neg = neg_log * (target<40)
#     loss_pos = (output - K2)**2 * (target>=40) * (output<K2)
    return torch.mean(neg_log)
def newloss1(output, target):
    
    neg_logg = output - target * torch.log(output)
    
    return torch.mean(neg_logg)

def train(tmodel, train_data, epoch, batch_size, optimizer, total_epochs, loss):
    tmodel.train()

    loss_all = 0.
    running_loss = 0.
    t_start = time.time()
    loss_recorder = []
    for idx,(data,y) in enumerate(train_data):

        y = y.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = tmodel(data)

        loss_value = loss(output,y)
#         loss = my_loss(output,y)
        
        loss_value.backward()
        running_loss += loss_value.item()
        loss_recorder.append(loss_value.item())
        loss_all += loss_value.item()
        optimizer.step()
        if idx % 5 == 1:    # print every 2000 mini-batches
            sys.stdout.write('\r[%d, %5d, %d] loss: %.5f' %
                  (epoch + 1, idx + 1, len(train_data), running_loss / 5))
            sys.stdout.flush()
            running_loss = 0.0
    print('\ntime: {}'.format(time.time()-t_start))
    if total_epochs != 1:
        if (epoch+1) % (total_epochs//2) == 0:
            for param_group in optimizer.param_groups:
                print('changing lr from {} to {}.'.format(param_group['lr'], 0.1 * param_group['lr']))
                param_group['lr'] = 0.1 * param_group['lr']
    return loss_all/len(train_data), loss_recorder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
#     parser.add_argument('--model', type=str, default='all', help='models: M_convED_10, M_convED_5, M_GRU, M_RNN')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--output_dim', type=int, help='output_dim')


#     parser.add_argument('--num-layers', type=int, default=None, help='num Conv layer')
#     parser.add_argument('--n-conv', type=int, default=8, help='n_conv')
    parser.add_argument('--lr', type=float, default=0.0001, help='num transformer layer')
#     parser.add_argument('--pos-neg-flag', type=str, default='all', help='pos, neg, all')
#     parser.add_argument('--conv-channels', type=int, default=64, help='8,16,32')
#     parser.add_argument('--activation', type=str, default='all', help='pos, neg, all')
    args = parser.parse_args()
    
    global train_data_unbatch, y_unbatch
    train_data_unbatch, y_unbatch = load_train_data()
    train_data_unbatch, y_unbatch = shuffle(train_data_unbatch, y_unbatch)
    train_data = DataLoader(list(zip(train_data_unbatch, y_unbatch)), batch_size=args.batch_size)
    global test_data_unbatch, test_y_unbatch
    test_data_unbatch, test_y_unbatch = load_test_data()
    
    models = {'M_convED_10': M_convED_10,
             }
    dim_dict = [200,240]
    losses = {'l1loss': L1Loss(),
            'mse': MSELoss(),
              'Poi': newloss1
              }
#     if not args.model == 'all':
#         model_name = args.model
#         model = models[model_name]
#         loss_name = "entropy"
#         loss = losses[loss_name]
#         path = 'results_new/SE100/{}/{}'.format(model_name, loss_name)
#         seed=0
#         print('MODEL: {} # {} {}'.format(model_name, loss_name, seed))
#         main(seed, args, model, path, loss)
    for dim in dim_dict:
        args.output_dim = dim
        for model_name in models:
                model = models[model_name]
                for loss_name in losses:
                    loss = losses[loss_name]
                    path = 'Result/next_ten/{}/{}/{}'.format(args.output_dim, model_name, loss_name)
                    seed = 1
                    print('MODEL: {} # {} {}'.format(model_name, loss_name, seed))
                    main(seed, args, model, path, loss)
