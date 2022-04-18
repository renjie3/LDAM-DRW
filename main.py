import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--local', default='', type=str, help='Run on dev node.')
parser.add_argument('--job_id', default='local', type=str, help='job_id')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--imbalance', action='store_true', default=False)
parser.add_argument('--imbalance_test', action='store_true', default=False)
parser.add_argument('--imbalance_memory', action='store_true', default=False)
parser.add_argument('--fine_label', action='store_true', default=False)
parser.add_argument('--fine_label_test', action='store_true', default=False)
parser.add_argument('--fine_label_memory', action='store_true', default=False)
parser.add_argument('--mode', default='', type=str, help='What function to be used')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='Path to load model.')

# args parse
args = parser.parse_args()

import os

if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

import utils
from model import Model
import datetime

import pickle

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, epoch, epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def normal(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs):
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_loss_acc': [], }
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.job_id, args.feature_dim, args.temperature, args.k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    best_train_loss = 8
    best_train_loss_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, epochs)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_best_test_acc_model.pth'.format(save_name_pre))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_best_train_loss_model.pth'.format(save_name_pre))
        results['best_test_acc'].append(best_acc)
        results['best_loss_acc'].append(best_train_loss_acc)
        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

def plot_feature(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs):
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_loss_acc': [], }
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), args.job_id, args.feature_dim, args.temperature, args.k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    feature_bank = []
    plot_feature = []
    plot_label = []
    plot_idx = []

    with torch.no_grad():
        model.eval()
        # generate feature bank
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            feature, out = model(data.cuda(non_blocking=True))
            feature_bank.append(feature)
    # [D, N]
    feature_bank = torch.cat(feature_bank, dim=0).detach().cpu().numpy()
    # [N]
    feature_labels = np.array(memory_loader.dataset.targets)
    c = 10
    plot_num = 500
    for i in range(c):
        class_idx = np.where(feature_labels == i)[0]
        plot_num_class = min(plot_num, len(class_idx))
        # print(plot_num_class)
        class_idx = class_idx[:plot_num_class]
        plot_idx.append(class_idx)
    # print(plot_idx)

    fine_class_num = 75

    major_idx_list = []
    sub_major_id2clsuter_list = []
    minor_idx_list = []
    for i in range(c):
        class_idx = np.where(feature_labels == i)[0]
        if len(class_idx) > 300: # this is just to tell whether it is major ot minor
            # print(i)
            major_feature = feature_bank[class_idx]
            results = utils.run_kmeans(major_feature, [fine_class_num], 0, temperature)
            sub_major_id2clsuter = results['im2cluster'][0].detach().cpu().numpy()
            major_idx_list.append(class_idx)
            sub_major_id2clsuter_list.append(sub_major_id2clsuter)
        else:
            minor_idx_list.append(class_idx)
        # input('done')
    
    fine_label = np.zeros(feature_labels.shape)
    for i, major_idx in enumerate(major_idx_list):
        sub_major_id2clsuter = sub_major_id2clsuter_list[i]
        for j, idx in enumerate(major_idx):
            fine_label[idx] = sub_major_id2clsuter[j] + i*fine_class_num

    fine_class_count = np.max(fine_label) + 1
    
    for i, minor_idx in enumerate(minor_idx_list):
        for idx in minor_idx:
            fine_label[idx] = fine_class_count
        fine_class_count += 1

    # f = open('./fine_label_{}.pkl'.format(fine_class_num), 'wb')
    # pickle.dump(fine_label, f)
    # f.close()

    plot_idx = np.concatenate(plot_idx, axis=0)
    plot_feature = feature_bank[plot_idx]
    # feature_labels = torch.tensor(train_loader.dataset.targets, device=device)
    plot_labels = fine_label[plot_idx]

    utils.plot_feature(plot_feature, plot_labels, save_name_pre)

def knn_test_fine_label(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs):
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_loss_acc': [], }
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), args.job_id, args.feature_dim, args.temperature, args.k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    feature_bank = []
    test_fine_label = []

    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            feature, out = model(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank.device).long()
        c = int(np.max(feature_labels.detach().cpu().numpy()) + 1)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_loader)
        test_target = []
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = model(data)
            test_target.append(target)

            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            print(one_hot_label.shape)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            test_fine_label.append(pred_labels[:, 0])
            # print(pred_labels[:, 0])
            # input()
        
        test_target = torch.cat(test_target, dim=0)
        test_fine_label = torch.cat(test_fine_label, dim=0)

    # TODO: make minor label unchanged
    for label in test_target:
        if label.item() >= 5:
            test_fine_label = label.item() + 370

    f = open('./fine_label_{}_test.pkl'.format(75), 'wb')
    pickle.dump(test_fine_label, f)
    f.close()
    

if __name__ == '__main__':
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    if not args.imbalance:
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    else:
        train_data = utils.IMBALANCECIFAR10Pair(root='data', imb_type='step', train=True, transform=utils.train_transform, download=True, fine_label_flag=args.fine_label)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
        if args.imbalance_memory:
            memory_data = utils.IMBALANCECIFAR10Pair(root='data', imb_type='step', train=True, transform=utils.test_transform, download=True, fine_label_flag=args.fine_label_memory)
            memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        else:
            memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
            memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        if args.imbalance_test:
            test_data = utils.IMBALANCECIFAR10Pair(root='data', imb_type='step', train=False, transform=utils.test_transform, download=True, fine_label_flag=args.fine_label_test)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        else:
            test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    if args.load_model:
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        if 'epoch' in checkpoints and 'state_dict' in checkpoints and 'optimizer' in checkpoints:
                model.load_state_dict(checkpoints['state_dict'])
                optimizer.load_state_dict(checkpoints['optimizer'])
        else:
            model.load_state_dict(checkpoints)
            # logger.info("File %s loaded!" % (load_model_path))

    if args.mode == 'normal':
        normal(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs)
    elif args.mode == 'plot_feature':
        plot_feature(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs)
    elif args.mode == 'knn_test_fine_label':
        knn_test_fine_label(model, train_loader, optimizer, memory_loader, test_loader, batch_size, epochs)
    else:
        raise('Wrong mode!!!')

