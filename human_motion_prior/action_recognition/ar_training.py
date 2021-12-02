import os

import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import torch.optim as optim
import math
import torch

from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type='focal', beta=0.9996, gamma=1):
    """
    provided by BABEL
    https://github.com/abhinanda-punnakkal/BABEL/blob/5b1eb790718de5b85998ec373281a12a82ea160e/action_recognition/class_balanced_loss.py
    """

    def focal_loss(labels, logits, alpha, gamma):
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log1p(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


class EmbedData(Dataset):
    def __init__(self, data_split, num_class, suffix):
        split = num_class
        data = np.load('ik_embedding/ik_embedding_babel{}_train_{}.npz'.format(split, suffix))

        self.embeddings = data['embedding']
        self.labels = data['labels']

        weight = np.unique(self.labels, return_counts=True)[1]
        self.sample_per_cls = weight

        self.u = np.mean(self.embeddings, axis=0)
        self.std = np.std(self.embeddings, axis=0)
        np.savez('ik_embedding/ik_embedding_ustd.npz', u=self.u, std=self.std)

        if data_split == 'val':
            data = np.load('ik_embedding/ik_embedding_babel{}_val_{}.npz'.format(split, suffix))
        elif data_split == 'test':
            data = np.load('ik_embedding/ik_embedding_babel{}_test_{}.npz'.format(split, suffix))

        if data_split == 'val' or data_split == 'test':
            # due to the subsample data augmentation in embedding generation
            # here we use combine two samples to give the final results
            self.embeddings = data['embedding'].reshape(-1, 2, 256)
            self.labels = data['labels'].reshape(-1, 2)[:, 0]

    def __getitem__(self, idx):
        # em = torch.tensor((self.embeddings[idx] - self.u) / self.std, dtype=torch.float32)
        em = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return em, label

    def __len__(self):
        return self.embeddings.shape[0]


class Net(nn.Module):
    def __init__(self, in_channel, num_class):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(in_channel, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, num_class)
        # nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        self.drop = nn.Dropout(0.5)

    def swish(self, Xin):
        return Xin / (1 + torch.exp(-Xin))

    def forward(self, z):
        feat = F.leaky_relu(self.fc0((z)))
        feat = F.leaky_relu(self.fc1(feat))
        return self.fc(self.drop(feat)), feat


def test(net, loader_val, num_class):
    net.eval()
    total_correct = 0
    total_sample = 0
    detail_acc = [[0, 0] for _ in range(num_class)]
    save_score = []
    for iter, data in enumerate(loader_val):
        em, gt_label = data
        em = em.to(device)
        gt_label = gt_label.to(device)

        # due to the subsample data augmentation in embedding generation
        # here we use combine two samples to give the final results
        for cnt in range(2):
            if cnt == 0:
                logits, _ = net(em[:, cnt])
            else:
                logits = logits + net(em[:, cnt])[0]

        save_score.append(logits.detach().cpu().numpy())

        _, predict_label = torch.max(logits, 1)

        total_correct += (predict_label == gt_label).sum().item()
        total_sample += predict_label.shape[0]

        for c in range(num_class):
            detail_acc[c][0] += (predict_label[gt_label == c] == gt_label[gt_label == c]).sum().item()
            detail_acc[c][1] += (gt_label[gt_label == c]).shape[0]

    save_score = np.concatenate(save_score, axis=0)
    np.savez('./submission{}.npz'.format(num_class), save_score)
    print('=============== acc: {} ============='.format(total_correct * 1.0 / total_sample))
    for c in range(num_class):
        print('class {}: acc: {}'.format(c, detail_acc[c][0] * 1.0 / detail_acc[c][1]))


if __name__ == '__main__':
    device = torch.device('cuda')
    FRAME_NUM = 128
    batch_size = 256 * 10
    num_class = 60
    embedding_channel = 256

    if not os.path.exists('ar_logs'):
        os.makedirs('ar_logs')

    net = Net(embedding_channel, num_class)
    net.to(device)
    opt_adam = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_schedule = optim.lr_scheduler.StepLR(opt_adam, step_size=200, gamma=0.5)

    # net.load_state_dict(torch.load('./ar_log/best.pth'))

    ds_train = EmbedData(data_split='train', num_class=num_class, suffix='noise')
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ds_val = EmbedData(data_split='val', num_class=num_class, suffix='noise')
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    ds_test = EmbedData(data_split='test', num_class=num_class, suffix='noise')
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    best = 0
    best_t = 0

    for e in range(1000):
        net.train()
        total_correct = 0.0
        total_sample = 0.0
        for iter, data in enumerate(loader_train):
            em, gt_label = data
            em, gt_label = em.to(device), gt_label.to(device)

            logits, feat = net(em)

            loss = CB_loss(gt_label, logits, ds_train.sample_per_cls, num_class)
            opt_adam.zero_grad()
            loss.backward()
            opt_adam.step()

            _, predict_label = torch.max(logits, dim=1)

            total_correct += (predict_label == gt_label).sum().item()
            total_sample += predict_label.shape[0]

        train_acc = total_correct * 1.0 / total_sample
        torch.save(net.state_dict(), './ar_logs/checkpoint.pth')

        net.eval()
        total_correct = 0
        total_sample = 0
        detail_acc = [[0, 0] for _ in range(num_class)]
        for iter, data in enumerate(loader_val):
            em, gt_label = data
            em = em.to(device)
            gt_label = gt_label.to(device)

            # due to the subsample data augmentation in embedding generation
            # here we use combine two samples to give the final results
            for cnt in range(2):
                if cnt == 0:
                    logits, _ = net(em[:, cnt])
                else:
                    logits = net(em[:, cnt])[0] + logits

            _, predict_label = torch.max(logits, 1)

            total_correct += (predict_label == gt_label).sum().item()
            total_sample += predict_label.shape[0]
            for c in range(num_class):
                detail_acc[c][0] += (predict_label[gt_label == c] == gt_label[gt_label == c]).sum().item()
                detail_acc[c][1] += (gt_label[gt_label == c]).shape[0]

        class_acc = np.zeros(num_class)
        for c in range(num_class):
            class_acc[c] = detail_acc[c][0] * 1.0 / detail_acc[c][1]

        if total_correct * 1.0 / total_sample > best:
            best = total_correct * 1.0 / total_sample
            best_t = class_acc.mean()
            torch.save(net.state_dict(), './ar_logs/best.pth')

        print('{}, train acc: {:.4f}, eval acc: {:.4f}, {:.4f} // best: {:.4f}, {:.4f}'.format(e, train_acc,
                                                                                               total_correct * 1.0 / total_sample,
                                                                                               class_acc.mean(), best,
                                                                                               best_t))
        lr_schedule.step()
