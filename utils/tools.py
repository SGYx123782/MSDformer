import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj =='type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 1 else args.learning_rate * (0.8 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual1(true, preds=None, seasonal=None, trend=None, hp=None, gttt=None, gtres=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    matplotlib.use('TkAgg')

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.figure()

    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Fedformer [27]', linewidth=2)
        # plt.plot(preds - true, label='Prediction-GroundTruth', c="g", alpha=0.6, linewidth=2)
        # plt.plot(preds - preds, label='Zero', c="k", linewidth=2)
    if seasonal is not None:
        plt.plot(trend, label='Trend Component', linewidth=2)
        plt.plot(seasonal, label='Seasonal Component', linewidth=2)
        # plt.plot(hp, label='Historical_Part', linewidth=2)

    # plt.plot(true, label='GroundTruth', linewidth=2)
    # if preds is not None:
    #     plt.plot(preds, label='Prediction', linewidth=2)
    #     plt.plot(preds-true, label='Prediction-GroundTruth', c="g", alpha=0.6, linewidth=2)
    #     plt.plot(preds - preds, label='Zero', c="k", linewidth = 2)

    sparse_indices = np.arange(0, true.shape[0], 25)
    # plt.plot(true, label='GroundTruth', marker=">", markevery=sparse_indices, linewidth=2)
    # if preds is not None:
    #     plt.plot(preds, label='Prediction', marker="x", markevery=sparse_indices, linewidth=2)
    #
    # plt.plot(true, label='GroundTruth', linewidth=2)
    # if seasonal is not None:
    #     plt.plot(trend, label='Prediction_Trend', marker="o", markevery=sparse_indices, linewidth=2)
    #     plt.plot(seasonal, label='Prediction_Seasonal', marker="^", markevery=sparse_indices, linewidth=2)
    #     # plt.plot(hp, label='Historical_Part', linewidth=2)
    # plt.yticks([-1.4, -1.2,-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4], labels=["-1.4", "-1.2", "-1.0", " -0.8", "-0.6","-0.4","-0.2", "0", "0.2", "0.4"],fontsize=13)
    # plt.xticks([0, 25,  50, 75,  100, 125, 150, 175, 200], labels=["0", "25", "50", "75", "100","125","150", "175", "200" ],fontsize=13)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim([0, true.shape[0]])
    # plt.ylim([-1.4, 0.4])
    plt.legend(fontsize =12)

    plt.xlabel('Reference Length', fontsize=16)
    plt.ylabel('Standrad Values', fontsize=16)

    # plt.grid(axis='y', linestyle='--')
    # plt.grid(axis='x', linestyle='--')
    # plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.grid(axis='x', linestyle='--')

    plt.savefig(name, bbox_inches='tight')

def choose_seed(fix_seed):
    ##选择随机种子数，与以前的文章表示相同
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)