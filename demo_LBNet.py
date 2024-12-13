#coding=utf-8
import argparse

import numpy
import numpy as np
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import cvxpy as cp
from UnmixingUtils import UnmixingUtils
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import scipy.io
from tqdm import tqdm

class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A, _a):
        super(L1NMF_Net, self).__init__()
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        theta = np.ones((1, R)) * 0.01 * L
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2
        self.p = nn.ParameterList()
        self.a = nn.ParameterList()
        self.L = nn.ParameterList()
        self.theta = nn.ParameterList()
        self.L2 = nn.ParameterList()
        self.W_a = nn.ParameterList()
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.L.append(nn.Parameter(torch.FloatTensor([L])))
            self.L2.append(nn.Parameter(torch.FloatTensor([L2])))
            self.theta.append(nn.Parameter(torch.FloatTensor(theta)))
            self.p.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.W_a.append(nn.Parameter(torch.FloatTensor(temp)))
            self.a.append(nn.Parameter(torch.FloatTensor([_a])))
        self.layerNum = layerNum

    def forward(self, X, _M, _A):
        self.W_m = _A
        M = list()
        M.append(_M)
        A = list()
        _A = _A.T
        A.append(_A)
        Z = list()
        Z.append(_A)
        
        for k in range(self.layerNum):
            theta = self.theta[k].repeat(A[-1].size(1), 1).T
            T = M[-1].mm(A[-1]) - X
            self.a[k] = self.self_update_a(self.a[k])
            try:
                A_inv = torch.linalg.inv(self.W_a[k].mm(self.W_a[k].T) + self.a[k] * torch.eye(self.W_a[0].shape[0]).to(device))
            except RuntimeError as e:
                break
            _Z = Z[-1] - self.L[k]*self.W_a[k].T.mm(A_inv).mm(T)
            Z.append(_Z)
            _A = self.sum2one(F.relu(self.self_active(_Z, self.p[k], theta)))
            A.append(_A)
            T = M[-1].mm(A[-1]) - X
            _M = M[-1] - T.mm(self.L2[k] * self.W_m)
            _M = F.relu(_M)
            M.append(_M)
        return M, A
    def half_thresholding(self, z_hat, mu):
        c=pow(54,1/3)/4
        tau=z_hat.abs()-c*pow(mu,2/3)
        v=z_hat
        ind=tau>0
        v[ind]=2/3*z_hat[ind]*(1+torch.cos(2*math.pi/3-2/3*torch.acos(mu[ind]/8*pow(z_hat[ind].abs()/3,-1.5))))
        v[tau<0]=0
        return v
    def soft_thresholding(self, z_hat, mu):
        return z_hat.sign() * F.relu(z_hat.abs() - mu)
    def self_active(self, x, p, lam):
        tau=pow(2*(1-p)*lam,1/(2-p))+p*lam*pow(2*lam*(1-p), (p-1)/(2-p))
        v = x
        ind = (x-tau) > 0
        ind2=(x-tau)<=0
        v[ind]=x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2]=0
        v[v>1]=1
        return v
    def self_update_a(self, x):
        if x <= 0:
            x = torch.FloatTensor([1e-3]).to(device)
        return x
    def calW(self,D):
        (m,n)=D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        constraint = [cp.diag(W.T @ D) == 1]
        prob = cp.Problem(obj, constraint)
        result = prob.solve(solver=cp.SCS, max_iters=1000)
        return W.value
    def sum2one(self, Z):
        temp = Z.sum(0)
        temp = temp.repeat(Z.size(0), 1) + 0.0001
        return Z / temp
class RandomDataset(Dataset):
    def __init__(self, data, label, length):
        self.data = data
        self.len = length
        self.label = label

    def __getitem__(self, item):
        return torch.Tensor(self.data[:,item]).float(), torch.Tensor(self.label[:,item]).float()

    def __len__(self):
        return self.len


def prepare_data(dataFile):
    data = scio.loadmat(dataFile)
    X = data['Y']
    A = data['M']
    s = data['A'].T
    A0 = data['M1']
    S0 = data['A1']
    return X, A, s, A0, S0

def prepare_train(X, s, trainFile):
    train_index = scio.loadmat(trainFile)
    train_index = train_index['train']
    train_index = train_index-1
    train_data = np.squeeze(X[:, train_index])
    train_labels = np.squeeze(s[:, train_index])
    nrtrain = np.size(train_index, 1)
    return train_data, train_labels, nrtrain

def prepare_init(initFile):
    init = scio.loadmat(initFile)
    A0 = init['Cn']
    S0 = init['o'][0, 0]['S']
    return A0, S0


def set_param(layerNum, lr, lrD, batch_size=4096):
    parser = argparse.ArgumentParser(description="LISTA-Net")
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
    parser.add_argument('--layer_num', type=int, default=layerNum, help='phase number of ISTA-Net')
    parser.add_argument('--learning_rate_decoder', type=float, default=lrD, help='learning rate for decoder')
    parser.add_argument('--learning_rate', type=float, default=lr, help='learning rate')
    parser.add_argument('--batch_size', type=float, default=batch_size, help='batch size')
    parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
    parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log directory')
    args = parser.parse_args(args=[])
    return args

class MyLoss(nn.Module):
    def forward(self, input, P):
        loss = input.T @ (P @ input)
        loss = torch.trace(loss)
        loss = abs(loss)/input.shape[1]
        return loss
def save_data(filename, A, S, A_true, Distance, meanDistance, rmse_all, rmse, stop_SAD, stop_rmse, min_SAD, min_rmse):
    Distance = np.concatenate((Distance[0], meanDistance), axis=0)  # 水平合并
    rmse_all = np.concatenate((np.atleast_1d(rmse_all), np.atleast_1d(rmse)), axis=0)

    scipy.io.savemat(filename, {'A': A.to('cpu').detach().numpy(),
                                                'S': S.to('cpu').detach().numpy(),
                                                'A_true': A_true,
                                                'SAD': Distance, 'rmse': rmse_all,
                                'stop_SAD':stop_SAD, 'stop_rmse':stop_rmse,
                                'min_SAD':min_SAD, 'min_rmse':min_rmse})

def save_sad_values_to_mat(sad_values, stop_index, filename, save_dir='result/'):
    best_index = sad_values.index(min(sad_values)) + 1
    scipy.io.savemat(save_dir+filename, {'sad_values': sad_values, 'stop_index': stop_index, 'best_index': best_index})


def plot_sad_values_from_mat(filename, save_dir='figure/'):
    data = scipy.io.loadmat(filename)
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    sad_values = data['sad_values'][0]
    stop_index = data['stop_index'][0]
    best_index = data['best_index'][0]
    x = list(range(1, sad_values.shape[0]+1))
    plt.plot(x, sad_values, label='Values')
    plt.scatter(stop_index, sad_values[stop_index-1], c='red', label='Stop')
    plt.scatter(best_index, sad_values[best_index-1], c='green', label='Best')
    plt.text(stop_index, sad_values[stop_index-1], str(np.round(sad_values[stop_index-1], 4)), ha='right')
    plt.text(best_index, sad_values[best_index-1], str(np.round(sad_values[best_index-1], 4)), ha='left')

    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title(filename)
    plt.legend()
    plt.savefig(save_dir+'%s.png' % (filename))
    plt.clf()

def create_file_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(lrD,layerNum, lr, train_data, test_data, nrtrain, A0, S0, X, A, s, SNR, lam, _a, learning_rate_a):
    util = UnmixingUtils(A, s.T)
    _M = torch.FloatTensor(A0).to(device)
    batch_size = nrtrain
    args = set_param(layerNum, lr, lrD, batch_size=batch_size)
    model_dir = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f" % (
        args.model_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    log_file_name = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f.txt" % (
        args.log_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    criterion2 = MyLoss()
    meanDistance = []
    meanDistance.append(0)
    rmse = 0
    model = L1NMF_Net(args.layer_num, A0, S0, _a)
    model.to(device)
    criterion = nn.MSELoss(reduction='sum')

    trainloader = DataLoader(dataset=RandomDataset(train_data, test_data, nrtrain), batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=False)
    learning_rate = args.learning_rate
    learning_rate_decoder=args.learning_rate_decoder
    opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                      {'params': [a for a in model.a],
                       'lr': learning_rate_a},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    running_loss = 0.0
    last_loss = 1
    loss = 0
    start_time = time.time()
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        a = 1e-3
        if epoch_i <= 5 and epoch_i % 2 == 0:
            learning_rate = learning_rate / 25
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                      {'params': [a for a in model.a],
                       'lr': learning_rate_a},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        if epoch_i > 10 and epoch_i % 10 == 0:
            learning_rate = learning_rate / 1.5
            learning_rate_decoder = learning_rate_decoder / 1.5
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p]},
                      {'params': [a for a in model.a],
                       'lr': learning_rate_a},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                       model.theta],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        for data_batch in trainloader:
            batch_x, batch_label = data_batch
            _A = torch.FloatTensor(batch_label).to(device)
            batch_x = batch_x.to(device)
            # shadow
            output_end, output_abun = model(batch_x.T,_M,_A)
            loss = sum([criterion2(output_end[i + 1] @ output_abun[i + 1] - batch_x.T,
                                     torch.linalg.inv(output_end[i + 1] @ output_end[i + 1].T  + a * torch.eye(output_end[i + 1].shape[0]).to(device))
                        ) for i in range(layerNum)])/layerNum
            opt.zero_grad()
            loss.backward()
            opt.step()
        for i in range(layerNum):
            t1 = model.p[i].data
            t1[t1 < 0] = 1e-4
            t1[t1 > 1] = 1
            model.p[i].data.copy_(t1)
            running_loss += loss.item()
        temp = abs(running_loss - last_loss) / last_loss
        output_data = 'train===epoch: %d, loss:  %.5f, tol: %.6f\n' % (epoch_i, running_loss, temp)
        print(output_data)
        last_loss = running_loss
        running_loss = 0.0
    end_time = time.time()
    _A0 = torch.FloatTensor(A0).to(device)
    _S0 = torch.FloatTensor(S0).to(device)
    out1, out2 = model(torch.FloatTensor(X).to(device), _A0, _S0.T)
    Distance, meanDistance, sor = util.hyperSAD(out1[-1].to('cpu').detach().numpy())
    rmse, rmse_all = util.hyperRMSE(out2[-1].to('cpu').T.detach().numpy(), sor)
    output_data = 'Res: SAD: %.4f RMSE:  %.4f' % (meanDistance, rmse)
    print(output_data)
    elapsed_time = end_time - start_time
    print("花费时间： " + str(elapsed_time))
    M = A
    A = s
    M_est = out1[-1].to('cpu').detach().numpy()
    A_est = out2[-1].to('cpu').detach().numpy()
    create_file_if_not_exists(savename)
    scipy.io.savemat(savename, {'M':M,'A':A,'M_est':M_est,'A_est':A_est})
    if plot ==True:
        title = 'LBNet'
        A_true_np = A
        E_np = M_est
        E_true = M
        spectral = E_np.shape[0]
        num = A_true_np.shape[0]
        if len(A_true_np.shape)==2:
            rc1 = int(np.sqrt(A_true_np.shape[1]))
            rc2 = rc1
        else:
            rc1 = A_true_np.shape[1]
            rc2 = A_true_np.shape[2]
        total_rows = E_np.shape[1]
        fig, axs = plt.subplots(total_rows, 1, figsize=(6, 6 * total_rows))
        fig2, axs2 = plt.subplots(total_rows, 1, figsize=(6,6 * total_rows))
        axs[0].set_title(title, fontsize=28)
        axs2[0].set_title(title, fontsize=28)
        for i in range(num):
            Y = A_true_np[i:i+1, :rc1*rc2].reshape([1, rc1, rc2])
            im = axs[i].imshow(Y[0, :, :], cmap='viridis', vmin=0, vmax=1, aspect='equal')
            X = np.array(range(spectral))
            Y2 = E_np[:spectral, i:i+1]
            Y = E_true[:spectral, i:i+1]
            axs2[i].plot(X, Y, label=f'Reference Endmember {i+1}', color='blue', linewidth=2)
            axs2[i].plot(X, Y2, linestyle='--', label=f'Estimated Endmember {i+1}', color='red', linewidth=2)
            axs2[i].legend(fontsize=12)
            axs[i].axis('off')
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig2.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.show()
    return meanDistance[0], rmse, elapsed_time

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = "syntheticdB20"
    dataFile = dataset+'_dataset.mat'
    trainFile = 'train_4096_1000.mat'
    layerNum = 24
    lr = 1e-1
    learning_rate_a = 1e-2
    lrD=1e-8
    learning_rate_a = 1e-2
    X, A, s, A0, S0 = prepare_data(dataFile)
    plot = True
    train_data, train_labels, nrtrain = prepare_train(X, S0, trainFile)
    sigma = 0
    savename = "result/LB_%s_result.mat" % (dataset)
    save_dir = dataset+"/"
    SAD, rmse, elapsed_time=train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels,
                    nrtrain=nrtrain, A0=A0, S0=S0,
        X=X, A=A, s=s.T, SNR='20dB', lam=0, _a=1, learning_rate_a=learning_rate_a)


