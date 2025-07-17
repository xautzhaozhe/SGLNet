import numpy as np
import scipy.io as sio
import os
import torchvision.transforms as transforms
import time
import random
from utils import ROC_AUC, Mahalanobis, MyTrainData
from SGLNet import SGLNet
from functionfile import *
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 定义早停类
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


def main(data_name, di):
    setup_seed(20)
    data = sio.loadmat('data/{}.mat'.format(data_name))
    input_data = data['data']
    print('{} size:'.format(data_name), input_data.shape)
    input_data = input_data / (np.max(input_data))
    Y = data['data'].astype(float).transpose(2, 0, 1)
    A = data['map'].astype(float)
    L, row, col = Y.shape
    Y = Y / (np.max(Y))
    Y = torch.from_numpy(Y).to(device)

    # Network setting
    EPOCH = 2000
    WD = 1e-6

    # Data loader
    time_start = time.time()
    train_dataset = MyTrainData(img=Y, gt=Y, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    net = SGLNet(in_ch=L, out_ch=L, head_ch=48, local_blc=2, sq=10, mb=di).to(device)
    L2_loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=5 * 1e-4, weight_decay=WD)
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    # train
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            net.train()
            out = net(x)
            loss_re = L2_loss(out, x)
            total_loss = loss_re
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += loss_re.item() * x.size(0)
            torch.save(net, 'checkpoint/{}_model.pth'.format(data_name))

        epoch_loss = running_loss / len(train_loader.dataset)
        if early_stopper.should_stop(epoch_loss):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    # test
    for i, (x, y) in enumerate(train_loader):
        net.eval()
        net = torch.load('checkpoint/{}_model.pth'.format(data_name))
        out = net(x)
        re_data = out.squeeze(0)
        re_data = re_data.detach().cpu().numpy().transpose(1, 2, 0)
        re_data = re_data / (np.max(re_data))
        result1 = Mahalanobis(input_data)
        result1 = result1 / np.max(result1)
        result2 = Mahalanobis(re_data - input_data)
        result2 = result2 / np.max(result2)
        result = result1 * result2
        time_end = time.time()
        print('{} training time：'.format(data_name), time_end - time_start)
        sio.savemat('./Result/{}_result.mat'.format(data_name), {'result': result})
        ROC_AUC(result, A)
        plt.imshow(result)
        plt.savefig('./Result/{}_result.png'.format(data_name), dpi=600)


if __name__ == '__main__':

    name_list = ['Hyperion', 'SPECTIR', 'abu-beach-3', 'abu-beach-4']
    for name in name_list:
        if name in ['SPECTIR']:
            di = 16
        else:
            di = 128
        main(data_name=name, di=di)
        print('********************')
