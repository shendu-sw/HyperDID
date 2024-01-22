import os
import time
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

##from model.vit_pytorch import ViT
from model.multiscale_vit import MultiScaleTNT # MultiScaleViT
#from model.model_init import weight_init as weight_init
from data_process import chooose_train_and_test_point, mirror_hsi, train_and_test_data, train_and_test_label
from evaluate import AvgrageMeter, accuracy, output_metric, cal_results, print_args, train_epoch, valid_epoch, test_epoch
#from model.CNN3D import HamidaEtAl
from model.SSFTTnet import SSFTTnet
#from model.LDA_SLIC import LDA_SLIC
#from model.RNN import GRU
from model.CEGCN import CEGCN
from model.TPPI.models import pResNet
from model.HybridSN import HybridSN
from model.contrastive_learning import ContrastiveLearn, PMLP, MLP, HamidaEtAl, pResNet, GRU


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston2013'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--gpu_id', type=int, default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--multiscale', type=int, default=7, help='scale infor')
parser.add_argument('--patch_dim', type=int, default=64, help='scale infor')
parser.add_argument('--pixel_dim', type=int, default=16, help='scale infor')
parser.add_argument('--depth', type=int, default=5, help='scale infor')
parser.add_argument('--env_num', type=int, default=2, help='scale infor')
parser.add_argument('--alpha', type=float, default=1.0, help='scale infor')
parser.add_argument('--beta', type=float, default=1.0, help='scale infor')
parser.add_argument('--gama', type=float, default=1.0, help='scale infor')
parser.add_argument('--sample', type=float, default=1.0, help='sample rate')
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#device = torch.device()
#-------------------------------------------------------------------------------
torch.cuda.set_device(args.gpu_id)
#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/indian/IndianPine.mat')
    color_mat = loadmat('./data/indian/AVIRIS_colormap.mat')
    TR = data['TR']
    TE = data['TE']
    input = data['input'] #(145,145,200)
elif args.dataset == 'Pavia':
    '''data = loadmat('/home/ubuntu/data/hyperspectral_dataset/PaviaU/PaviaU.mat')
    TRLabel = loadmat('/home/ubuntu/data/hyperspectral_dataset/PaviaU/1/TRLabel.mat')
    TSLabel = loadmat('/home/ubuntu/data/hyperspectral_dataset/PaviaU/1/TSLabel.mat')'''
    data = loadmat('./data/pavia/Pavia.mat')
    color_mat = loadmat('./data/pavia/pavia_colormap.mat')
    #print(data)
    input = data['input']
    TR = data['TR']
    TE = data['TE']
    #data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston2013':
    data = loadmat('./data/houston2013/Houston.mat')
    color_mat = loadmat('./data/houston2013/houston2013_colormap.mat')
    TR = data['TR']
    TE = data['TE']
    input = data['input'] #(145,145,200)
    
else:
    raise ValueError("Unkknow dataset")

label = TR + TE

num_classes = np.max(TR)

color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i] - input_min)/(input_max - input_min)

# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))

#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)  # 根据patch空间大小进行扩展

#-------------------------------------------------------------------------------
# load data
x_train, x_test, x_true = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
x_train=torch.from_numpy(x_train).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
# ****************environment class********************* #
#print('88888888888888888888888888:', x_train.reshape(x_train.shape[0],-1).shape)
#env_cls=KMeans(n_clusters=args.env_num, random_state=0).fit(x_train.reshape(x_train.shape[0],-1))
env_cls=AgglomerativeClustering(n_clusters=args.env_num).fit(x_train.reshape(x_train.shape[0],-1))
print("111111111111111111111111111111111111111111")
env_cls=torch.from_numpy(env_cls.labels_).type(torch.LongTensor) #[695]
print(env_cls)
print(y_train.shape)
Label_train=Data.TensorDataset(x_train, y_train, env_cls)
x_test=torch.from_numpy(x_test).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)
x_true=torch.from_numpy(x_true).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x_true,y_true)

print('########################', Label_train)

sample = args.sample

train_length, val_length = int(len(Label_train) * sample), len(
            Label_train
        ) - int(len(Label_train) * (sample))
print('##########################',train_length)
Label_train, _ = torch.utils.data.random_split(
                        Label_train, [train_length, val_length]
        )

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

print(label_train_loader)

######################################################################
######################################################################
model = ContrastiveLearn(band, num_classes, patch_size=args.patches)
#model = HamidaEtAl(band, num_classes)
#model = pResNet(args.dataset)
# model = GRU(num_classes, args.patches)
######################################################################
#pseudo_num_class = num_classes
discriminator = PMLP(layers=[128, 64, 64], num_classes=args.env_num)

contrastor = MLP(layers=[128, 64, 64], num_classes=args.env_num)
######################################################################

model = model.cuda()
discriminator = discriminator.cuda()
contrastor = contrastor.cuda()

# criterion
criterion = nn.CrossEntropyLoss().cuda()

env_criterion = nn.CrossEntropyLoss().cuda()

# pseudo_criterion
pseudo_criterion = nn.CosineEmbeddingLoss().cuda()
adver_pseudo_criterion = nn.CosineEmbeddingLoss().cuda()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, dampening=0, weight_decay=args.weight_decay, nesterov=False)
optimizer1 = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate, momentum=0.9, dampening=0, weight_decay=args.weight_decay, nesterov=False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)

OA3=0
AA_mean3=0
Kappa3=0
AA3=0

#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    
    tic=time.time()
    model.load_state_dict(torch.load('./checkpoint/'+args.dataset+'.pkl')) 
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    print("**************************************************")
    #print("Parameter:")
    
    toc=time.time()
    etime=toc-tic
    print('time:', etime)
    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    #pre_u = test_epoch(model, label_test_loader, criterion, optimizer)
    
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    plt.subplot(1,1,1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('./results/'+args.dataset+'.png')
    savemat('./results/matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches): 
        scheduler.step()

        # train model
        #t0=time.time()
        model.train()

        train_acc, train_obj, tar_t, pre_t = train_epoch(model, discriminator, contrastor, label_train_loader, criterion, pseudo_criterion,adver_pseudo_criterion,env_criterion, optimizer, optimizer1, alpha=args.alpha, beta=args.beta, gama=args.gama)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                        .format(epoch+1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            
            if (OA3 < OA2):
                OA3 = OA2
                AA_mean3 = AA_mean2
                Kappa3 = Kappa2
                AA3 = AA2
                torch.save(model.state_dict(), './checkpoint/'+args.dataset+'.pkl')
                print("Epoch: {:03d} OA: {:.4f} AA: {:.4f} Kappa: {:.4f}"
                        .format(epoch+1, OA3, AA_mean3, Kappa3))

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA3, AA_mean3, Kappa3))
print(AA3)
print("**************************************************")
print("Parameter:")

print_args(vars(args))

