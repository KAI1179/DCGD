import torch
import torchvision
import torch.nn as nn
import numpy as np
# from models import *
# from resnet_56 import *
import torchvision.transforms as transforms
import os
import scipy.io as scio
# from _xiu_923_self_1_model import small_network
# from _erxiu_r56_self_model import small_network
# from tinydataset import create_loader
# from models import *
# from models_resnet110 import *
# from wrapper import wrapper
# from resnet_sskd import resnet110
from mdistiller.models.cifar.resnet import resnet110, resnet56
from mdistiller.engine.utils import load_checkpoint
from mdistiller.models.cifar.M_resnet20 import M_resnet20
from mdistiller.models.cifar.M_resnet32 import M_resnet32
from mdistiller.models.cifar.M_resnet44 import M_resnet44
from mdistiller.distillers import Vanilla
from mdistiller.models.cifar.wrn import wrn_40_2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])


# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
# ])

transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )


trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# train_loader, test_loader, num_classes = create_loader(128, './data/', 'tiny-imagenet-200')

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# import pickle
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo)
#     return dict
#
# trainobj = unpickle('./data/cifar-100-python.tar.gz')
# for item in trainobj:
#     print(item)
# '''
# outputs:
# 	filenames
# 	batch_label
# 	fine_labels
# 	coarse_labels
# 	data
# '''
#
# exit()

# Model
print('==> Building model...')
# net = VGG('VGG16')
# net = VGG('VGG19')
# net = ResNet50()
# net = ResNet18()
# net = small_network()
# net = WRN40_2(100)
# net = resnet110()
# net = M_resnet20
# net = wrn_40_2(num_classes=100)
net = resnet56(num_classes=100)
# net = Vanilla(net)
net = net.cuda()
# model = torch.nn.DataParallel(net)
# net.load_state_dict(load_checkpoint('/data1/xukai_1/code/KD/mdistiller-master/download_ckpts/cifar_teachers/resnet110_vanilla/ckpt_epoch_240.pth')["model"])
# net.load_state_dict(load_checkpoint('/data1/xukai_1/code/KD/mdistiller-master/./output/cifar100_baselines/dkd,wrn_40_2,r20/student_best')["model"])
# net.load_state_dict(load_checkpoint('/data1/xukai_1/code/KD/mdistiller-master/download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth')["model"])
net.load_state_dict(load_checkpoint('/home/data3/xukai/code/KD/DKD/download_ckpts/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth')["model"])

# net = resnet110(num_classes=100)
# net = resnet44()
# net.linear = nn.Linear(64, 100)

# net.linear = nn.Linear(64, 200)
# net.cuda()
# net = torch.nn.DataParallel(net)

# net = model_dict['resnet110'](num_classes=100).cuda()
# state_dict = torch.load('./checkpoint/teacher_resnet110_seed0/ckpt/best.pth')['state_dict']
# net.load_state_dict(state_dict)
# net = wrapper(module=net).cuda()

# checkpoint = torch.load('./checkpoint/resnet110_teacher_1/ckpt.pth')
# checkpoint = torch.load('./checkpoint/vgg16_REA/ckpt.pth')
# checkpoint = torch.load('./checkpoint/resnet50/ckpt.pth')
# net.load_state_dict(checkpoint['net'])

net.eval()

# save_path = './checkpoint/_xiu_907_self_2_model/ckpt.pth'
# save_path = './checkpoint/_erxiu_r56_self_model/ckpt.pth'
# save_path = './checkpoint/ckpt.pth'
# save_path = './checkpoint/vgg16/ckpt.pth'
# save_path = './checkpoint/resnet56/ckpt.pth'
# save_path = './checkpoint/_956_1_1_model/ckpt.pth'
# save_path = './checkpoint/vgg19/ckpt.pth'
# save_path = './checkpoint/resnet50/ckpt.pth'
# save_path = './checkpoint/resnet18/ckpt.pth'
# save_path = './checkpoint/WRN-40-2_1/ckpt.pth'
# save_path = './checkpoint/resnet110/ckpt.pth'
# state_dict = torch.load('./checkpoint/teacher_resnet110_seed0/ckpt/best.pth')['state_dict']
# net.load_state_dict(state_dict)


# net.load_state_dict(torch.load(save_path)['net'])

# # save_path = './checkpoint/_8_model/ckpt.pth'
# save_path = './checkpoint/resnet5050/ckpt.pth'
# # save_path = './checkpoint/vgg19/ckpt.pth'
# # save_path = './checkpoint/resnet50/ckpt.pth'
# net.load_state_dict(torch.load(save_path)['net'])


ff = torch.FloatTensor(4, 11).zero_().cuda()
ff_flag = True


total = 0
correct = 0

error_logits = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # outputs = net(inputs)

        outputs, _ = net(inputs)
        # loss = criterion(outputs, targets)
        #
        # test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        # targets_list = targets.tolist()
        # predicted_list = predicted.tolist()

        # for i in range(len(targets_list)):
        #     if targets_list[i] != predicted_list[i]:
        #         outputs_list = outputs.tolist()
        #         outputs_list.insert(0, targets_list[i])
        #         error_logits.append(outputs_list)


        correct += predicted.eq(targets).sum().item()

    # epoch_loss = test_loss / (batch_idx + 1)
    epoch_acc = correct / total
    print('Test Acc: {:.4f}'.format( epoch_acc))

# exit()



with torch.no_grad():
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        # outputs = net(inputs)
        outputs, _ = net(inputs)

        targets = targets.reshape([-1, 1])
        targets = targets.float()
        outputs = outputs.float()

        cat_mid = []
        cat_mid.append(targets)
        cat_mid.append(outputs)
        mid = torch.cat(cat_mid, dim=1)

        if ff_flag == True:
            cat_ff = []
            cat_ff.append(mid)
            ff_flag = False
        else:
            cat_ff = []
            cat_ff.append(mid)
            cat_ff.append(ff)

        ff = torch.cat(cat_ff, dim=0)



# print(ff.shape)

ff_numpy = ff.cpu().numpy()
print(ff_numpy.shape)

ff_numpy = ff_numpy[ff_numpy[:,0].argsort()]  # 排序

ff_numpy_predict = np.argmax(ff_numpy[:, 1:], axis=1)

# print(ff_numpy)
# print(ff_numpy_predict)

tar_pred = np.vstack((ff_numpy[:,0], ff_numpy_predict))
# print(tar_pred)
# print(tar_pred.shape)
#
# exit()
# print(tar_pred)
# exit()
# correct_test = './result/correct_test_resnet56.mat'
# correct_test = './result/_956_1_1.mat'
# correct_test = './result/_958.mat'
# correct_test = './result/_xiu_923_self_1_model.mat'
# correct_test = './result/_erxiu_r56_self_model.mat'
# correct_test = './result/_xiu_901_self_1.mat'
# correct_test = './result/correct_test_resnet50.mat'
# correct_test = './result/correct_test_vgg16.mat'
# correct_test = './result/resnet110_dkd.mat'
# correct_test = './result/wrn40_2_dkd.mat'
correct_test = './result/resnet56_dkd.mat'
scio.savemat(correct_test, {'train':tar_pred})




