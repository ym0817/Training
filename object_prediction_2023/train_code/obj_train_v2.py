from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
import numpy as np
from obj_dataset import Dataset
from tensorboard_gp import TensorBoard
from datetime import datetime
from mobilenetv2 import mobile_half



LOG = True
if LOG:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(datetime.now().strftime('%Y%m%d_%H%M')+'_log')


def train(epoch, net, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_id = 0
    for (inputs, targets) in tqdm(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets.long()).sum().item()

        iters = epoch * len(trainloader) + batch_id
        if iters % 100 == 0:
            # acc = predicted.eq(targets.long()).sum().item()*1.0/targets.shape[0]
            los = loss*1.0/targets.shape[0]
            writer.add_scalar('train_loss', los, iters)
            # writer.add_scalar('train_acc', acc, iters)
            print("Iters: {:0>6d}/[{:0>2d}], loss: {:.8f}, learning rate: {}".format(iters, epoch, los, optimizer.param_groups[0]['lr']))
        batch_id += 1
        

def test(epoch, net, valloader, criterion, device):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            # _, predicted = outputs.max(1)
            total += targets.size(0)
            # correct += predicted.eq(targets.long()).sum().item()
    print("Iters: [{:0>2d}], loss: {:.8f}  ".format(epoch, test_loss / total))

    # Save checkpoint.
    # acc = 1.*correct/total
    # writer.add_scalar('test_acc', acc, epoch)
    # print("Acc in the val_dataset:%s" % acc)
    return test_loss


def main(args):
    # set net
    # net = ShuffleNetV2_MetaACON()
    # net = mobilenet_v2(pretrained=False)
    # net = ResNet18(num_classes= 8)
    # backbone_type = 'res18_'

    net = net = mobile_half(8)
    backbone_type = 'object_mbv2_'




    savemodel_dir = os.path.join(args.save_dir, backbone_type + datetime.now().strftime('%m%d_%H%M'))
    if not os.path.exists(savemodel_dir):
        os.makedirs(savemodel_dir)

    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if multi_gpus:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.MSELoss().to(device)
    criterion = torch.nn.SmoothL1Loss().to(device)
    # criterion = torch.nn.L1Loss().to(device)


    # criterion = LSR()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)


    dataset = Dataset(args.file_list)
    val_percent =0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    trainloader = data.DataLoader(train_dataset,batch_size=args.train_batchsize,shuffle=True,num_workers=args.num_workers)
    valloader = data.DataLoader(val_dataset,batch_size=args.val_batchsize,shuffle=True,num_workers=args.num_workers)

    # tensor_board = TensorBoard(64, 3, 80, 80)
    # tensor_board.visual_model(net)
    loss = 0
    for epoch in range(args.total_epoch + 1):
        learn_rate = scheduler.get_last_lr()[0]
        print("Learn_rate:%s" % learn_rate)
        train(epoch, net, trainloader, optimizer, criterion,device)
        val_loss = test(epoch, net, valloader, criterion, device)
        scheduler.step()

        # if val_loss < 10:
        #     print('Saving..')
        #     state = {'net': net.state_dict(), 'acc': val_loss,'epoch': epoch }
        #     if not os.path.exists(savemodel_dir):
        #         os.makedirs(savemodel_dir)
        #     model_name = 'epoch' + str(epoch) + 'val_loss' + str(round(val_loss,4)) + '_model.pth'
        #     torch.save(state, os.path.join(savemodel_dir,model_name))
        #     best_acc = val_loss

        if val_loss >= 0.6 or epoch % 3 == 0:
            model_name = 'epoch' + str(epoch) + '_model.pth'
            torch.save(net.state_dict(), os.path.join(savemodel_dir,model_name))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')

    parser.add_argument('--file_list', type=str, default='/media/ym/DATA1/Arhud_Prediction/DataProcess/object_csvfiles', help='train list')
    parser.add_argument('--gpus', type=str, default='-1', help='model prefix')
    parser.add_argument('--train_batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--val_batchsize', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=52)
    parser.add_argument('--total_epoch', type=int, default=1000, help='total epochs')
    # parser.add_argument('--display_freq', type=int, default=100, help='print loss frequency')
    # parser.add_argument('--save_freq', type=int, default=4000, help='save frequency')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='model save dir')
    # parser.add_argument('--pad', type=bool, default=True, help=' wether image is paded')


    args = parser.parse_args()

    main(args)





