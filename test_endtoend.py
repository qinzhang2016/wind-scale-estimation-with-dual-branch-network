from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from Model_optical import WindModelE2E
from VideoDataSet import VideoDataset
import torch
import os
import shutil
import argparse
from visdom import Visdom
import numpy as np
import random
from utils import AverageMeter
import torch.nn as nn
import torch.optim as optim
import cv2

from focal_loss import FocalLossV2
from losses.flow_loss import unFlowLoss
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description = 'Testing')
parser.add_argument('--model', default='./save_model_focal_unsperoptical', type=str, help = 'path to model')
parser.add_argument('--arch', default = 'resnet18', help = 'model architecture')
parser.add_argument('--lstm-layers', default=1, type=int, help='number of lstm layers')
parser.add_argument('--hidden-size', default=128, type=int, help='output size of LSTM hidden layers')
parser.add_argument('--fc-size', default=0, type=int, help='size of fully connected layer before LSTM')
parser.add_argument('--epochs', default=1000, type=int, help='manual epoch number')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-step', default=50, type=float, help='learning rate decay frequency')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--alpha', default=10, type=int)
parser.add_argument('--frame_dir', type=str, default='frame', help='path to frames')
parser.add_argument('--img_save_dir', type=str, default='results',
                            help='path to storage generated feature maps if needed')
parser.add_argument('--occ_from_back', default=True)
parser.add_argument('--w_l1', default=0.15, type=str)
parser.add_argument('--w_scales', default=[1.0, 1.0, 1.0, 1.0, 0.0])
parser.add_argument('--w_sm_scales', default=[1.0, 0.0, 0.0, 0.0, 0.0])
parser.add_argument('--w_smooth', default=75.0)
parser.add_argument('--w_ssim', default=0.85)
parser.add_argument('--w_ternary', default=0.0)
parser.add_argument('--warp_pad', default="border")
parser.add_argument('--with_bk', default= True)
parser.add_argument('--smooth_2nd', default= True)
args = parser.parse_args()



def save_checkpoint(state, is_best, epoch,  best_prec, val_prec, filename='.pth.tar'):
    torch.save(state, os.path.join('./save_model_focal_unsperoptical/', str(epoch) + '_test' + str(best_prec) + '_val' + str(val_prec) + filename),_use_new_zipfile_serialization=False)

def adjust_learning_rate(optimizer, epoch):
    if not epoch % args.lr_step and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    output = output.type(torch.float)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def update_confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix



def test(val_loader, model, criterion, criterion1, con_matrix):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (inputs, target, _) in enumerate(val_loader):
        #input_var = [input.cuda() for input in inputs]
        input_var = inputs.cuda()
        target_var = target.cuda()

        # compute output
        with torch.no_grad():
            output, _, ofs = model(input_var)
            #output = output[:, -1, :]

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 1))
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)


        for m in range(16):
            for n in range(9):
                save_result = (ofs[m,n,:,:,:]*255).cpu().numpy().astype(np.uint8)
                save_result = np.squeeze(save_result)
                save_result = np.transpose(save_result,axes=[1,2,0])

                cv2.imwrite(os.path.join('/home/data/xujialang/Wind_Project/Wind_TwoStream/save_model_focal_unsperoptical/images', str(i) + '_' + str(m) + '_' + str(n) + '_optical.png'), save_result)
        
        for m in range(16):
            for n in range(10):
                save_input = (input_var[m,n,:,:,:]*255).cpu().numpy().astype(np.uint8)
                save_input = np.squeeze(save_input)
                save_input = np.transpose(save_input,axes=[1,2,0])
                
                cv2.imwrite(os.path.join('/home/data/xujialang/Wind_Project/Wind_TwoStream/save_model_focal_unsperoptical/images', str(i) + '_' + str(m) + '_' + str(n) + '_rgb.png'), save_input)


        # confussion matrix
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # cm = confusion_matrix(target,np.argmax(output.data.cpu(),axis=1),labels)
        update_confusion_matrix(np.argmax(output.data.cpu(), axis=1), target, con_matrix)

        print('Test: [{0}/{1}]\t'
              'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader),
            top1=top1,
            top5=top5))

    return (top1.avg, top5.avg, con_matrix)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    data_dir_test = '/home/data/xujialang/Wind_Project/VisualWind_split/test'

    normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    transform = (transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToPILImage(),
        # transforms.RandomGrayscale(),
        # transforms.RandomCrop([224,224]),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.ToTensor()]
        # normalize]
    ),
                 transforms.Compose([
                     transforms.Resize([256,256]),
                     transforms.ToPILImage(),
                    #  transforms.CenterCrop(224),
                     transforms.ToTensor()]
                    #  normalize]
                 )
    )
    
    test_dataset = VideoDataset(data_dir_test, transform[1], flow=False, w=256, h=256)

    shuffle_dataset = True
    random_seed = 42
    if shuffle_dataset:
        setup_seed(random_seed)

    # Creating PT data samplers and loaders:
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if os.path.exists('/home/data/xujialang/Wind_Project/Wind_TwoStream/save_model_focal_unsperoptical/27_test86.44266916576184_val86.57467532467533.pth.tar'):
        # load existing model
        model_info = torch.load('/home/data/xujialang/Wind_Project/Wind_TwoStream/save_model_focal_unsperoptical/27_test86.44266916576184_val86.57467532467533.pth.tar')
        print("==> loading existing model '{}' ".format(model_info['arch']))
        model = WindModelE2E(args.arch, 11, args.lstm_layers, args.hidden_size)
        # print(model)
        model.cuda()
        # model.load_state_dict(model_info['state_dict'])
        model.load_state_dict(model_info['state_dict'])
        # best_prec = model_info['best_prec']
        # cur_epoch = model_info['epoch']
    else:
        print('No pre-trainning model')
    
    # Training on epochs
    con_matrix = np.zeros([11, 11])

    prec1, prec5, con_matrix = test(test_loader, model, 0, 0, con_matrix)

    print("------Validation Result------")
    print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1))
    print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
    print(con_matrix)
    print("-----------------------------")
