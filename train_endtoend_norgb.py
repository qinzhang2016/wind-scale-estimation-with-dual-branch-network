from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from Model_optical_noRGB import WindModelE2E
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

parser = argparse.ArgumentParser(description = 'Training')
parser.add_argument('--model', default='./save_model_focal_unsperoptical_norgb', type=str, help = 'path to model')
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
    torch.save(state, os.path.join('./save_model_focal_unsperoptical_norgb/', str(epoch) + '_test' + str(best_prec) + '_val' + str(val_prec) + filename),_use_new_zipfile_serialization=False)

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


def train(train_loader, model, criterion, criterion1, optical_loss_func, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    scaler = GradScaler()
    model.train()  # switch to train mode

    for i, (inputs, target, _) in enumerate(train_loader):
        #input_var = [input.cuda() for input in inputs]
        input_var = inputs.cuda()
        target_var = target.cuda()
        
        input_var = torch.where(torch.isnan(input_var), torch.full_like(input_var, 0), input_var)

        target_var_onehot = torch.nn.functional.one_hot(target_var, 11)

        # Check input
        # input_var_check = input_var.cpu().numpy()
        # print(np.any(np.isnan(input_var_check)))
        
        # target_var_check = target_var.cpu().numpy()
        # print(np.any(np.isnan(target_var_check)))

        # zero the parameter gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        # if i ==0:
        #     for m in range(16):
        #         for n in range(10):
        #             save_result = (input_var[m,n,:,:,:]*255).cpu().numpy().astype(np.uint8)
        #             save_result = np.squeeze(save_result)
        #             save_result = np.transpose(save_result,axes=[1,2,0])
        #             cv2.imwrite(os.path.join('/home/bigspace/xujialang/Wind_Project/dataset_rgb/', str(i) + str(m) + str(n) + '_result.png'), save_result)

        

        # compute output
        with autocast():
            output, frame_1221_frame = model(input_var)
        #output = output[:, -1, :]
        # output_check =output.cpu().detach().numpy()
        
        # print(np.any(np.isnan(output_check)))

        # output, frame_1221_frame = torch.where(torch.isnan(output), torch.full_like(output, 0), output),  torch.where(torch.isnan(output), torch.full_like(output, 0), output)

        # output_check =output.cpu().detach().numpy()
        
        # print(np.any(np.isnan(output_check)))

        # Caculate optical loss
        length = inputs.size(1)
        optical_loss = 0
        l_ph = 0
        l_sm = 0
        for m in range(length-1):
            first = input_var[:, m]
            second = input_var[:, m+1]
            img_pair = torch.cat([first, second], 1)
            optical_loss_i, l_ph_i, l_sm_i, _ = optical_loss_func(frame_1221_frame[m], img_pair)
            optical_loss+= optical_loss_i
            l_ph+= l_ph_i
            l_sm+= l_sm_i

        print('optical_loss: {}, l_ph: {}, l_sm: {}'.format(optical_loss,l_ph,l_sm))

        loss = criterion(output, target_var) + criterion1(output, target_var_onehot) + (optical_loss + l_ph + l_sm)/(length-1)
        losses.update(loss.item(), 1)

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 1))
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)
        
        # assert torch.isnan(loss).sum() == 0, print(loss)
        
        # compute gradient
        scaler.scale(loss).backward()
        # clip gradient
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        # assert torch.isnan(model.mu).sum() == 0, print(model.mu)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        # assert torch.isnan(model.mu).sum() == 0, print(model.mu)
        # assert torch.isnan(model.mu.grad).sum() == 0, print(model.mu.grad)

        print('Epoch: [{0}][{1}/{2}]\t'
              'lr {lr:.5f}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, i, len(train_loader),
            lr=optimizer.param_groups[-1]['lr'],
            loss=losses,
            top1=top1,
            top5=top5))

    return (losses.avg, top1.avg)


def update_confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def validate(val_loader, model, criterion, criterion1, con_matrix):
    # losses = AverageMeter()
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
            output, _ = model(input_var)
            #output = output[:, -1, :]
            # loss = criterion(output, target_var) + criterion1(output, target_var)
            # losses.update(loss.item(), 1)

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 1))
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)

        # confussion matrix
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # cm = confusion_matrix(target,np.argmax(output.data.cpu(),axis=1),labels)
        update_confusion_matrix(np.argmax(output.data.cpu(), axis=1), target, con_matrix)

        print('val: [{0}/{1}]\t'
              'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader),
            top1=top1,
            top5=top5))

    return (top1.avg, top5.avg, con_matrix)

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
            output, _ = model(input_var)
            #output = output[:, -1, :]

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 1))
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)

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
    # torch.backends.cudnn.benchmark = True
    # # initialize visdom
    # viz = Visdom(env='train_process')
    # train_loss_x, train_loss_y = 0, 0
    # win1 = viz.line(X=np.array([train_loss_x]), Y=np.array([train_loss_y]), opts=dict(title='train_Loss'))

    # # viz = Visdom(env='train_Acc')
    # train_acc_x, train_acc_y = 0, 0
    # win2 = viz.line(X=np.array([train_acc_x]), Y=np.array([train_acc_y]), opts=dict(title='train_Acc'))

    # # viz = Visdom(env='test_Acc')
    # test_acc_x, test_acc_y = 0, 0
    # win3 = viz.line(X=np.array([test_acc_x]), Y=np.array([test_acc_y]), opts=dict(title='test_Acc'))

    data_dir_train = '/home/data/xujialang/Wind_Project/VisualWind_split/train'
    data_dir_val = '/home/data/xujialang/Wind_Project/VisualWind_split/val'
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

    train_dataset = VideoDataset(data_dir_train, transform[0], flow=False, w=256, h=256)
    val_dataset = VideoDataset(data_dir_val, transform[1], flow=False, w=256, h=256)
    test_dataset = VideoDataset(data_dir_test, transform[1], flow=False, w=256, h=256)

    shuffle_dataset = True
    random_seed = 42
    if shuffle_dataset:
        setup_seed(random_seed)

    # Creating PT data samplers and loaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if os.path.exists(os.path.join(args.model, 'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(args.model, 'checkpoint.pth.tar'))
        print("==> loading existing model '{}' ".format(model_info['arch']))
        model = WindModelE2E(args.arch, 11, args.lstm_layers, args.hidden_size)
        # print(model)
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
        # model.ARFlow.load_state_dict(model_info['state_dict1'])
        # model.Classifier.load_state_dict(model_info['state_dict2'])
        # model.Classifier_rgb.load_state_dict(model_info['state_dict3'])
        best_prec = model_info['best_prec']
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.model):
            os.makedirs(args.model)
        # load and create model
        print("==> creating model '{}' ".format(args.arch))
        model = WindModelE2E(args.arch, 11, args.lstm_layers, args.hidden_size)
        # print(model)
        model.cuda()
        cur_epoch = 0

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion1 = FocalLossV2()
    optical_loss_func = unFlowLoss(args).cuda()

    criterion = criterion.cuda()
    criterion1 = criterion1.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=2, eta_min=0, last_epoch=-1)

    best_prec = 0

    # Training on epochs
    con_matrix = np.zeros([11, 11])
    for epoch in range(cur_epoch, args.epochs):
        # optimizer = adjust_learning_rate(optimizer, epoch)


        print(
            "---------------------------------------------------Training---------------------------------------------------")

        # train on one epoch
        train_loss, train_top1 = train(train_loader, model, criterion, criterion1, optical_loss_func, optimizer, epoch)

        print(
            "--------------------------------------------------Validation--------------------------------------------------")

        # evaluate on validation set
        prec1_val, prec5_val, con_matrix_val = validate(val_loader, model, criterion, criterion1, con_matrix)

        prec1, prec5, con_matrix = test(test_loader, model, criterion, criterion1, con_matrix)

        print("------Validation Result------")
        print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1))
        print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
        print("-----------------------------")

        # # visdom
        # viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win=win1, update='append')
        # viz.line(X=np.array([epoch]), Y=np.array([train_top1]), win=win2, update='append')
        # viz.line(X=np.array([epoch]), Y=np.array([prec1]), win=win3, update='append')

        # remember best top1 accuracy and save checkpoint
        if prec1 > best_prec:
            is_best = prec1 > best_prec
            best_prec = max(prec1, best_prec)
            print('best_prec %f' %best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'num_classes': 11,
                'lstm_layers': args.lstm_layers,
                'hidden_size': args.hidden_size,
                'fc_size': args.fc_size,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
                'confusion_matrix': con_matrix}, is_best, epoch+1, best_prec, prec1_val)
