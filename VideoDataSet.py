import torch
import cv2
import os
import numpy as np
import re
from torch.utils.data import Dataset
import random

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, time_seq=10, frame_rate=1, w=224, h=224, flow=False, flow_vector=False, spectral=False, grey=False):
        self.root_dir = root_dir
        self.transform = transform
        self.time_seq = time_seq
        self.frame_rate = frame_rate
        self.num_frames = time_seq * frame_rate
        self.w = w
        self.h = h
        self.flow = flow
        self.flow_vector = flow_vector
        self.spectral = spectral
        self.grey = grey
        self.classes = sorted(os.listdir(self.root_dir), key = lambda i:int(re.match(r'(\d+)',i).group()))  #number of classes
        self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]   #number of samples in each class
        self.acc_count = [self.count[0]]     #accumulated number of samples
        for i in range(1, len(self.count)):
            self.acc_count.append(self.acc_count[i - 1] + self.count[i])

    def __len__(self):
        return np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))

    def __getitem__(self, idx):
        # 找到idx相应的label
        for i in range(len(self.acc_count)):
            if idx < self.acc_count[i]:
                label = i
                break
        
        class_path = self.root_dir + '/' + self.classes[label]
        if label:
            file_path = class_path + '/' + sorted(os.listdir(class_path))[idx - self.acc_count[label]]
        else:
            file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) / self.frame_rate  # 帧率
        i = 0
        j = 0
        frames = torch.zeros(self.num_frames, 3, self.w, self.h)
        success, frame = cap.read()
        frame = np.nan_to_num(frame)
        while success:
            if i % fps == 0:
                frame = torch.from_numpy(frame)
                frame = frame.permute(2,0,1)  #HWc2CHW
                if self.transform:
                    frame = self.transform(frame)
                frames[j, :, :, :] = frame
                j = j + 1
                if j == self.num_frames:
                    break
            success, frame = cap.read()
            i = i + 1

        _, file_name = os.path.split(file_path)

        if self.flow_vector:
            flows = torch.FloatTensor(self.num_frames-1, 2, self.w, self.h)
            for i in range(len(frames)-1):
                first = cv2.cvtColor(frames[i].permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
                second = cv2.cvtColor(frames[i+1].permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
                inst = cv2.optflow.createOptFlow_DeepFlow()
                flow1 = inst.calc(first,second, None)
                flows[i, :, :, :] = torch.from_numpy(flow1).permute(2,0,1)
            return flows, label, file_name    #num_frames-1*2*w*h
        elif self.flow:
            flows2 = torch.FloatTensor(self.num_frames - 1, 3, self.w, self.h)
            for i in range(len(frames)-1):
                first = cv2.cvtColor(frames[i].permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
                second = cv2.cvtColor(frames[i+1].permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
                inst = cv2.optflow.createOptFlow_DeepFlow()
                flow1 = inst.calc(first,second, None)
                flow2 = show_flow_hsv(flow1)
                flows2[i, :, :, :] = torch.from_numpy(flow2).permute(2,0,1)  # num_frames-1*3*w*h
            return flows2, label, file_name
        else:
            return frames, label, file_name    #num_frames*channel*w*h

def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # 将直角坐标系光流场转成极坐标系
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    # 光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2  # angle弧度转角度
        hsv[..., 2] = 255
        # m = np.mean(mag)
        m = np.max(mag)
        if m!=0:
            mag = mag / m
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 279.9325980018167
        m = np.max(mag)
        mag = mag / m
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
    # hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from torchvision import transforms

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (inputs, target, file_name) in enumerate(train_loader):
        #input_var = [input.cuda() for input in inputs]
        # target_var = target.cuda(1)

        # Check input
        input_var_check = inputs.numpy()
        target_var_check = target.numpy()
        if (np.any(np.isnan(input_var_check))) or (np.any(np.isnan(target_var_check))):
            print(file_name)
        
        # print(i)
        # target_var_check = target_var.cpu().numpy()
        # assert not np.any(np.isnan(input_var_check))
        # assert not np.any(np.isnan(target_var_check))
        
        # input_var = torch.where(torch.isnan(input_var), torch.full_like(input_var, 0), input_var)

        # print(f.shape)
        # break
    for i, (inputs, target, file_name) in enumerate(val_loader):
        #input_var = [input.cuda() for input in inputs]
        # target_var = target.cuda(1)

        # Check input
        input_var_check = inputs.numpy()
        target_var_check = target.numpy()
        if (np.any(np.isnan(input_var_check))) or (np.any(np.isnan(target_var_check))):
            print(file_name)
        
        # print(i)
        # target_var_check = target_var.cpu().numpy()
        # assert not np.any(np.isnan(input_var_check))
        # assert not np.any(np.isnan(target_var_check))
        
        # input_var = torch.where(torch.isnan(input_var), torch.full_like(input_var, 0), input_var)

        # print(f.shape)
        # break

    for i, (inputs, target, file_name) in enumerate(test_loader):
        #input_var = [input.cuda() for input in inputs]
        # target_var = target.cuda(1)

        # Check input
        input_var_check = inputs.numpy()
        target_var_check = target.numpy()
        if (np.any(np.isnan(input_var_check))) or (np.any(np.isnan(target_var_check))):
            print(file_name)
        
        # print(i)
        # target_var_check = target_var.cpu().numpy()
        # assert not np.any(np.isnan(input_var_check))
        # assert not np.any(np.isnan(target_var_check))
        
        # input_var = torch.where(torch.isnan(input_var), torch.full_like(input_var, 0), input_var)

        # print(f.shape)
        # break

