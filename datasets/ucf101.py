# this file is built to load data from dataset

import os
import torch.utils.data as data
import cv2
import sys
import random
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse
import collections
import torch.nn.functional as F
from config import ActivityConfig as cfg
from PIL import Image
from prefetch_generator import BackgroundGenerator

sys.path.append('..')
envs = os.environ


# using prefetch_generator
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SSL_Dataset(data.Dataset):

    def __init__(self, root, mode='train', args=None):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])

        self.root = root
        self.mode = mode
        self.args = args
        self.toPIL = transforms.ToPILImage()
        self.tensortrans = transforms.Compose([transforms.ToTensor()])

        self.split = '1'

        train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
        test_split_path = os.path.join(root, "split", 'testlist0' + self.split + '.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        self.test_split = pd.read_csv(test_split_path, header=None, sep=' ')[0]

        if mode == 'train':
            self.list = self.train_split
        else:
            self.list = self.test_split

        self.batch_size = 8
        self.sample_step_list = [1, 2, 4, 8]
        self.sample_retrieval = collections.OrderedDict()

    def __getitem__(self, index):

        videodata, sample_step_label = self.loadcvvideo_Finsert(index, sample_step=None)
        videodata, p_label = self.frames_permute(videodata)
        clip = self.crop(videodata)
        # print("flag 1: ", clip.size())
        sample_step = self.sample_step_list[sample_step_label]
        sample_inds = torch.arange(0, len(videodata), step=sample_step)
        clip_rgb = clip[:, sample_inds, :, :]
        # print("THIS IS THE SHAPE OF RGB CLIP: ", clip.size())
        # build rgb_diff
        if cfg.TRAIN.FRAME_DIFF.REQUIREMENT:
            clip_diff = self.getFrameDiff(clip_rgb, cfg.TRAIN.FRAME_DIFF.LOW, cfg.TRAIN.FRAME_DIFF.HIGH)

        return clip_rgb, clip_diff, sample_step_label, p_label

    def frames_permute(self, frames):

        p_label = np.random.randint(low=0, high=8)
        frames_count = len(frames)
        p_imgs = []

        for i in range(frames_count):
            # in this loop, just flip single image in video but do not permute the whole video
            img = Image.fromarray(frames[i])
            if p_label == 0:
                # nothing to do
                p_img = img
            elif p_label == 1:
                # flip image upside down
                p_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif p_label == 2:
                # flip image left 2 right
                p_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif p_label == 3:
                # flip image upside down and left 2 right
                p_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                p_img = p_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif p_label == 4:
                # permute video to play-back
                p_img = img
            elif p_label == 5:
                # flip image upside down and permute video to play-back
                p_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif p_label == 6:
                # flip image left 2 right and permute video to play-back
                p_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif p_label == 7:
                # flip image upside down and left 2 right and permute video to play-back
                p_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                p_img = p_img.transpose(Image.FLIP_LEFT_RIGHT)

            p_img = np.array(p_img)
            p_imgs.append(p_img)

        if p_label == 4 or p_label == 5 or p_label == 6 or p_label == 7:
            # reverse the whole video stream by reverse the frames array
            p_imgs.reverse()

        return p_imgs, p_label

    def getFrameDiff(self, clip, low, high):
        # the shape of clip is permuted as (channel, frames, height, width)
        # after unsqueeze, the shape of clip is permuted as (batch, channel, frames. height, width)
        # batch: 1
        # channel: 3
        # frames: 16
        # height: 112
        # width: 112
        rc, rt, rh, rw = clip.size()
        img1 = clip[:, 0, ::]
        img2 = clip[:, 5, ::]
        img3 = clip[:, 10, ::]
        img4 = clip[:, 15, ::]
        imgs = [img1, img2, img3, img4]
        imgs_clip = torch.stack(imgs, dim=1)
        imgs_clip = torch.unsqueeze(imgs_clip, dim=0)
        diff_clip = []
        for t in range(imgs_clip.size(2) - 1):
            diff_frame = torch.pow((imgs_clip[:, : , t + 1, :, :] - imgs_clip[:, :, t, :, :]), 2)
            diff_clip.append(diff_frame)
        diff_clip = torch.stack(diff_clip, dim=2)
        diff_clip = F.avg_pool3d(diff_clip, kernel_size=(3, 28, 28), stride=(3, 7, 7))
        pb, pc, pt, ph, pw = diff_clip.size()
        reshape_diff_clip = diff_clip.view(pb, pc, pt, -1)
        diff_clip_min = torch.min(reshape_diff_clip, dim=3, keepdim=True)[0].expand_as(reshape_diff_clip)
        diff_clip_max = torch.max(reshape_diff_clip, dim=3, keepdim=True)[0].expand_as(reshape_diff_clip)
        diff_clip_max = torch.clamp(diff_clip_max, min=1e-06)
        reshape_diff_clip = (reshape_diff_clip - diff_clip_min) / (diff_clip_max - diff_clip_min)
        reshape_diff_clip = (high - low) * reshape_diff_clip + low
        diff_clip = reshape_diff_clip.view(pb, pc, pt, ph, pw)
        diff_patch = F.interpolate(diff_clip, size=(rt, rh, rw), mode='trilinear', align_corners=False)
        diff_patch = torch.squeeze(diff_patch, dim=0)
        return diff_patch

    def loadcvvideo_Finsert(self, index, sample_step=None):

        need = 16
        fname = self.list[index]
        fname = os.path.join(self.root, 'video', fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if sample_step is None:
            sample_step_label = np.random.randint(low=0, high=len(self.sample_step_list))
            sample_step = self.sample_step_list[sample_step_label]
        else:
            sample_step_label = self.sample_step_list.index(sample_step)

        sample_len = need * sample_step
        shortest_len = sample_len + 1
        while frame_count < shortest_len:
            index = np.random.randint(self.__len__())
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)
            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start = np.random.randint(0, frame_count - shortest_len + 1)
        if start > 0:
            start = start - 1
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < sample_len and retaining):
            retaining, frame = capture.read()
            if retaining is False:
                count += 1
                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()

        while len(buffer) < sample_len:
            index = np.random.randint(self.__len__());
            print('retaining:{} buffer_len:{} sample_len:{}'.format(retaining, len(buffer), sample_len))
            buffer, sample_step_label = self.loadcvvideo_Finsert(index, sample_step)
            print('reload')

        return buffer, sample_step_label

    def crop(self, frames):
        video_clips = []
        seed = random.random()
        for frame in frames:
            random.seed(seed)
            frame = self.toPIL(frame)
            frame = self.transforms(frame)
            video_clips.append(frame)
        clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        return clip

    def __len__(self):

        return len(self.list)


class Finetune_Dataset(data.Dataset):

    # def __init__(self, root, mode='train', args=None):
    def __init__(self, root, mode='train', split='1'):
        self.root = root
        self.mode = mode
        self.split = split
        self.data_name = "UCF-101"

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.ToTensor()
        ])

        self.videos = []
        self.labels = []
        self.toPIL = transforms.ToPILImage()

        class_idx_path = os.path.join(root, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.mode == 'train':
            train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('===> Use split ' + self.split)

    def loadcvvideo(self, fname, count_need=16):
        fname = os.path.join(self.root, 'video', fname)
        capture = cv2.VideoCapture(fname)

        if self.data_name == 'UCF-101':
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if count_need == 0:
            count_need = frame_count
        start = np.random.randint(0, frame_count - count_need + 1)

        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < count_need and retaining):
            retaining, frame = capture.read()

            if retaining == False:
                count += 1
                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1

        capture.release()

        return buffer, retaining

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_split) - 1
        else:
            return len(self.test_split) - 1

    def __getitem__(self, index):
        if self.mode == 'train':
            videoname = self.train_split[index]
        else:
            videoname = self.test_split[index]

        if self.mode == 'train':
            videodata, retaining = self.loadcvvideo(videoname, count_need=16)
            while retaining == False or len(videodata) < 16:
                print('reload')
                index = np.random.randint(self.__len__())
                videoname = self.train_split[index]
                videodata, retaining = self.loadcvvideo(videoname, count_need=16)

            video_clips = []
            seed = random.random()

            for frame in videodata:
                random.seed(seed)
                frame = self.toPIL(frame)
                frame = self.transforms(frame)
                video_clips.append(frame)

            clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        elif self.mode == 'test':
            videodata, retaining = self.loadcvvideo(videoname, count_need=0)
            while retaining == False or len(videodata) < 16:
                print('reload')
                index = np.random.randint(self.__len__())

                videoname = self.test_split[index]
                videodata, retaining = self.loadcvvideo(videoname, count_need=16)
            clip = self.gettest(videodata)
        label = self.class_label2idx[videoname[:videoname.find('/')]]

        return clip, label - 1

    def randomflip(self, buffer):
        print('===> flip')
        if np.random.randint() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def gettest(self, videodata):
        length = len(videodata)

        all_clips = []

        for i in np.linspace(8, length-8, 10):
            clip_start = int(i - 8)
            clip = videodata[clip_start: clip_start + 16]
            trans_clip = []
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = self.toPIL(frame)
                frame = self.transforms(frame)
                trans_clip.append(frame)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])

            all_clips.append(clip)

        return torch.stack(all_clips)



def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--lpls', type=bool, default=False, help='use lpls_loss or not')
    parser.add_argument('--msr', type=bool, default=False, help='use multi sample rate or not')
    parser.add_argument('--vcop', type=bool, default=True, help='predict video clip order or not')
    parser.add_argument('--num_order', type=int, default=2, help='number of video clip order to predict')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    '''
    args = parse_args()
    com = SSL_Dataset('/home/guojie/Dataset/UCF-101-origin', mode='train', args=args)
    train_dataloader = DataLoader(com, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    print(cfg.EXP_TAG)

    for i, (clip_rgb, clip_diff, sample_step_label, permute_label) in enumerate(train_dataloader):
        print("this is rgb clip size: ", clip_rgb.size())
        print("this is diff clip size: ", clip_diff.size())
        print("this is sample label: ", sample_step_label)
        print("this is permute label: ", permute_label)
        print('----------------------------------------------------------------------------------------------')
    '''

    args = parse_args()
    # def __init__(self, root, mode='train', split='1'):
    com = Finetune_Dataset('/home/guojie/Dataset/UCF-101-origin', mode='train', split='1')
    train_dataloader = DataLoader(com, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    for i, (clip, label) in enumerate(train_dataloader):
        print('this is clip: ', clip.size())
        print('this is label: ', label)
        print('---------------------------------------------------------------------------------------------')