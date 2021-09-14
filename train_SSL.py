# 2021-09-01 this file is built to train model by self-supervised training strategy
# 2021-09-14 modified, support wandb package to log experiment information

from config import ActivityConfig as cfg
from tools import accuracy, AverageMeter, print_config, parse_args, save_config, show_info
from torch import nn, optim
import os
from models import C3D
from datasets import ucf101
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from apex import amp
from prefetch_generator import BackgroundGenerator
import sys
import wandb


def train(train_loader, model, criterion_CE, optimizer, epoch, start_loss, start_acc, root_path=None):
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    model.train()

    train_result_dict = {
        'train_loss': start_loss,
        'train_acc': start_acc,
    }
    final_loss = None
    final_acc = None

    for step, (clip_rgb, clip_diff, sample_step_label, p_label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # prepare data, send data to CUDA
        clip_rgb = clip_rgb.cuda()
        clip_diff = clip_diff.cuda()
        sample_step_label = sample_step_label.cuda()
        p_label = p_label.cuda()
        input_tensor = torch.cat((clip_rgb, clip_diff), dim=2)
        input_tensor = input_tensor.cuda()

        # train and calculate loss
        rgb_output, diff_output = model(input_tensor)
        result = None
        if cfg.TRAIN.FUSION == 'weight-average':
            result = rgb_output * cfg.TRAIN.RGB_WEIGHT + diff_output * cfg.TRAIN.DIFF_WEIGHT
        loss = criterion_CE(result, p_label)

        # update model
        optimizer.zero_grad()
        # use APEX to accelerate
        if cfg.APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif not cfg.APEX:
            loss.backward()
        optimizer.step()

        # update information
        losses.update(loss.item(), input_tensor.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        prec_class = accuracy(result.data, p_label, topk=(1,))[0]
        acc.update(prec_class.item(), input_tensor.size(0))

        # log information in weight-and-bias
        train_result_dict['train_loss'] = losses.avg
        train_result_dict['train_acc'] = acc.avg
        wandb.log({**train_result_dict, **{'epoch': epoch + 1}})

        # show information in one step and log
        if (step + 1) % cfg.SHOW_INFO == 0:
            # TODO: build a function to show information in terminal
            # def show_info(epoch, step, loader_len, batch_time, data_time, losses, acc, log_type, root_path):
            show_info(epoch, step, len(train_loader), batch_time, data_time, losses, acc, 'train', root_path)

        # log final loss and acc
        final_loss = losses.avg
        final_acc = acc.avg

    # show information in one epoch and log
    # show(info)

    return final_loss, final_acc


def validation(valid_loader, model, criterion_CE, optimizer, epoch, start_loss, start_acc, root_path=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()
    end = time.time()

    valid_result_dict = {
        'valid_loss': start_loss,
        'valid_acc': start_acc,
    }
    final_loss = None
    final_acc = None

    with torch.no_grad():
        for step, (clip_rgb, clip_diff, sample_step_label, p_label) in enumerate(valid_loader):
            data_time.update(time.time() - end)

            # prepare data, send data to CUDA
            clip_rgb = clip_rgb.cuda()
            clip_diff = clip_diff.cuda()
            sample_step_label = sample_step_label.cuda()
            p_label = p_label.cuda()
            input_tensor = torch.cat((clip_rgb, clip_diff), dim=2)
            input_tensor = input_tensor.cuda()

            # train and calculate loss
            rgb_output, diff_output = model(input_tensor)
            result = None
            if cfg.TRAIN.FUSION == 'weight-average':
                result = rgb_output * cfg.TRAIN.RGB_WEIGHT + diff_output * cfg.TRAIN.DIFF_WEIGHT
            loss = criterion_CE(result, p_label)

            # update information
            losses.update(loss.item(), input_tensor.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            prec_class = accuracy(result, p_label, topk=(1,))[0]
            acc.update(prec_class.item(), input_tensor.size(0))

            # log information in weight-and-bias
            valid_result_dict['valid_loss'] = losses.avg
            valid_result_dict['valid_acc'] = acc.avg
            wandb.log({**valid_result_dict, **{'epoch': epoch + 1}})

            if (step + 1) % cfg.SHOW_INFO == 0:
                # TODO: build a function to show information in terminal
                # def show_info(epoch, step, loader_len, batch_time, data_time, losses, acc, log_type, root_path):
                show_info(epoch, step, len(valid_loader), batch_time, data_time, losses, acc, 'valid', root_path)

            # log final loss and acc
            final_loss = losses.avg
            final_acc = acc.avg

    avg_loss = losses.avg
    return final_loss, final_acc, avg_loss


def main():
    # prepare environment, build weight-and-bias environment, and show config information
    wandb.init(
        project=str(cfg.EXP_TAG),
        name=str(time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime()) + '_SSL'),
        notes='SSL',
        config=cfg
    )
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    print_config()

    # set path to save model and log, and save config parameter as a log file
    save_path = os.path.join(cfg.SAVE_PATH, cfg.EXP_TAG, cfg.TIMESTAMP)
    if not os.path.exists(save_path):
        print("===> Workspace path is not built, will be created!")
        os.makedirs(save_path)
        print("===> Workspace has been built!")
    save_config(save_path)

    # get model
    model = None
    if cfg.MODEL_NAME == 'c3d':
        model = C3D.C3D(num_classes=cfg.DATASET.CLASS_NUM, train_type=cfg.TRAIN.TYPE)

    # get dataset, and build dataloader for training, validating and testing
    train_dataset = None
    valid_dataset = None
    if cfg.DATASET.NAME == "UCF-101-origin":
        dataset = ucf101.SSL_Dataset(root=os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.NAME), mode='train', args=args)
        val_size = cfg.DATASET.VAL_SIZE
        train_dataset, valid_dataset = random_split(dataset, (len(dataset) - val_size, val_size))
    # build data loader, using DataLoaderX or not
    train_loader = None
    valid_loader = None
    if cfg.DATASET.LOAD_TYPE == 'normal':
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  drop_last=True)
    elif cfg.DATASET.LOAD_TYPE == 'DataLoaderX':
        train_loader = ucf101.DataLoaderX(train_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.TRAIN.NUM_WORKERS,
                                          drop_last=True)
        valid_loader = ucf101.DataLoaderX(valid_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.TRAIN.NUM_WORKERS,
                                          drop_last=True)

    # set optimizer and scheduler
    model_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            model_params += [{'params': [value], 'lr':cfg.TRAIN.LEARNING_RATE}]
    optimizer = optim.SGD(model_params,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     min_lr=cfg.TRAIN.MIN_LR,
                                                     patience=cfg.TRAIN.PATIENCE,
                                                     factor=cfg.TRAIN.FACTOR)

    # load model to CUDA
    model = model.cuda()

    # using APEX to accelerate training
    if cfg.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # prepare other components, send data and model into CUDA
    if cfg.MULTI_GPU:
        model = nn.DataParallel(model)
    
    criterion_CE = nn.CrossEntropyLoss().cuda()

    # train and validate loop
    prev_best_val_loss = 100
    prev_best_loss_model_path = None
    prev_train_loss = 0
    prev_train_acc = 0
    prev_valid_loss = 0
    prev_valid_acc = 0
    for epoch in tqdm(range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.START_EPOCH+cfg.TRAIN.EPOCH)):
        prev_train_loss, prev_train_acc = train(train_loader, model, criterion_CE, optimizer, epoch, prev_train_loss, prev_train_acc, root_path=save_path)
        prev_valid_loss, prev_valid_acc, val_loss = validation(valid_loader, model, criterion_CE, optimizer, epoch, prev_valid_loss, prev_valid_acc, root_path=save_path)
        # save model if current model is better than previous
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(save_path, 'best_val_loss_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss
            if prev_best_loss_model_path:
                os.remove(prev_best_loss_model_path)
            prev_best_loss_model_path = model_path
        scheduler.step(val_loss)

        # save checkpoints
        if epoch % cfg.SHOW_INFO == 0:
            checkpoints = os.path.join(save_path, 'model_checkpoint_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), checkpoints)
            print('===> Checkpoint will be saved to: ', checkpoints)

        wandb.log({'val_loss_epoch': val_loss, 'epoch': epoch+1})


if __name__ == '__main__':
    seed = cfg.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main()