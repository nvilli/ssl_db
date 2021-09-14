from config import ActivityConfig as cfg
import argparse
import torch
import os
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_config():
    # print(cfg)
    for key, value in cfg.items():
        print(str(key) + ": " + str(value))


def save_config(root_path):
    conf_log_path = os.path.join(root_path, "conf.log")
    conf_log = open(conf_log_path, "a")
    for key, value in cfg.items():
        s_str = str(key) + ": " + str(value) + '\n'
        conf_log.write(s_str)
    conf_log.close()
    print("===> Config log save success!")


def parse_args():
    parser = argparse.ArgumentParser(description='SSL_learning')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
    parser.add_argument('--gpus', type=int, default=1, help='gpus that can be used')
    args = parser.parse_args()
    return args


def show_info(epoch, step, loader_len, batch_time, data_time, losses, acc, log_type, root_path):
    log_file_name = cfg.TIMESTAMP + '.log'
    log_file_path = os.path.join(root_path, log_file_name)
    file = open(log_file_path, 'a')

    p_str = "-------------------------------------------------------------------"
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> " + str(log_type) + " Epoch:[{0}][{1}/{2}]".format(epoch, step + 1, loader_len)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> " + str(log_type) + " Data time:{data_time:.3f}  Batch time:{batch_time:.3f}".format(data_time=data_time.val,
                                                                                                       batch_time=batch_time.val)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> " + str(log_type) + " Loss:{loss:.5f}".format(loss=losses.avg)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> " + str(log_type) + " Accuracy:{acc:.3f}".format(acc=acc.avg)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    file.close()
