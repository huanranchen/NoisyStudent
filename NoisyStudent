import os
import numpy as np
import torchvision
from torchvision import models
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F


def smoothing_cross_entropy(x, y):
    '''
    -y log pre
    :param x: N, D
    :param y: N, D
    :return:
    '''
    if x.shape != y.shape:
        return F.cross_entropy(x, y)
    return F.kl_div(F.log_softmax(x, dim=1), y)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet50(num_classes=60, )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.apply(self._init_weights)
        print('resnet' * 100)

    def forward(self, x):
        x = self.model(x)
        return x

    def load_model(self):
        start_state = torch.load('model.ckpt', map_location=self.device)
        self.model.load_state_dict(start_state['model_state'])
        print('using loaded model')
        print('-' * 100)

    def save_model(self):
        result = {}
        result['model_state'] = self.model.state_dict()
        torch.save(result, 'model.ckpt')


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(now_step):
        if now_step < num_warmup_steps:  # 小于的话 学习率比单调递增
            return float(now_step) / float(max(1, num_warmup_steps))
        # 大于的话，换一个比例继续调整学习率
        progress = float(now_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class NoisyStudent():
    def __init__(self,
                 batch_size=32,
                 lr=4e-3,
                 weight_decay=0.05, ):
        self.result = {}
        from data.data import get_loader, get_test_loader
        self.train_loader = get_loader(batch_size=batch_size,
                                       valid_category=None,
                                       train_image_path='./public_dg_0416/train/',
                                       valid_image_path='./public_dg_0416/train/',
                                       label2id_path='./dg_label_id_mapping.json')
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path='./dg_label_id_mapping.json')
        self.test_loader_student, self.label2id = get_test_loader(batch_size=batch_size,
                                                                  transforms='train',
                                                                  label2id_path='./dg_label_id_mapping.json')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if os.path.exists('model.ckpt'):
            self.model.load_model()

    def save_result(self):
        from data.data import write_result
        result = {}
        for name, pre in list(self.result.items()):
            _, result[name] = torch.max(pre, dim = 1).item()

        write_result(result)


    def predict(self):
        with torch.no_grad():
            print('teacher are giving his predictions!')
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.to(self.device)
                x = self.model(x)
                x = F.softmax(x, dim=1)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D

            print('teacher have given his predictions!')
            print('-' * 100)

    def get_label(self, names):
        y = []
        for name in list(names):
            y.append(self.result[name])

        return torch.cat(y, dim=0).to(self.device)

    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16_training=True,
              warmup_epoch=1,
              warmup_cycle=12000,
              ):
        # scheduler = get_cosine_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=len(train_loader)*warmup_epoch,
        #                                             num_training_steps=total_epoch*len(train_loader),
        #                                             num_cycles=warmup_cycle)
        criterion = smoothing_cross_entropy
        for epoch in range(1, total_epoch + 1):
            # first, predict
            self.model.eval()
            # self.predict()

            # train
            self.model.train()
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)

            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                x = self.model(x)  # N, 60
                _, pre = torch.max(x, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                loss = criterion(x, y)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                step += 1
                # scheduler.step()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            print(f'epoch {epoch}, train loader loss = {train_loss}, acc = {train_acc}')

            #############################################################################################

            self.model.train()
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.test_loader_student)
            for x, y in pbar:
                x = x.to(self.device)
                y = self.get_label(y)
                x = self.model(x)  # N, 60
                _, pre = torch.max(x, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                loss = criterion(x, y)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                step += 1
                # scheduler.step()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')

            self.model.save_model()


if __name__ == '__main__':
    x = NoisyStudent()
    x.train(total_epoch=1)
