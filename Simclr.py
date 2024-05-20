import logging
import os
import sys

import numpy as np
from itertools import combinations
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter:
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

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.model = kwargs['model'].to(device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        
        self.n_views = 2
        self.fp16_precision = False
        self.temperature = 0.1
        self.epochs = 40
        self.warmup_epochs = 30


        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        print(f"Logging has been saved at {self.writer.log_dir}.")
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.n_views * batch_size, self.n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        return logits, labels

    def contrastive_loss(self, features, patientIds):
        pids = np.array(patientIds, dtype=object)
        pid1, pid2 = np.meshgrid(pids, pids)
        pid_matrix = pid1 + '-' + pid2
        pids_of_interest = np.unique(pids+'-'+pids)
        bool_matrix_of_interest = np.isin(pid_matrix, pids_of_interest).astype(int)

        rows1, cols1 = np.where(np.triu(bool_matrix_of_interest, k=1))
        rows2, cols2 = np.where(np.tril(bool_matrix_of_interest, k=-1))

        nviews = set(range(features.shape[1]))
        view_combinations = combinations(nviews, 2)

        loss = 0
        ncombinations = 0

        for lead1, lead2 in view_combinations:
            view1_array = features[:, lead1, :]
            view2_array = features[:, lead2, :]

            norm1_vector = view1_array.norm(dim=1, keepdim=True)
            norm2_vector = view2_array.norm(dim=1, keepdim=True)

            sim_matrix = torch.mm(view1_array, view2_array.t())
            norm_matrix = torch.mm(norm1_vector, norm2_vector.t())
            
            temperature=0.1
            argument = sim_matrix / (norm_matrix * temperature)
            sim_matrix_exp = torch.exp(argument)

            triu_elements = sim_matrix_exp[rows1, cols1]
            tril_elements = sim_matrix_exp[rows2, cols2]
            diag_elements = torch.diag(sim_matrix_exp)
            
            triu_sum = torch.sum(sim_matrix_exp, dim=1)
            tril_sum = torch.sum(sim_matrix_exp, dim=0)

            loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))

            loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[cols2]))

            loss += loss_diag1 + loss_diag2
            loss_terms = 2

            if len(rows1) > 0:
                loss += loss_triu
                loss_terms += 1
            if len(rows2) > 0:
                loss += loss_tril
                loss_terms += 1
            ncombinations += 1

        loss = loss/(loss_terms*ncombinations)
        return loss, sim_matrix_exp, torch.diag(torch.ones_like(sim_matrix_exp))


    def train(self, train_loader):

        scaler = GradScaler(enabled=self.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        logging.info(f"Start SimCLR training for {self.epochs} epochs.")
        logging.info(f"Training with initial learning rate lr={self.lr} and batch size={self.batch_size} and warmup epochs={self.warmup_epochs}.")

        for epoch_counter in range(self.epochs):
            print(f"Epoch {epoch_counter}")
            for images, patientIds in tqdm(train_loader):
                
                images1 = images[0].to(device)
                images2 = images[1].to(device)

                with autocast(enabled=self.fp16_precision):
                    features1 = self.model(images1)
                    features2 = self.model(images2)
                    features = torch.cat([features1, features2], dim=1)
                    features = torch.nn.functional.normalize(features, p=2, dim=2)
                    loss, logits, labels = self.contrastive_loss(features, patientIds)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                loss_meter.update(loss.item(), images1.size(0))
                if n_iter % 1 == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    acc1_meter.update(top1[0], images1.size(0))
                    acc5_meter.update(top5[0], images1.size(0))
                    self.writer.add_scalar('loss', loss_meter.avg, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', acc1_meter.avg, global_step=n_iter)
                    self.writer.add_scalar('acc/top5', acc5_meter.avg, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= self.warmup_epochs:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}")

            # save model checkpoints
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': self.epochs,
                # 'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        logging.info("Training has finished.")
