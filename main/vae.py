from __future__ import print_function
import numpy as np
import logging
import os
import sys

import torch
import torch.nn
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.multiprocessing

import utils
from data import get_loaders
from log_utils import TBXWrapper
from log_utils import Profiler
from args import get_opt
from estim.optim import OptimizerFactory
tb_logger = TBXWrapper()
torch.multiprocessing.set_sharing_strategy('file_system')


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none')
    BCE = BCE.sum(-1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)

    return BCE + KLD


def vae_loss(model, data, reduction='mean', weights=1):
    data = data[0].cuda()
    model.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar)*weights
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def test(tb_logger, model, test_loader,
         opt, niters, set_name='Test', prefix='V'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_loss += vae_loss(model, data, reduction='sum')

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def train(tb_logger, epoch, train_loader, model, optimizer, opt, test_loader,
          save_checkpoint, train_test_loader):
    batch_time = Profiler()
    model.train()
    profiler = Profiler()
    epoch_iters = int(np.ceil(1. * len(train_loader.dataset) / opt.batch_size))
    optimizer.logger.reset()
    for batch_idx in range(opt.epoch_iters):
        profiler.start()
        # sgd step
        loss = optimizer.step(profiler)

        batch_time.toc('Time')
        batch_time.end()
        optimizer.niters += 1
        niters = optimizer.niters

        # if True:
        if batch_idx % opt.log_interval == 0:
            gvar_log = ''
            prof_log = ''
            if (batch_idx % opt.gvar_log_iter == 0
                    and optimizer.niters >= opt.gvar_start):
                gvar_log = '\t' + optimizer.gvar.log_var(model, niters)
            if opt.log_profiler:
                prof_log = '\t' + str(profiler)

            logging.info(
                'Epoch: [{0}][{1}/{2}]({niters})\t'
                'Loss: {loss:.6f}\t'
                '{batch_time}\t'
                '{opt_log}{gvar_log}{prof_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=str(batch_time),
                    opt_log=str(optimizer.logger),
                    gvar_log=gvar_log,
                    prof_log=prof_log,
                    niters=niters))
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            test(tb_logger, model, test_loader, opt, optimizer.niters)
            # save_checkpoint(model, float(prec1), opt, optimizer,
            #                 gvar=optimizer.gvar)
            tb_logger.save_log()


def main():
    opt = get_opt()
    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)
    logfname = os.path.join(opt.logger_name, 'log.txt')
    logging.basicConfig(
        filename=logfname,
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        # TODO: remove deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    opt.no_transform = True
    train_loader, test_loader, train_test_loader = get_loaders(opt)

    if opt.epoch_iters == 0:
        opt.epoch_iters = int(
            np.ceil(1. * len(train_loader.dataset) / opt.batch_size))
    opt.maxiter = opt.epoch_iters * opt.epochs
    if opt.g_epoch:
        opt.gvar_start *= opt.epoch_iters
        opt.g_bsnap_iter *= opt.epoch_iters
        opt.g_optim_start = (opt.g_optim_start * opt.epoch_iters) + 1
        opt.g_reinit_iter = opt.g_reinit_iter * opt.epoch_iters
    opt.g_reinit_iter = int(opt.g_reinit_iter)

    model = VAE().cuda()
    model.criterion = vae_loss

    optimizer = OptimizerFactory(model, train_loader, tb_logger, opt)
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    model_path = os.path.join(opt.resume, opt.ckpt_name)
    if opt.resume != '':
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            best_prec1 = checkpoint['best_prec1']
            if opt.g_resume:
                optimizer.gvar.load_state_dict(checkpoint['gvar'])
            else:
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model'])
                save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(model_path, epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    if opt.niters > 0:
        max_iters = opt.niters
    else:
        max_iters = opt.epochs * opt.epoch_iters
    while optimizer.niters < max_iters:
        optimizer.epoch = epoch
        utils.adjust_lr(optimizer, opt)
        ecode = train(
            tb_logger,
            epoch, train_loader, model, optimizer, opt, test_loader,
            save_checkpoint, train_test_loader)
        if ecode == -1:
            break
        epoch += 1
    tb_logger.save_log()


if __name__ == '__main__':
    main()
