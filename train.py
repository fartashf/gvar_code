from __future__ import print_function
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import logging
from log_utils import AverageMeter, Profiler
import time
# import nonacc_autograd
from optim.jvp import add_weight_noise, remove_weight_noise
from optim.sgd import optim_log  # , sma_update
from autograd.nonacc import nonacc_grad, acc_grad
from test import test


def train(tb_logger, epoch, train_loader, model, optimizer, opt, test_loader,
          save_checkpoint, train_test_loader):
    batch_time = AverageMeter()
    optimizer.profiler = Profiler()
    gluster = optimizer.gluster
    model.train()
    end = time.time()
    # update sampler weights
    if opt.sampler and epoch >= opt.sampler_start_epoch and not opt.scheduler:
        # optimizer.weights += 1e-5
        # optimizer.weights /= optimizer.weights.sum()
        optimizer.logger.reset()
        logger = optimizer.logger
        niters = optimizer.niters
        optimizer.weights = train_loader.sampler.update(optimizer)
        logging.info(str(logger))
        logger.tb_log(tb_logger, step=niters)
    epoch_iters = int(np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    optimizer.logger.reset()
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        if optimizer.niters == opt.epochs*epoch_iters:
            break
        model.train()  # SERIOUSLY ??? AGAIN ???!!!!
        if gluster:
            gluster.activate()
        optimizer.profiler.start()
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if opt.wnoise:
            add_weight_noise(optimizer, opt)
        optimizer.zero_grad()
        if gluster:
            gluster.zero()
        output = model(data)
        if opt.optim == 'dmom':
            optimizer.profiler.tic()
            loss = model.criterion(output, target, reduction='none')
            optimizer.profiler.toc('forward')
            grads = nonacc_grad(model, loss)
            optimizer.profiler.toc('backward')
            optimizer.step(idx, grads, loss, target)
        elif opt.optim == 'dmom_jvp' or opt.optim == 'dmom_ns':
            optimizer.profiler.tic()
            loss = model.criterion(output, target, reduction='none')
            # TODO: /batch_size
            optimizer.profiler.toc('forward')
            optimizer.step(idx, loss, target)
        elif opt.optim == 'ssgd':
            loss = model.criterion(output, target, reduction='none')
            grads = nonacc_grad(model, loss)
            # grads = nonacc_grad_backward(model, loss)
            acc_grad(model, grads)
            optimizer.step()
        elif opt.optim == 'adddmom':
            loss_i = model.criterion(output, target, reduction='none')
            loss = optimizer.step(idx, loss_i)
        elif opt.scheduler:
            scheduler = optimizer.scheduler
            optimizer.profiler.tic()
            loss = model.criterion(output, target, reduction='none')
            optimizer.profiler.toc('forward')
            alpha = loss.data.cpu().numpy()
            idxc = idx.cpu().numpy()
            scheduler.target[idx] = target.data.cpu().numpy()
            scheduler.update_alpha(idxc, alpha)
            optimizer.profiler.toc('alpha')
            weights = torch.Tensor(scheduler.weights[idx]/len(idx)).cuda()
            loss.backward(weights)
            optimizer.profiler.toc('backward')
            optim_log(optimizer, optimizer.logger)
            optimizer.step()
            # sma_update(optimizer, opt.sma_momentum)
            optimizer.profiler.toc('step')
        else:
            optimizer.profiler.tic()
            loss = model.criterion(output, target)
            optimizer.profiler.toc('forward')
            # grad = torch.ones_like(loss)
            # nonacc_autograd.backward([loss], [grad])
            loss.backward()
            optimizer.profiler.toc('backward')
            optim_log(optimizer, optimizer.logger)
            optimizer.step()
            # sma_update(optimizer, opt.sma_momentum)
            optimizer.profiler.toc('step')
        if gluster:
            gluster.em_update()
        if opt.wnoise:
            remove_weight_noise(optimizer, opt)
        optimizer.niters += 1
        if opt.scheduler:
            train_loader.sampler.scheduler.niters += 1
        optimizer.profiler.end()

        batch_time.update(time.time() - end)
        end = time.time()
        # optimizer.logger.update('bs', len(idx), 1)
        niters = optimizer.niters
        loss = loss.mean()
        if batch_idx % opt.log_interval == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss:.6f}\t'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                '{opt_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=batch_time,
                    opt_log=str(optimizer.logger)))
            if opt.log_profiler:
                logging.info(str(optimizer.profiler))
        if np.isnan(float(loss)):
            return -1
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            tb_logger.log_value('batch_time', batch_time.val,
                                step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)

        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, gluster, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            prec1 = test(tb_logger,
                         model, gluster, test_loader, opt, optimizer.niters)
            save_checkpoint(model, float(prec1), opt, optimizer)
            tb_logger.save_log()

    if opt.optim == 'dmom' or opt.optim == 'dmom_jvp':  # and epoch > 9:
        optimizer.logger.reset()
        optimizer.log_perc()
        logging.info(str(optimizer.logger))
        optimizer.logger.tb_log(tb_logger, step=niters)

    if opt.scheduler:
        scheduler = optimizer.scheduler
        # if opt.log_image:
        #     vis_samples(scheduler.alpha, train_loader.dataset, opt.epochs)

        scheduler.logger.reset()
        scheduler.log_perc()
        logging.info(str(scheduler.logger))
        scheduler.logger.tb_log(tb_logger, step=optimizer.niters)
