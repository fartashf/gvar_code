import torch
import torch.nn
import torch.multiprocessing

from estim.sgd import SGDEstimator
from estim.gluster import GlusterOnlineEstimator, GlusterBatchEstimator
from estim.svrg import SVRGEstimator


class MinVarianceGradient(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.model = model
        sgd = SGDEstimator(data_loader, opt, tb_logger)
        # gluster or SVRG
        if opt.g_estim == 'gluster':
            if opt.g_online:
                gest = GlusterOnlineEstimator(
                        data_loader, opt, tb_logger)
            else:
                gest = GlusterBatchEstimator(
                        data_loader, opt, tb_logger)
        elif opt.g_estim == 'svrg':
            gest = SVRGEstimator(data_loader, opt, tb_logger)
        elif opt.g_estim == 'sgd':
            gest = SGDEstimator(data_loader, opt, tb_logger)
        self.sgd = sgd
        self.gest = gest
        self.opt = opt
        self.tb_logger = tb_logger
        self.init_snapshot = False
        self.gest_used = False

    def snap_batch(self, model, niters):
        model.eval()
        # model.train() # TODO: SVRG might have trouble with dropout
        self.gest.snap_batch(model, niters)
        self.init_snapshot = True

    def snap_online(self, model, niters):
        model.eval()  # TODO: keep train
        self.gest.snap_online(model, niters)

    def log_var(self, model, niters):
        tb_logger = self.tb_logger
        model.eval()
        if not self.init_snapshot:
            return ''
        gviter = self.opt.gvar_estim_iter
        Ege, var_e, snr_e = self.gest.get_Ege_var(model, gviter)
        # TODO: make this hyperparam
        Esgd, var_s, snr_s = self.sgd.get_Ege_var(model, gviter)
        bias = torch.mean(torch.cat(
            [(ee-gg).abs().flatten() for ee, gg in zip(Ege, Esgd)]))
        # print(np.sum([ee.pow(2).sum().item() for ee in Esgd]))
        # print(np.sum([gg.pow(2).sum().item() for gg in Ege]))
        tb_logger.log_value('grad_bias', float(bias), step=niters)
        tb_logger.log_value('est_var', float(var_e), step=niters)
        tb_logger.log_value('sgd_var', float(var_s), step=niters)
        tb_logger.log_value('est_snr', float(snr_e), step=niters)
        tb_logger.log_value('sgd_snr', float(snr_s), step=niters)
        sgd_x, est_x = ('', '[X]') if self.gest_used else ('[X]', '')
        return ('G Bias: %.8f\t'
                '%sSGD Var: %.8f\t %sEst Var: %.8f\t'
                'SGD SNR: %.8f\t Est SNR: %.8f' % (
                    bias, sgd_x, var_s, est_x, var_e, snr_s, snr_e))

    def grad(self, niters):
        model = self.model
        model.train()
        if not self.opt.g_optim or niters < self.opt.g_optim_start:
            self.gest_used = False
            return self.sgd.grad(model, in_place=True)
        self.gest_used = True
        return self.gest.grad(model, in_place=True)
