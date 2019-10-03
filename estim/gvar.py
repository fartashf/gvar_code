import torch
import torch.nn
import torch.multiprocessing

from estim.sgd import SGDEstimator
from estim.gluster import GlusterOnlineEstimator, GlusterBatchEstimator
from estim.svrg import SVRGEstimator
# from estim.ntk import NTKEstimator
from estim.nuq import NUQEstimator
from estim.nuq import NUQEstimatorSingleGPUParallel
from estim.nuq import NUQEstimatorMultiGPUParallel


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
        elif opt.g_estim == 'ntk':
            gest = NTKEstimator(data_loader, opt, tb_logger)
        elif opt.g_estim == 'nuq':
            if opt.nuq_parallel == 'no':
                gest = NUQEstimator(data_loader, opt, tb_logger)
            elif opt.nuq_parallel == 'gpu1':
                gest = NUQEstimatorSingleGPUParallel(
                    data_loader, opt, tb_logger)
            else:
                gest = NUQEstimatorMultiGPUParallel(
                    data_loader, opt, tb_logger)
        self.sgd = sgd
        self.gest = gest
        self.opt = opt
        self.tb_logger = tb_logger
        self.init_snapshot = False
        self.gest_used = False
        self.gest_counter = 0
        self.Esgd = 0
        self.last_log_iter = 0
        self.opt = opt

    def is_log_iter(self, niters):
        opt = self.opt
        if (niters-self.last_log_iter >= opt.gvar_log_iter
                and niters >= opt.gvar_start):
            self.last_log_iter = niters
            return True
        return False

    def snap_batch(self, model, niters):
        # model.eval()  # done inside SVRG
        model.train()
        # Cosine sim
        # gviter = self.opt.gvar_estim_iter
        # self.Esgd = self.sgd.get_Ege_var(model, gviter)[0]
        self.gest.snap_batch(model, niters)
        self.init_snapshot = True
        self.gest_counter = 0

    def snap_online(self, model, niters):
        # model.eval()  # TODO: keep train
        model.train()
        self.gest.snap_online(model, niters)

    def snap_model(self, model):
        self.gest.snap_model(model)

    def log_var(self, model, niters):
        tb_logger = self.tb_logger
        # model.eval()
        if not self.init_snapshot:
            return ''
        gviter = self.opt.gvar_estim_iter
        Ege, var_e, snr_e, nv_e = self.gest.get_Ege_var(model, gviter)
        Esgd, var_s, snr_s, nv_s = self.sgd.get_Ege_var(model, gviter)
        bias = torch.mean(torch.cat(
            [(ee-gg).abs().flatten() for ee, gg in zip(Ege, Esgd)]))
        # print(np.sum([ee.pow(2).sum().item() for ee in Esgd]))
        # print(np.sum([gg.pow(2).sum().item() for gg in Ege]))
        tb_logger.log_value('grad_bias', float(bias), step=niters)
        tb_logger.log_value('est_var', float(var_e), step=niters)
        tb_logger.log_value('sgd_var', float(var_s), step=niters)
        tb_logger.log_value('est_snr', float(snr_e), step=niters)
        tb_logger.log_value('sgd_snr', float(snr_s), step=niters)
        tb_logger.log_value('est_nvar', float(nv_e), step=niters)
        tb_logger.log_value('sgd_nvar', float(nv_s), step=niters)
        sgd_x, est_x = ('', '[X]') if self.gest_used else ('[X]', '')
        # neo = torch.sqrt(torch.sum(torch.cat([(ee*ee).flatten() for ee in
        #                  self.Esgd])))
        # nen = torch.sqrt(torch.sum(torch.cat([(ee*ee).flatten() for ee in
        #                  Esgd])))
        # eed = torch.sum(torch.cat(
        #     [(ee/neo*gg/nen).flatten() for ee, gg in zip(self.Esgd, Esgd)]))
        # return ('G Bias: %.8f\t'
        #         '%sSGD Var: %.8f\t %sEst Var: %.8f\t'
        #         'SGD SNR: %.8f\t Est SNR: %.8f\t'
        #         # 'Esgd cos sim: %.8f' % (
        #         # bias, sgd_x, var_s, est_x, var_e, snr_s, snr_e, eed))
        #         # % (bias, sgd_x, var_s, est_x, var_e))
        #         % (bias, sgd_x, var_s, est_x, var_e, snr_s, snr_e))
        return ('G Bias: %.8f\t'
                '%sSGD Var: %.8f\t %sEst Var: %.8f\t'
                'SGD N-Var: %.8f\t Est N-Var: %.8f\t'
                % (bias, sgd_x, var_s, est_x, var_e, nv_s, nv_e))

    def grad(self, niters):
        model = self.model
        model.train()
        use_sgd = self.use_sgd(niters)
        if use_sgd:
            self.gest_used = False
            return self.sgd.grad(model, in_place=True)
        self.gest_used = True
        self.gest_counter += 1
        return self.gest.grad(model, in_place=True)

    def use_sgd(self, niters):
        use_sgd = not self.opt.g_optim or niters < self.opt.g_optim_start
        use_sgd = use_sgd or (self.opt.g_optim_max > 0 and
                              self.gest_counter >= self.opt.g_optim_max)
        return use_sgd

    def state_dict(self):
        return self.gest.state_dict()

    def load_state_dict(self, state):
        self.gest.load_state_dict(state)
        self.init_snapshot = True
