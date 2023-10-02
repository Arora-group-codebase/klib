from torch import optim


class KScheduler:
    
    VAR_DICT = {
        'wd': 'weight_decay'
    }
    
    optimizer: optim.Optimizer
    step_ctr: int
    epoch: int          # epoch ID
    epoch_finished: int # number of finished epoch

    def __init__(self, optimizer, var, args):
        self.optimizer = optimizer
        self.var = var
        self.step_ctr = 0
        self.epoch = 0
        self.epoch_finished = 0
        if self.var in self.VAR_DICT:
            self.varkey = self.VAR_DICT[self.var]
        else:
            self.varkey = self.var
    

    def resume(self, epoch, step_ctr):
        self.epoch = epoch
        self.step_ctr = step_ctr
    

    def on_train_step_start(self, index_in_epoch, total_steps_in_epoch):
        raise NotImplementedError()
    
    
    def log(self, type, log):
        for gid, g in enumerate(self.optimizer.param_groups):
            suf = '' if len(self.optimizer.param_groups) == 1 else f'/{gid}'
            log[f"{type}/{self.var}{suf}"] = g[self.varkey]
    

    def on_train_step_end(self, index_in_epoch, total_steps_in_epoch):
        self.step_ctr += 1
        if index_in_epoch == 0:
            self.epoch += 1
        if index_in_epoch == total_steps_in_epoch - 1:
            self.epoch_finished += 1

    
    def _adjust(self, val):
        for g in self.optimizer.param_groups:
            g[self.varkey] = val
    

    def _decay(self, gamma):
        for g in self.optimizer.param_groups:
            g[self.varkey] *= gamma

    @staticmethod
    def add_argparse_args(parser, var):
        raise NotImplementedError()
