from torch import optim

class KScheduler:
    
    optimizer: optim.Optimizer
    step_ctr: int = 0
    epoch: int = 0

    def __init__(self, optimizer):
        self.optimizer = optimizer
    

    def resume(self, epoch, step_ctr):
        self.epoch = epoch
        self.step_ctr = step_ctr
    

    def on_train_step_start(self, index_in_epoch):
        raise NotImplementedError()
    
    
    def log(self, type, log):
        for gid, g in enumerate(self.optimizer.param_groups):
            suf = '' if len(self.optimizer.param_groups) == 1 else f'/{gid}'
            log[f"{type}/lr{suf}"] = g['lr']
            log[f"{type}/wd{suf}"] = g['weight_decay']
    

    def on_train_step_end(self, index_in_epoch):
        self.step_ctr += 1
        if index_in_epoch == 0:
            self.epoch += 1

    
    def _adjust_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    

    def _decay_lr(self, gamma):
        for g in self.optimizer.param_groups:
            g['lr'] *= gamma


class KConstScheduler(KScheduler):
    def on_train_step_start(self, index_in_epoch):
        pass


class KMultiStepScheduler(KScheduler):
    
    def __init__(self, optimizer, *, warmup_sche, lrdecay_sche, gamma=0.1):
        super().__init__(optimizer)
        
        self.warmup_sche = warmup_sche
        self.lrdecay_sche = lrdecay_sche
        self.gamma = gamma
    
    def on_train_step_start(self, index_in_epoch):
        if self.step_ctr < len(self.warmup_sche):
            self._adjust_lr(self.warmup_sche[self.step_ctr])

        if index_in_epoch == 0:
            if self.epoch in self.lrdecay_sche:
                self._decay_lr(self.gamma)
