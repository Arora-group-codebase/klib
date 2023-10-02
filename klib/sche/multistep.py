from .base import KScheduler

class KMultistepScheduler(KScheduler):
    
    def __init__(self, optimizer, var, args):
        super().__init__(optimizer, var, args)

        if var == 'lr':
            if args.base_batch_size_for_lr != -1:
                scale = args.total_batch_size / args.base_batch_size_for_lr
            else:
                scale = 1
            rescaled_lr = args.lr * scale

            if args.warmup:
                if args.warmup_start_lr is None:
                    args.warmup_start_lr = args.lr
            else:
                args.warmup_epochs = 0
            
            self.warmup_end_val = rescaled_lr
            self.warmup_start_val = args.warmup_start_lr
            self.warmup_epochs = args.warmup_epochs

            self.decay_sche = args.lrdecay_sche

            if len(args.gamma) == 0:
                args.gamma = [0.1]
            if len(args.gamma) == 1:
                self.gamma = args.gamma * len(self.decay_sche)
                args.gamma = args.gamma[0]
            else:
                self.gamma = args.gamma
        else:
            if not getattr(args, f"{var}_warmup"):
                setattr(args, f"{var}_warmup_epochs", 0)
            
            self.warmup_end_val = getattr(args, var)
            self.warmup_start_val = getattr(args, f"warmup_start_{var}")
            self.warmup_epochs = getattr(args, f"{var}_warmup_epochs")

            self.decay_sche = getattr(args, f"{var}_decay_sche")

            if len(getattr(args, f"{var}_gamma")) == 0:
                setattr(args, f"{var}_gamma", [0.1])
            if len(getattr(args, f"{var}_gamma")) == 1:
                self.gamma = getattr(args, f"{var}_gamma") * len(self.decay_sche)
                setattr(args, f"{var}_gamma", getattr(args, f"{var}_gamma")[0])
            else:
                self.gamma = getattr(args, f"{var}_gamma")
    
    def on_train_step_start(self, index_in_epoch, total_steps_in_epoch):
        if self.epoch_finished < self.warmup_epochs:
            t = (self.epoch_finished + index_in_epoch / total_steps_in_epoch) / self.warmup_epochs
            self._adjust(self.warmup_start_val + (self.warmup_end_val - self.warmup_start_val) * t)
            
        if index_in_epoch == 0:
            if self.epoch_finished == self.warmup_epochs:
                self._adjust(self.warmup_end_val)
            for i, x in enumerate(self.decay_sche):
                if x == self.epoch:
                    self._decay(self.gamma[i])

    @staticmethod
    def add_argparse_args(parser, var):

        group = parser.add_argument_group('multistep scheduler')

        if var == 'lr':
            group.add_argument('--warmup', type=int)
            group.add_argument('--warmup-epochs', type=int)
            group.add_argument('--base-batch-size-for-lr', type=int)
            group.add_argument('--gamma', type=float, nargs='+', default=[])
            group.add_argument('--lrdecay-sche', nargs='+', type=int, default=[])
            group.add_argument('--warmup-start-lr', type=float)
        else:
            group.add_argument(f"--{var}-warmup", type=int)
            group.add_argument(f"--{var}-warmup-epochs", type=int)
            group.add_argument(f"--{var}-gamma", type=float, nargs='+', default=[])
            group.add_argument(f"--{var}-decay-sche", nargs='+', type=int, default=[])
            group.add_argument(f"--warmup-start-{var}", type=float)
