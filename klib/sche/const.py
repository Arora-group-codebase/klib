from .base import KScheduler

class KConstScheduler(KScheduler):
    def on_train_step_start(self, index_in_epoch, total_steps_in_epoch):
        pass

    @staticmethod
    def add_argparse_args(parser, var):
        pass