import torch
from klib.trainer import BaseTrainer
from klib.kfreqctrl import KFreqController
from .base import Hooker
import numpy as np
import copy

class DescentHooker(Hooker):
    """
    Compute E[L(theta_{t+1})] - L(theta_t)
    WARNING: It may change the batch stats of BN
    """
    
    def __init__(self, trainer: BaseTrainer) -> None:
        self.freq_ctrl = KFreqController(
            trainer.args.descent_hooker_freq,
            factor=trainer.args.descent_hooker_freq_factor,
            cont=trainer.args.descent_hooker_freq_cont
        )
        self.n_samples = trainer.args.descent_hooker_n_samples
        self.loss_eval_per_sample = trainer.args.descent_hooker_loss_per_sample

        if trainer.world_size > 1:
            raise NotImplementedError()
        if trainer.args.autocast_dtype != 'float32':
            raise NotImplementedError()
        if trainer.n_grad_accumu > 1:
            raise NotImplementedError()

    
    def extra_evaluate(self, trainer: BaseTrainer):
        if self.freq_ctrl.ok():
            it = iter(trainer.train_dataloader.enum(self.n_samples * (1 + self.loss_eval_per_sample)))

            trainer.kmodel.model.train()

            orig_model = copy.deepcopy(trainer.kmodel.model)
            orig_opt_state = copy.deepcopy(trainer.optimizer.state_dict())

            cur_loss = []
            next_loss = []

            total_grad = {id(p): torch.zeros_like(p) for p in trainer.kmodel.trainable_params()}
            total_grad2 = {id(p): torch.zeros_like(p) for p in trainer.kmodel.trainable_params()}

            for i in range(self.n_samples):
                _, (inputs_g, targets_g) = next(it)
                
                if trainer.args.descent_hooker_lr is not None:
                    for g in trainer.optimizer.param_groups:
                        g['lr'] = trainer.args.descent_hooker_lr

                trainer.optimizer.zero_grad()
                
                outputs_g = trainer.kmodel.model(inputs_g)
                loss_g = trainer.kmodel.criterion(outputs_g, targets_g)
                loss_g.backward()

                for p in trainer.kmodel.trainable_params():
                    total_grad[id(p)] += p.grad
                    total_grad2[id(p)] += p.grad ** 2

                trainer.optimizer.step()

                with torch.no_grad():
                    for j in range(self.loss_eval_per_sample):
                        _, (inputs_l , targets_l) = next(it)

                        cur_outputs = orig_model(inputs_l)
                        cur_loss.append(trainer.kmodel.criterion(cur_outputs, targets_l).item())

                        next_outputs = trainer.kmodel.model(inputs_l)
                        next_loss.append(trainer.kmodel.criterion(next_outputs, targets_l).item())
                        
                        trainer.debug_log(f"descent: cur loss {cur_loss}, next loss {next_loss}")

                trainer.kmodel.model.load_state_dict(orig_model.state_dict())
                trainer.optimizer.load_state_dict(orig_opt_state)
            
            try:
                next(it)
            except StopIteration:
                pass
            
            cur_loss = np.array(cur_loss)
            next_loss = np.array(next_loss)
            delta = next_loss - cur_loss

            grad_norm2 = sum((((g / self.n_samples) ** 2).sum() for _, g in total_grad.items())).item()
            grad2_sum = sum(((g2 / self.n_samples).sum() for _, g2 in total_grad2.items())).item()
            
            trainer.step_log['descent/avg_grad/norm2'] = grad_norm2
            trainer.step_log['descent/avg_grad/norm'] = grad_norm2 ** 0.5
            trainer.step_log['descent/avg_grad_norm2'] = grad2_sum
            trainer.step_log['descent/grad_noise/trace'] = grad2_sum - grad_norm2

            trainer.step_log['descent/cur_loss/mean'] = cur_loss.mean()
            trainer.step_log['descent/cur_loss/std'] = cur_loss.std()
            trainer.step_log['descent/next_loss/mean'] = next_loss.mean()
            trainer.step_log['descent/next_loss/std'] = next_loss.std()
            trainer.step_log['descent/delta/mean'] = delta.mean()
            trainer.step_log['descent/delta/std'] = delta.std()

    
    def update_freq(self, trainer: BaseTrainer):
        self.freq_ctrl.step()
    

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group('descent hooker')
        group.add_argument('--descent-hooker-freq', type=int)
        group.add_argument('--descent-hooker-lr', type=float)
        group.add_argument('--descent-hooker-freq-factor', type=int, default=1)
        group.add_argument('--descent-hooker-freq-cont', type=int, default=10)
        group.add_argument('--descent-hooker-n-samples', type=int, default=500)
        group.add_argument('--descent-hooker-loss-per-sample', type=int, default=1)
