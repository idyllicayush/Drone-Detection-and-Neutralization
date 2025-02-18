import torch

class MASOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, lambda_a=0.5, lambda_s=0.5, 
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, momentum=0.9):
        """
        Mixing ADAM and SGD (MAS) Optimizer
        
        :param params: Model parameters
        :param lr: Learning rate
        :param lambda_a: Weight for ADAM contribution
        :param lambda_s: Weight for SGD contribution
        :param beta1: ADAM beta1 parameter (momentum for first moment estimate)
        :param beta2: ADAM beta2 parameter (momentum for second moment estimate)
        :param eps: ADAM epsilon to avoid division by zero
        :param weight_decay: Weight decay for regularization
        :param momentum: SGD momentum parameter
        """
        defaults = dict(lr=lr, lambda_a=lambda_a, lambda_s=lambda_s,
                        beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
                        momentum=momentum)
        super(MASOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambda_a = group['lambda_a']
            lambda_s = group['lambda_s']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # ADAM first moment
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # ADAM second moment
                    state['velocity'] = torch.zeros_like(p.data)  # SGD velocity

                # Update step count
                state['step'] += 1
                step = state['step']

                # ADAM update
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                adapted_lr_adam = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                adam_update = adapted_lr_adam * exp_avg / denom

                # SGD update
                velocity = state['velocity']
                velocity.mul_(momentum).add_(grad)
                sgd_update = lr * velocity

                # MAS update (combining ADAM and SGD)
                combined_update = lambda_s * sgd_update + lambda_a * adam_update
                p.data.add_(-combined_update)

        return loss
