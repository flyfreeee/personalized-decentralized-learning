from torch.optim import Optimizer


class dlOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda)
        super(dlOptimizer, self).__init__(params, defaults)
    
    def step(self, aggregated_models):
        loss = None

        for group in self.param_groups:
            # for i in range(len(group['params'])):
            #     p = group['params'][i]
            #     grad_regularizer = 0
            #     for m in aggregated_models:
            #         grad_regularizer += p.data - m.data[i]
            #     p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * grad_regularizer)
            for p, m in zip(group['params'], aggregated_models):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (len(m)*p.data - m.sum()))
        return group['params'], loss
