import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

def multistep_restart(T_max, epoch):
    epoch = float(epoch)
    restart_period = T_max
    while epoch/restart_period > 1.0:
        epoch = epoch - restart_period

    if epoch < 200:
        return 1
    if epoch < 400:
        return 0.5
    if epoch < 600:
        return 0.25
    if epoch < 800:
        return 0.125
    return 0.125*0.5

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.type == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.SGD.momentum}
    elif args.type == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.Adam.beta1, args.Adam.beta2),
            'eps': args.Adam.epsilon
        }
    elif args.type == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.RMSprop.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    if args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        print(milestones)
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
        
    if args.decay_type == 'restart':
        scheduler = lrs.LambdaLR(my_optimizer, lambda epoch: multistep_restart(args.period, epoch))

    return scheduler