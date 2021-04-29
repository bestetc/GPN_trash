from torch.optim import lr_scheduler

def get(optimizer, sheduler_type, **kwargs):
    ''' Return learning rate sheduler.
    
    Sheduler_type should be str: 'step' or 'cos'.
    'step' for step decay learning rate sheduler.
    'cos' for cosine learning rate decay with warm restart.
    
    **kwargs:
    step_len: 
    quantity of epoch between learning rate decay at 10 times.
    Used with sheduler_type = 'step' only.
    
    cycle_len: 
    quantity of epoch till the learning rate decay from initial to zero.
    Used with sheduler_type = 'cos' only.
    
    If you want to use cosine learning rate decay without warm restart set cycle_len=num_epoch
    
    '''

    if sheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=kwargs['step_len'], gamma=0.1)
    elif sheduler_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=kwargs['cycle_len'], eta_min=0)
    elif sheduler_type == 'warmup':
        scheduler = lr_scheduler.CyclicLR(
                        optimizer, 
                        base_lr=kwargs['learning_rate'] / (kwargs['batch_per_epoch'] * kwargs['warmup_epoch']), 
                        max_lr=kwargs['learning_rate'],
                        step_size_up=((kwargs['batch_per_epoch'] + 1) * kwargs['warmup_epoch']),
                        step_size_down=0,
                        cycle_momentum=False
                        )
    return scheduler
