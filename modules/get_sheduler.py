from torch.optim import lr_scheduler

def get(optimizer, sheduler_type, cycles):
    if sheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cycles, gamma=0.1)
    elif sheduler_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cycles, eta_min=0)
    return scheduler