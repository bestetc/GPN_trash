from torch.optim import SGD, Adam, AdamW

def get(model, optimizer_type, learning_rate):
    ''' Return optimizer '''
    if optimizer_type == 'SGD':
        getting_optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_type == 'Adam':
        getting_optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0001, amsgrad=False)
    elif optimizer_type == 'AdamW':
        getting_optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0001, amsgrad=False)
    
    return getting_optimizer
