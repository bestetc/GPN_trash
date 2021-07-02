""" Module contain some usefull tools for neural networks.
    
    Custom metrics: SmoothCrossEntropyLoss class
    Optimizers: get_optimizer function
    Shedulers: get_scheduler function
    Metrics: true_accuracy and approx_accuracy functions
    Training loop simplify: make_step function

"""

import math
import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim import lr_scheduler
from torch import no_grad
from torch.utils.data import Subset, DataLoader
from numpy.random import randint

from .custom_optimizers import RAdam, RangerRAdam

class SmoothCrossEntropyLoss(_WeightedLoss):
    """ Calculate CrossEntropyLoss with smoothing labels.
    
    Parameters
    ----------
    weight: torch.Tensor, optional 
        Manual rescaling weight given to each class.
        If given, has to be a Torch.Tensor with shape [class_qty]
    reduction: str, optional
        Specifies the reduction to apply to the output.
        'mean': the mean of the output is taken.
        'sum': the output will be summed
        'none': no reduction applied.
    smoothing: float, optional
        change the construction of true probability to 1 - smoothing.
    
    See Also
    --------
    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks.
    Part 5.2
        
    """   
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = torch.nn.functional.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))
    

def get_optimizer(model, optimizer_type, learning_rate, **kwargs):
    """ Return optimizer 
    
    Realize three type of optimizer.
        
    Parameters
    ----------   
    model: nn.Module
        Neural network model.
    optimizer_type: str
        Define optimizer type.
    learning_rate: float
        Initial learning rate.
    kwargs: dict, optional
        optimizer specific parameters
    
    Returns
    -------
    optimizer: torch.optim
            
    """
    if optimizer_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type == 'Ranger':
        optimizer = RangerRAdam(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError('Unknown optimizer_type value')
    
    return optimizer

def get_scheduler(optimizer, scheduler_type, **kwargs):
    """ Return learning rate scheduler.
    
    Realize three type of scheduler.
    
    Parameters
    ----------
    optimizer: torch.optim
        optimizer picked for training
    scheduler_type: str
        define scheduler type
        'step' - decrease learning rate in 10 time step by step.
        'cos' - decrease learning rate using a cosine annealing schedule.
        'warmup' - increase learning rate from zero to initial.
    **kwargs : dict,
        learning_rate: float
            Initial learning rate.
        step_len: int
            Quantity of epochs between learning rate decay at 10 times. 
            Use with 'step' scheduler type only.
        cycle_len: int
            Quantity of epochs till the learning rate decay from initial to zero.
            Use with 'step' scheduler type only.
        batch_per_epoch: int
            Quantity batches in datasets.
        warmup_epoch: int
            Quantity epochs to rise learning rate from zero to initial.
        
    Returns
    -------
    scheduler: torch.optim.lr_scheduler
    
    See Also
    --------
    torch.optim.lr_scheduler.StepLR
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    torch.optim.lr_scheduler.CyclicLR
    
    """
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=kwargs['step_size'], gamma=0.1)
    elif scheduler_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=kwargs['cycle_len'], eta_min=0)
    elif scheduler_type == 'warmup':
        scheduler = lr_scheduler.CyclicLR(
                        optimizer, 
                        base_lr=kwargs['learning_rate'] / (kwargs['batch_per_epoch'] * kwargs['warmup_epoch']), 
                        max_lr=kwargs['learning_rate'],
                        step_size_up=(kwargs['batch_per_epoch'] + 1) * kwargs['warmup_epoch'],
                        step_size_down=0,
                        cycle_momentum=False
                        )
    return scheduler

def true_accuracy(dataloader, model, device):
    """ Calculate accuracy metric. 
    
    Function create two local variables 'total' and 'correct' that used 
    for compute the correspondence of true labels and predict labels.
    
    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        DataLoader which contain dataset.
    model: torch.nn.Module
        Neural network model.
    deice: str
        Device where model are located.
        
    Returns
    -------
    float
        Accuracy metric result.
    
    See Also
    --------
    approx_accuracy
        
    Examples
    --------
    >>> dataloader = torch.utils.data.DataLoader(testset, shuffle=False)
    >>> model = torch.load('model_path')
    >>> model.eval()
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> accuracy = metrics_calc.true_accuracy(dataloader, model, device)
    
    """
    total, correct = 0, 0
    with no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.data.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    
def approx_accuracy(dataloader, model, device, fraction):
    """ Calculate accuracy metric on the part of dataset.
    
    Function create Subset of dataset and use true_accuracy() for accuracy calc.
    Dataset reduced in {denominator} times. 
    
    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        DataLoader which contain dataset.
    model: torch.nn.Module
        Neural network model.
    device: str
        Device where model are located.
    fraction: float
        Fraction of dataset which used in accuracy calculation.
        
    Returns
    -------
    float
        Accuracy metric result.
    
    See Also
    --------
    true_accuracy
    
    Examples
    --------
    >>> dataloader = torch.utils.data.DataLoader(testset, shuffle=False)
    >>> model = torch.load('model_path')
    >>> model.eval()
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> accuracy = train_tools.approx_accuracy(dataloader, model, device, fraction=0.1)
    
    """
    if 0 < fraction <= 1:
        dataset = dataloader.dataset
        data_subset = Subset(dataset, randint(0, high=len(dataset), size=int(len(dataset) * fraction)))
        data_subset_loader = DataLoader(data_subset, shuffle=False)
        accuracy = true_accuracy(data_subset_loader, model, device)
    else:
        raise ValueError('fraction have wrong value')
    return accuracy
    
def make_step(data, optimizer, model, criterion, device):
    """ Train loop in one function. 
    
    Parameters
    ----------
    data: torch.tensor
        Batch from dataset.
    optimizer: torch.optim
        Optimizer picked for training.
    model: nn.Module
        Neural network model.
    criterion: torch.nn.modules.loss
        Loss function picked for training.
    device: str
        Device where model are located.
        
    Returns
    -------
    outputs: torch.tensor
        Model outputs.
    loss: torch.tensor
        Loss function result.
    
    Examples
    --------
    >>> loss, outputs = train_tools.make_step(data, optimizer, model, criterion, device)
    
    """
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss, outputs
