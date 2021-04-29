from torch import no_grad
from torch.utils.data import Subset, DataLoader
from numpy.random import randint

def true_accuracy(dataloader, model, device):
    ''' Calculate accuracy metric. 
    
    Function create two local variables 'total' and 'correct' that used 
    for compute the correspondence of true labels and predict labels.
    
    Args: 
        dataloader: DataLoader which contain dataset with labels.
        model: model what we want to check.
        device: device where model are allocate.
        
    Return:
        float
    
    Standalone usage:
    
    >>> dataloader = torch.utils.data.DataLoader(testset, shuffle=False)
    >>> model = torch.load('model_path')
    >>> model.eval()
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> accuracy = metrics_calc.true_accuracy(dataloader, model, device)
    '''
    total, correct = 0, 0
    with no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.data.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    
def approx_accuracy(dataset, model, device, denominator):
    ''' Calculate accuracy metric on the part of dataset.
    
    Function create Subset of dataset and use true_accuracy() for accuracy calc.
    Dataset reduced in {denominator} times. 
    For example 'denominator = 4' mean that would be used 1/4 of dataset.
    
    Args: 
        dataset: full dataset with labels.
        model: checked model.
        device: device where model are allocate.
        denominator: define part of dataset which would be use for accuracy calc
                
    Return:
        float
    
    '''
    
    data_subset = Subset(dataset, randint(0,high=len(dataset), size=len(dataset)//denominator))
    data_subset_loader = DataLoader(data_subset, shuffle=False)
    accuracy = true_accuracy(data_subset_loader, model, device)
    
    return accuracy
    