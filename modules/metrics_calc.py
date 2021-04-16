from torch import no_grad
from torch.utils.data import Subset, DataLoader
from numpy.random import randint

def true_accuracy(dataloader, model, device):
    total, correct = 0, 0
    with no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
    #         outputs = outputs.to(device)
            _, predicted = outputs.data.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    
def approx_accuracy(dataset, model, device, reduce):
    total, correct = 0, 0
    with no_grad():
        data_subset = Subset(dataset, randint(0,high=len(dataset), size=len(dataset)//reduce))
        data_subset_loader = DataLoader(data_subset, shuffle=False)
        for images, labels in data_subset_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
#             outputs = outputs.to(device)
            _, predicted = outputs.data.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    