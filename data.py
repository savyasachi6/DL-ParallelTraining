import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataloaders(rank, world_size):
    """
    Set up data loaders for CIFAR-10 dataset, considering distributed training.
    
    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    
    Returns:
        tuple: Training and testing data loaders
    """
    print(f"Rank {rank}: Setting up dataloaders with world_size={world_size}")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if world_size > 1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, sampler=train_sampler
    )
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4
    )
    print(f"Rank {rank}: Dataloaders initialized")
    return trainloader, testloader
