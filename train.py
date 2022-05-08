import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from originalModel import Network
import hyperparameters as hp


def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()

    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = (60_000//hp.batch_size) + 1
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=total):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    return acc


def main():
    torch.manual_seed(hp.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    loader_kwargs = {'batch_size': hp.batch_size,
                    'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(degrees=5)
        ])

    training_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    testing_data = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_data,**loader_kwargs)
    test_loader = torch.utils.data.DataLoader(testing_data, **loader_kwargs)

    model = Network().to(device)
    count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = StepLR(optimizer, step_size=hp.lr_step_size, gamma=hp.gamma)

    for epoch in range(1, hp.epochs + 1):
        print("\nEpoch: {}".format(epoch))
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

        scheduler.step()

        if acc >= 98:
            torch.save(model.state_dict(), "_mnist.pt")


if __name__ == '__main__':
    main()