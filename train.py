import torch
import torch.nn.functional as F
import hyperparameters as hp


def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()

    return total_params


def train(model, device, train_loader, optimizer, lossfn, epoch):
    model.train()
    total = (60_000 // hp.batch_size) + 1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, label="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        label, test_loss, correct, len(test_loader.dataset), acc))

    return acc
