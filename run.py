import hyperparameters as hp
from model import Network
from train import test, train, count_parameters

import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

# torch variables
device = "cpu"

torch.use_deterministic_algorithms(True)

torch.manual_seed(hp.seed)
torch.cuda.manual_seed(hp.seed)
torch.cuda.manual_seed_all(hp.seed)
print(f'seed: {torch.random.initial_seed()}')


# create datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

training_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
testing_data = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=hp.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=hp.batch_size, shuffle=False)


# define model
model = Network(
    conv1_out = hp.conv1_out,
    conv2_out = hp.conv2_out,
    conv3_out = hp.conv3_out, 
    kernel_size1 = hp.kernel_size1,
    kernel_size2 = hp.kernel_size2,
    kernel_size3 = hp.kernel_size3
    ).to(device)


print(f'param count: {count_parameters(model)}')
print(model.linear_in)


# training hyperparameters
optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
lossfn = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=hp.lr_step_size, gamma=hp.gamma)


# train model
for epoch in tqdm(range(hp.epochs)):
    train(model, device, train_loader, optimizer, lossfn, epoch)
    test(model, device, train_loader, 'Train')
    test(model, device, test_loader)

