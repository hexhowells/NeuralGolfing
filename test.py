import hyperparameters as hp
from train import test, count_parameters
import torch
from torchvision import datasets, transforms

# Torch variables
device = "cpu"

# Create datasets
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
])

testing_data = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=hp.batch_size, shuffle=False)

# Load pre-trained model
model = torch.load("model-pruned.pth").to(device)

# Evaluate model
param_count = count_parameters(model)
accuracy = test(model, device, test_loader)

print(f"Parameter Count: {param_count}\nAccuracy: {accuracy:.2f}%")