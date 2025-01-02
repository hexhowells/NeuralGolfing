import hyperparameters as hp
from model import Network
from train import test, count_parameters

import torch
import numpy as np
from torchvision import datasets, transforms

from tqdm import tqdm
import torch.nn.utils.prune as prune

# Torch variables
device = "cpu"

torch.manual_seed(hp.seed)
torch.cuda.manual_seed(hp.seed)
torch.cuda.manual_seed_all(hp.seed)
print(f'seed: {torch.random.initial_seed()}')

# Create datasets
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
])

testing_data = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=hp.batch_size, shuffle=False)

# Load pre-trained model
model = torch.load("model.pth").to(device)

# Evaluate original accuracy
original_accuracy = test(model, device, test_loader)
print(f"Original Accuracy: {original_accuracy:.2f}%")
assert original_accuracy > 98, "Original accuracy is already below 98%."

# Prune and test weights one by one
pruned_weights = []  

for name, module in model.named_modules():
	if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
		weight = module.weight.data.clone()  # Original weights

		for idx in range(weight.numel()):
			original_value = weight.flatten()[idx].item()
			weight.flatten()[idx] = 0.0  # Set weight to 0 (prune)
			module.weight.data = weight.view_as(module.weight)  # Apply the modified weights
			
			# Test accuracy after pruning
			accuracy = test(model, device, test_loader)

			if accuracy >= 98:
				pruned_weights.append((name, idx))
				print(f"Pruning weight at layer {name} - idx {idx}. New Accuracy: {accuracy:.2f}%")
			else:
				# Restore the weight if accuracy drops below 98%
				weight.flatten()[idx] = original_value
				module.weight.data = weight.view_as(module.weight)
				print(f"Pruning weight at layer {name} - idx {idx}. New Accuracy: {accuracy:.2f}%. Reverting.")


# Final pruning summary
total_pruned = len(pruned_weights)
print("\nFinal Pruning Summary:")
for layer, idx in pruned_weights:
	print(f"Layer {layer}, Weight Index {idx} was pruned.")

final_accuracy = test(model, device, test_loader)
print(f"\nTotal Weights Pruned: {total_pruned}")
print(f"Final Accuracy After Pruning: {final_accuracy:.2f}%")

torch.save(model, "model-pruned.pth")
