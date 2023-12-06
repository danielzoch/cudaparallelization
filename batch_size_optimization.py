import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple neural network to demonstrate batch size optimization
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# Create dummy dataset
train_data = torch.randn((10000, 784)).to(device)
train_labels = torch.randint(0, 10, (10000,)).to(device)


# Experiment with different batch sizes
batch_sizes = [32, 64, 128, 256]
epoch_times_per_batch = {}

for batch_size in batch_sizes:
    print("\nTraining with batch size: {}".format(batch_size))

    # create DataLoader with the specified batch size
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    max_epoch_time = 60 # Set max allowed time for each epoch
    epoch_times = []

    for epoch in range(1, num_epochs +1):
        start_time = time.time()

        for inputs,labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        print("Epoch {} completed in {:.2f} seconds.".format(epoch, epoch_time))
        if epoch_time > max_epoch_time:

            print("watchdog timer triggered, stopping training.")
            break
                
    epoch_times_per_batch[batch_size] = epoch_times

    print("Training completed.")

# Compare results
plt.figure(figsize=(10, 6))

for batch_size, epoch_times in epoch_times_per_batch.items():
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, label="Batch Size {}".format(batch_size))

plt.xlabel("Epoch")
plt.ylabel("Epoch Time (seconds)")
plt.title("Training Epoch Times for Different Batch Size")
plt.legend()
plt.grid(True)
plt.show()
