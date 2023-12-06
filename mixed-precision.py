import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create a model, loss function, and optimizer
model = SimpleModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create some dummy data
inputs = torch.randn((64, 10)).cuda()
targets = torch.randn((64, 1)).cuda()

# Set up Automatic Mixed Precision (AMP) scaler
scaler = GradScaler()

# Training loop
for epoch in range(10):
    optimizer.zero_grad()

    # Use autocast to automatically perform mixed-precision training
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Use the scaler to perform backpropagation and gradient updates
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Optionally, you can disable autocast at the end of the training loop
# to ensure that subsequent operations are performed in full precision
torch.cuda.amp.autocast(False)

