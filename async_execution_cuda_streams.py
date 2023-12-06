import torch
import time

# Ensure that CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a complex computation
@torch.jit.script
def complex_computation(a, b, result):
    # Simulate a computationally intensive operation
    for _ in range(1000):
        result.copy_(torch.sin(a) + torch.cos(b))

# Set up a larger dataset
N = 1024000  # Larger dataset size
a = torch.ones(N, dtype=torch.float32, device=device)
b = torch.ones(N, dtype=torch.float32, device=device)
result = torch.zeros(N, dtype=torch.float32, device=device)

# Define grid and block sizes
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Perform asynchronous computation using CUDA streams
start_time_async = time.time()
stream = torch.cuda.Stream(device)
with torch.cuda.stream(stream):
    complex_computation(a, b, result)
end_time_async = time.time()

# Perform synchronous computation for comparison
start_time_sync = time.time()
result_sync = torch.sin(a) + torch.cos(b)
torch.cuda.synchronize()
end_time_sync = time.time()

# Calculate execution times
time_async = end_time_async - start_time_async
time_sync = end_time_sync - start_time_sync

print(f"Time with asynchronous execution: {time_async} seconds")
print(f"Time with synchronous execution: {time_sync} seconds")

