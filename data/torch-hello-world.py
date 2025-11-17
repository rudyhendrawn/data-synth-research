import time
import torch

device = "mps"

torch.manual_seed(1234)
TENSOR_A_CPU = torch.randn(5000, 5000)
TENSOR_B_CPU = torch.randn(5000, 5000)

torch.manual_seed(1234)
TENSOR_A_MPS = TENSOR_A_CPU.to(device)
TENSOR_B_MPS = TENSOR_B_CPU.to(device)

# for _ in range(100):
#     torch.matmul(torch.randn(100, 100).to(device), torch.randn(100, 100).to(device))

start_time = time.time()
torch.matmul(TENSOR_A_CPU, TENSOR_B_CPU)
print(f"CPU time: {time.time() - start_time:.6f} seconds")

start_time = time.time()
torch.matmul(TENSOR_A_MPS, TENSOR_B_MPS)
print(f"MPS time: {time.time() - start_time:.6f} seconds")