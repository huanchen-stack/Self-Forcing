import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# Dummy model definition
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Instantiate and move to GPU
model = MyModel().cuda()
inputs = torch.randn(32, 128).cuda()

# Run with profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("train_step"):
        out = model(inputs)
        loss = out.sum()
        loss.backward()

# Print a table of ops
print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=20
))

prof.export_chrome_trace("trace.json")