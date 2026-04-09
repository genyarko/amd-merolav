"""Sample script: partially migrated — mix of CUDA and ROCm patterns.

This simulates a file that was half-migrated manually.
"""

import os
import torch
import torch.nn as nn

# Already migrated
os.environ["HIP_VISIBLE_DEVICES"] = "0"
torch.backends.miopen.deterministic = True

# NOT yet migrated
torch.backends.cudnn.benchmark = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(10, 2).cuda()
