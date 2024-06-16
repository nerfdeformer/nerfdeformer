import torch
import numpy as np
phi = torch.zeros((10,3)).cuda()
R = torch.nn.Parameter(phi, requires_grad=True)