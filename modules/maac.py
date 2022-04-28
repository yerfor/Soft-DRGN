import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ac_dgn import CriticDGNNetwork, ActorDGNNetwork

CriticMAACNetwork = CriticDGNNetwork
ActorMAACNetwork = ActorDGNNetwork
