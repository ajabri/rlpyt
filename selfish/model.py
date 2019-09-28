import torch
import torch.nn as nn


'''
 TODO
 0. Central agent baseline
 1. Implement shared relational encoder function (basically graph net / self-attention)
 1.a. Baseline: shared MLP action decoder function
 1.b. Skill Pool: action decoders indexed by key (basically a non-parametric mixture of experts)
        Try different gating strategies
        
'''
