import numpy as np
import torch
import torch.nn as nn


def model_is_on_cpu(model):
    return next(model.parameters()).device.__str__() == 'cpu'


def to_cpu(model):
    # In torch, 'model.cpu()' would reload parameters even it is already on cpu.
    # This function is utilized for better performance.
    if model is None:
        return None
    if isinstance(model, torch.Tensor):
        if not model.device == torch.device("cpu"):
            return model.cpu()
        else:
            return model
    else:
        if not model_is_on_cpu(model):
            return model.cpu()
        else:
            return model


def to_cuda(model):
    # In torch, 'model.cuda()' would reload parameters even it is already on cpu.
    # This function is utilized for better performance.
    if model is None:
        return None
    if isinstance(model, torch.Tensor):
        if model.device == torch.device("cpu"):
            return model.cuda()
        else:
            return model
    else:
        if model_is_on_cpu(model):
            return model.cuda()
        else:
            return model


def hard_update(learned_model, target_model):
    for target_param, param in zip(target_model.parameters(), learned_model.parameters()):
        target_param.data.copy_(param.data)


def soft_update(learned_model, target_model, tau):
    for target_param, param in zip(target_model.parameters(), learned_model.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def apply_fn_2_dict(dic, fn, n_dim=1):
    if n_dim == 1:
        ret = {}
        for k, v in dic.items():
            # try:
            ret[k] = fn(v)
            # except:
            #     pass
        return ret
    elif n_dim == 2:
        adj = {}
        for k, v in dic.items():
            adj[k] = {}
            for k_t, t in v.items():
                # try:
                adj[k][k_t] = fn(t)
                # except:
                #     pass
        return adj
    else:
        raise NotImplementedError


def to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)


def dict_tensor(array_dict, n_dim=1):
    return apply_fn_2_dict(array_dict, to_tensor, n_dim)


def dict_cpu(tensor_dict, n_dim=1):
    return apply_fn_2_dict(tensor_dict, to_cpu, n_dim)


def dict_cuda(cpu_tensor_dict, n_dim=1):
    return apply_fn_2_dict(cpu_tensor_dict, to_cuda, n_dim)


def dict_numpy(cpu_tensor_dict, n_dim=1):
    return apply_fn_2_dict(cpu_tensor_dict, lambda x: x.detach().numpy(), n_dim)


def dict_squeeze(tensor_dict, n_dim=1):
    return apply_fn_2_dict(tensor_dict, lambda x: x.squeeze(), n_dim)


def dict_unsqueeze(tensor_dict, n_dim=1):
    return apply_fn_2_dict(tensor_dict, lambda x: x.unsqueeze(0), n_dim)


def dict_np_unsqueeze(array_dict, n_dim=1):
    return apply_fn_2_dict(array_dict, lambda x: x[np.newaxis, :], n_dim)


def cut_homo_comm_mask(mask, cut_prob):
    mask = torch.tensor(mask)
    n_ant = mask.shape[-1]
    rand_mask = (torch.rand(n_ant, n_ant) > cut_prob)
    for i in range(n_ant):
        rand_mask[i, i] = True
    sum_mask = rand_mask.int() + mask.int()
    cut_mask = (sum_mask == 2).int()
    return cut_mask.numpy()


def get_grad_norm(model, l=2):
    num_para = 0
    accu_grad = 0
    if isinstance(model, torch.nn.Module):
        params = model.parameters()
    elif isinstance(model, torch.nn.Parameter):
        params = [model]
    else:
        params = model
    for p in params:
        if p.grad is None:
            continue
        num_para += p.numel()
        if l == 1:
            accu_grad += p.grad.abs(1).sum()
        elif l == 2:
            accu_grad += p.grad.pow(2).sum()
        else:
            raise ValueError("Now we only implement l1/l2 norm !")
    if l == 2:
        accu_grad = accu_grad ** 0.5
    return accu_grad
