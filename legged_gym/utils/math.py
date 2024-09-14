import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize,torch_rand_float
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

# @ torch.jit.script
def exp_avg_filter(x, avg, alpha=0.8):
    """
    Simple exponential average filter
    """
    avg = alpha*x + (1-alpha)*avg
    return avg

# @ torch.jit.script
def random_sample(env_ids, low, high, device):
        """
        Generate random samples for each entry of env_ids
        """
        rand_pos = torch_rand_float(0, 1, (len(env_ids), len(low)),
                                    device=device)
        diff_pos = (high - low).repeat(len(env_ids),1)
        random_dof_pos = rand_pos*diff_pos + low.repeat(len(env_ids), 1)
        return random_dof_pos 