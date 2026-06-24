from __future__ import annotations

import numpy as np
from torch import tensor, is_tensor, from_numpy, float64, device as torch_device
from torch.utils._pytree import tree_map

# helper functions

def numpy_to_torch(x, device, cast_float64_to_float32 = False):
    def _to_torch(t):
        if isinstance(t, np.ndarray):
            t = from_numpy(t)
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = tensor(t)

        if not is_tensor(t):
            return t

        if cast_float64_to_float32 and t.dtype == float64:
            t = t.float()

        return t.to(device)
    return tree_map(_to_torch, x)

def torch_to_numpy(x, cast_float64_to_float32 = False):
    def _to_numpy(t):
        if is_tensor(t):
            t = t.detach().cpu().numpy()
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = np.array(t)

        if not isinstance(t, np.ndarray):
            return t

        if cast_float64_to_float32 and t.dtype == np.float64:
            t = t.astype(np.float32)

        return t
    return tree_map(_to_numpy, x)

# classes

class TensorWrapper:
    def __init__(
        self,
        env,
        device: str | torch_device = 'cpu',
        convert_in: bool = True,
        convert_out: bool = True,
        cast_float64_to_float32: bool = False
    ):
        self.env = env
        self.device = torch_device(device)
        self.convert_in = convert_in
        self.convert_out = convert_out
        self.cast_float64_to_float32 = cast_float64_to_float32

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (numpy_to_torch(obs, self.device, self.cast_float64_to_float32), info) if self.convert_out else (obs, info)

    def step(self, action):
        action = torch_to_numpy(action, self.cast_float64_to_float32) if self.convert_in else action
        out = self.env.step(action)
        
        if not self.convert_out:
            return out

        return *numpy_to_torch(out[:4], self.device, self.cast_float64_to_float32), out[4]
