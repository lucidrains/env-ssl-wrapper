# env-ssl-wrapper

One line turns any simulator's environment — mujoco warp, isaac sim, pybullet, gymnasium, pufferlib, dm_control, robosuite — into the same torch-native interface.

## Install

```bash
pip install env-ssl-wrapper
```

## Usage

```python
import torch
from env_ssl_wrapper import compose_env

env = compose_env(
    any_env,                                 # any env from any sim
    ('tensor', dict(device='cpu')),          # wrap with whatever you need
    'done_tracker',
)

obs, info = env.reset()                      # torch.float32, batched
while not env.all_done:
    actions = torch.randint(0, 2, (8,))
    obs, reward, terminated, truncated, info = env.step(actions)
```

Works identically for every simulator.

## Wrappers

Pass wrappers as strings (default config) or `(name, dict)` tuples (custom config), in any order.

| Wrapper | What it does |
|---|---|
| `standardize` | Normalizes any sim's `step`/`reset` signatures, vectorization, and autoreset into `(obs, reward, terminated, truncated, info)`. Applied automatically. |
| `time_limit` | Caps episodes, sets `truncated=True`. `('time_limit', dict(max_timesteps=200))` |
| `done_tracker` | Tracks per-env `episode_lengths`, exposes `env.all_done` / `env.needs_reset`. |
| `pad_episodes` | Standardizes padding for uneven vectorized episodes: done envs emit zeros (float/int) / `False` (bool) obs, and rewards are zeroed from the step after termination onward (the terminating step's own reward is the real terminal transition reward and is preserved). Applied automatically to vectorized envs. Works for autoreset (Isaac, gymnasium) and non-autoreset (pufferlib, maniskill) envs alike. |
| `auto_batch` | Gives single envs a leading batch dim: `(4,)` → `(1, 4)`. |
| `action_transform` | Rescales actions from a canonical `(0, 1)` range to the env's bounds. |
| `tensor` | NumPy → torch on a device, torch actions → numpy for the sim. |
| `flatten_obs` | Flattens dict/tuple observations into a single vector. |

Every env emits the same contract: obs `torch.float32`, rewards `torch.float32`, `terminated`/`truncated` `torch.bool`. `env.seed(n)` works on every sim.

Terminated envs are uniformly padded (zeros / `False` obs; rewards zeroed only after the terminating step, so the terminal transition's reward is never lost), and `info['final_observation']` — the true terminal obs, frozen per env and re-emitted while the env stays done — is always present once any env has terminated, with `info['_final_observation']` masking which envs it applies to. `env.is_done` always reflects the per-env done mask.

### Conforming Janky Simulators

Say you are handed a legacy or custom simulator with completely idiosyncratic signatures — non-standard method names, costs instead of rewards, inverted booleans, and custom rendering:

```python
class JankyRoboticsSim:
    def boot(self):
        return {'sensors': [0.1, -0.5, 1.2]}

    def actuate(self, torque):
        # returns sensor_dict, cost, is_alive
        return {'sensors': [0.2, -0.4, 1.1]}, 0.05, True

    def grab_pixels(self, w, h):
        return np.zeros((h, w, 3), dtype = np.uint8)
```

Wrap all that idiosyncrasy into an adapter in a few lines:

```python
import numpy as np
from env_ssl_wrapper import BaseEnvAdapter, register_adapter, compose_env

class JankySimAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        return isinstance(env, JankyRoboticsSim)

    def reset(self, **kwargs):
        obs = self.env.boot()
        return {'sensors': np.asarray(obs['sensors'])}, {}

    def step(self, action):
        data, cost, is_alive = self.env.actuate(action)
        return {'sensors': np.asarray(data['sensors'])}, -cost, not is_alive, False, {}

    def render(self, height, width, camera = None):
        return self.env.grab_pixels(width, height)

register_adapter(JankySimAdapter)
```

Now it behaves like every first-class simulator in the ecosystem:

```python
env = compose_env(
    JankyRoboticsSim(),
    ('image', dict(image_size = (64, 64))),
    'auto_batch',
    'tensor',
    'done_tracker'
)

obs, info = env.reset()
# obs['image']   -> torch.Size([1, 3, 64, 64])
# obs['sensors'] -> torch.Size([1, 3])

obs, reward, terminated, truncated, info = env.step(torch.randn(1, 1))
```

## Mock sims

`env_ssl_wrapper.mocks` ships dependency-free stand-ins emulating each simulator's quirks (`GymnasiumMockEnv`, `IsaacMockEnv`, `DMControlMockEnv`, ...) for testing your code without installing the real sims.

```python
from env_ssl_wrapper.mocks import IsaacMockEnv
env = compose_env(IsaacMockEnv(), 'tensor', 'done_tracker')
```

## Multiprocessing

Parallelize any single environment or factory into an autoresetting vector env:

```python
from env_ssl_wrapper import MultiprocessingVecEnv, compose_env

with MultiprocessingVecEnv('CartPole-v1', num_envs = 8) as env:
    env = compose_env(env, 'tensor', 'done_tracker')
    obs, info = env.reset()
```

## Tests

```bash
uv sync --extra test
uv run pytest tests/test_real_envs.py
```
