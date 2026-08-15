# env-ssl-wrapper

One line turns any simulator's environment — gymnasium, dm_control, isaac, maniskill, pybullet, robosuite, pufferlib — into the same torch-native interface.

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
| `auto_batch` | Gives single envs a leading batch dim: `(4,)` → `(1, 4)`. |
| `action_transform` | Rescales actions from a canonical `(0, 1)` range to the env's bounds. |
| `tensor` | NumPy → torch on a device, torch actions → numpy for the sim. |
| `flatten_obs` | Flattens dict/tuple observations into a single vector. |

Every env emits the same contract: obs `torch.float32`, rewards `torch.float32`, `terminated`/`truncated` `torch.bool`. `env.seed(n)` works on every sim.

## Mock sims

`env_ssl_wrapper.mocks` ships dependency-free stand-ins emulating each simulator's quirks (`GymnasiumMockEnv`, `IsaacMockEnv`, `DMControlMockEnv`, ...) for testing your code without installing the real sims.

```python
from env_ssl_wrapper.mocks import IsaacMockEnv
env = compose_env(IsaacMockEnv(), 'tensor', 'done_tracker')
```

## Tests

```bash
uv sync --extra test
uv run pytest tests/test_real_envs.py
```
