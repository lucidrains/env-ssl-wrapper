# env-ssl-wrapper

One line turns any simulator's environment — mujoco warp, isaac sim, pybullet, gymnasium, pufferlib, dm_control, robosuite — into the same torch-native interface.

## Install

```bash
pip install env-ssl-wrapper
```

## Standardize

```python
import torch
from env_ssl_wrapper import StandardizeEnvWrapper

# one master wrapper to standardize any simulator environment

env = StandardizeEnvWrapper(any_env)

obs, info = env.reset() # torch.float32, batched

while not env.all_done:
    actions = torch.randint(0, 2, (1,))
    obs, reward, terminated, truncated, info = env.step(actions)
```

Works identically for every simulator.

Or compose individual wrappers piecemeal:

```python
from env_ssl_wrapper import compose_env

env = compose_env(
    any_env,
    ('tensor', dict(device = 'cpu')),
    'done_tracker',
)
```

## Research Wrappers

### Memory Trace

A cheap memory that keeps track of an exponential moving average (EMA) of a state, or subset ([Eberhard et al., 2025](https://arxiv.org/abs/2503.15200)).

```python
import gymnasium as gym
from env_ssl_wrapper import StandardizeEnvWrapper, MemoryTraceWrapper

env = StandardizeEnvWrapper(gym.make('LunarLander-v3'))
env = MemoryTraceWrapper(env, lambdas = (0.9, 0.99))

obs, info = env.reset()
# obs['obs']        -> (1, 8)
# obs['trace_0.9']  -> (1, 8)
# obs['trace_0.99'] -> (1, 8)
```

Or on a subset of dictionary observations:

```python
env = MemoryTraceWrapper(env, lambdas = (0.9, 0.99), keys = 'proprio')
# obs['proprio_trace_0.9'] -> (1, 2)
```

Run the PPO benchmark on POMDP LunarLander:

```bash
uv run test_memory_trace.py
```

### Action Chunk

Open-loop action chunking: one `step` executes a whole chunk of actions and returns only the final state, temporally compressing the environment by the chunk length.

```python
import gymnasium as gym
import torch
from env_ssl_wrapper import StandardizeEnvWrapper, ActionChunkWrapper

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
env = ActionChunkWrapper(env, chunk_len = 2, gamma = 0.99)

obs, info = env.reset()

while True:
    actions = torch.randint(0, 2, (1, 2))  # (num_envs, chunk_len)
    obs, reward, terminated, truncated, info = env.step(actions)

    # reward                -> discounted sum of substeps (r0 + gamma * r1 + ...)
    # info['discount']      -> gamma ** chunk_length (macro-step discount for GAE / Bellman target)
    # info['chunk_length']  -> env steps actually executed (drops below chunk_len on terminal chunk)
    # info['chunk_rewards'] -> (1, chunk_length) raw per-step rewards

    if bool((terminated | truncated).item()):
        obs, info = env.reset()
```

Chunks are shaped `(num_envs, chunk_len, *action_shape)` (`(num_envs, chunk_len)` for discrete actions).

Execution stops early the moment any env terminates or truncates mid-chunk — the terminal state and done flags of that substep are returned and the rest of the chunk is dropped, so a new episode is never silently advanced.

Pass `gamma` (default `1.`) to discount intra-chunk rewards $r = \sum_{i=0}^{L-1} \gamma^i r_i$. The macro-transition discount factor to the next state is provided as `info['discount'] = gamma ** chunk_length`. `reward_mode` can also be `'mean'` or `'last'`.

Set `reward_mode = 'chunk'` to get the raw per-step rewards back as a reward chunk of shape `(num_envs, chunk_length)` (`(chunk_length,)` for a single env) instead of a reduced scalar — useful when a learner wants to do per-step credit assignment within the chunk. The chunk is truncated to the number of substeps actually executed (`info['chunk_length']`) and is undiscounted, so `gamma` only affects the aggregated modes.

Can also be passed directly to `StandardizeEnvWrapper` or `compose_env`:

```python
env = StandardizeEnvWrapper(gym.make('CartPole-v1'), chunk_len = 2, chunk_gamma = 0.99)
```

Run the chunked PPO benchmark on CartPole, or check chunked rollouts against a per-step reference:

```bash
uv run test_action_chunk.py --chunk_len=2                # PPO on CartPole
uv run test_action_chunk.py --sweep=True                 # PPO over several chunk lengths
uv run test_action_chunk.py --verify=True --sweep=True   # chunked rollouts match per-step reference
```

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
| `action_chunk` | Executes actions in open-loop chunks of length k. `('action_chunk', dict(chunk_len = 2, gamma = 0.99))` |

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
uv run pytest
```

## Citations

```bibtex
@article{eberhard2025partially,
    title   = {Partially Observable Reinforcement Learning with Memory Traces},
    author  = {Onno Eberhard and Michael Muehlebach and Claire Vernade},
    journal = {arXiv preprint arXiv:2503.15200},
    year    = {2025}
}
```
