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

### State Normalization

Running mean / std normalization of the states leaving the environment — Welford statistics, exact under any update order, one accumulator per obs key.

```python
import gymnasium as gym
from env_ssl_wrapper import StandardizeEnvWrapper, StateNormalizationWrapper

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
env = StateNormalizationWrapper(env)          # update_mode = 'live'

obs, info = env.reset()
obs, *_ = env.step(...)                       # normalized with the running statistics

env.mean('obs'), env.std('obs'), env.count('obs')
env.denormalize(obs)                          # back to the sim's units
```

For dictionary observations, you can normalize all floating point state leaves, target a subset of keys, or track multiple keys with separate statistics:

```python
# normalize only a subset of keys (other keys pass through untouched)
env = StateNormalizationWrapper(dict_env, keys = 'proprio')

# or multiple keys, each with its own separate running statistics
env = StateNormalizationWrapper(dict_env, keys = ('proprio', 'goal'))

# or through StandardizeEnvWrapper directly
env = StandardizeEnvWrapper(dict_env, normalize_state = 'proprio')
env = StandardizeEnvWrapper(dict_env, normalize_state = ('proprio', 'goal'))
```

`update_mode` decides *when* the statistics move:

| mode | statistics |
|---|---|
| `'live'` | folded in on every `reset` / `step` — a state is normalized with the statistics as they stood before it arrived (gymnasium / SB3 style) |
| `'deferred'` | frozen for the current rollout, new states accumulate in `env.pending` — commit them when the learner has updated |
| `'frozen'` | never folded in — evaluation |

The deferred schedule is the RSMNorm one from x-ppo: the actor and critic train against one snapshot of the statistics per rollout, and the commit happens after the actor-critic update.

```python
env = StateNormalizationWrapper(env, update_mode = 'deferred')

for iteration in range(...):
    ...                      # collect a rollout - states leave the env normalized with the committed statistics
    ppo_update(...)
    env.commit_stats()       # make this rollout's states current

# evaluation runs on a frozen copy of the training statistics
eval_env.load_state_dict(env.state_dict())
eval_env.update_mode = 'frozen'
```

`commit_stats()` needs no raw states: the pending accumulator holds exact sufficient statistics (count / mean / variance), so it can be stored in a memmap replay buffer as normalized states and committed at the update boundary all the same. For pipelines that keep raw states, `commit_stats(raw_states)` folds external data in — a tensor, an iterable of batches (a replay buffer dataloader), or a dict keyed by obs key.

Because states are normalized as they leave the env, every consumer sees the same tensor, so a transition written to a replay buffer pairs with the normalized state the actor actually consumed — consistent with its stored log prob and value no matter when the statistics were committed.

#### Distributed

Under `torchrun` every rank sees its own slice of the state distribution, so per-rank statistics are different random variables and each rank would normalize with a different affine transform. Committing globally all-reduces the pending statistics — every rank ends with the moments of the union of all ranks' states, identical to what a single process would have, and every rank normalizes identically:

```python
env.commit_stats(distributed = True)   # the deferred schedule - call it on every rank after the learner's update
env.sync_stats()                       # live mode - pull the per-rank running statistics together, e.g. per rollout
```

```bash
torchrun --nproc_per_node=4 test_state_norm_distributed.py
# deferred - commit_stats(distributed = True)
#   union states:   2373 | statistics count: 2374 | prior: 1
#   count == union count: True | identical across ranks: True | probe identical: True
#   max |mean - reference|: 5.551e-17 | max |std - reference|: 4.441e-16
```

Each rank folds its own terminal states and skips its own padded slots as usual; the merge strips each rank's pseudo count and adds one back, so the global count and moments come out the same as the single process reference.

Terminal states are folded from `info['final_observation']`, zero padded slots of autoresetting vector envs are skipped, and only floating point leaves are touched. A single pseudo-observation (mean 0, variance 1) is counted in from the start, so an untrained statistic normalizes near identity instead of amplifying a constant state dimension without bound — pass `prior_count = 0.` for pure empirical moments.

Run the benchmark — live vs deferred statistics, both against no normalization at all, on cartpole states put through a fixed scaler so the dimensions are badly conditioned:

```bash
uv run train_state_norm.py --steps=50000 --seeds=0,1,2
uv run train_state_norm.py --envs=CartPole-v1,Acrobot-v1 --variants=raw,severe
uv run analyze_state_norm.py     # per-arm finals / auc, and the paired live vs deferred difference

# and the same question one level down — the size of the states the actor consumes, step by step
uv run probe_state_norm.py
```

Also available through composition:

```python
env = compose_env(gym.make('CartPole-v1'), ('state_norm', dict(update_mode = 'deferred')))
env = StandardizeEnvWrapper(gym.make('CartPole-v1'), normalize_state = dict(update_mode = 'deferred'))
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

Chunks are shaped `(num_envs, chunk_len, *action_shape)` (`(num_envs, chunk_len)` for discrete actions). The wrapper can also be applied directly to raw gymnasium environments, single or vectorized — torch action chunks are converted to numpy for the underlying simulator.

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

### Time & Past Action

Two small wrappers that append temporal context to the observation — the step index, and the action that led into the current state.

```python
import gymnasium as gym
from env_ssl_wrapper import StandardizeEnvWrapper, TimeWrapper, PastActionsWrapper

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
env = TimeWrapper(env)          # obs['time']        -> (1, 1) raw ints, 0 at reset
env = PastActionsWrapper(env)   # obs['past_action'] -> (1,) the action that led here, zeros on the first step
```

`TimeWrapper` supplies the per-episode timestep — raw integers by default, or any function of them as the feature:

```python
from env_ssl_wrapper import fourier_time

env = TimeWrapper(env, transform = fourier_time)          # obs['time'] -> (1, 8) sin / cos features
env = TimeWrapper(env, transform = lambda t: t.float())   # raw floats
```

`PastActionsWrapper` supplies the action that led into the current observation. The first step of an episode has no past action: it emits zeros, or `None` when `first_action = None`.

```python
env = PastActionsWrapper(env, key = 'prev_action', first_action = None)
```

Keep everything tensor-valued with `has_past_actions_key` — the first step emits zeros, and a `Bool[num_envs]` field says which slots actually have a past action (this is what `PastActionEncoder` from `x-mlps-pytorch` reads):

```python
env = PastActionsWrapper(env, has_past_actions_key = 'has_past_actions')   # obs['has_past_actions'] -> (num_envs,) bools
```

Both wrappers track the batch slot by slot — an env that autoresets mid-rollout gets a fresh value — and both attach to the terminal `info['final_observation']` like the rest of the library.

Run the CartPole demo — the one-hot last action fed back in, against a zeros control, same learner, seed and budget:

```bash
uv run train_cartpole.py    # both arms converge to a perfect greedy 500.0
```

### Success-Guided Sampling

Adaptive curriculum over a bank of reset states, keeping episodes near the policy's frontier of competence — the configurations it currently solves about `target` of the time ([Octi Zhang et al., 2026](https://arxiv.org/abs/2610.12465))

```python
import gymnasium as gym
import torch
from env_ssl_wrapper import StandardizeEnvWrapper, SuccessGuidedSamplingWrapper

reset_states = [torch.zeros(4), torch.ones(4)]  # bank of reset states to sample over

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
env = SuccessGuidedSamplingWrapper(env, reset_configs = reset_states, target = 0.5, kappa = 2.)

obs, info = env.reset()
# info['sgs_config_id'] -> the config this episode started from (success rates update on episode end)
```

`reset_configs` holds reset states — or callables drawing one (`lambda rng: ...`) — passed through `options`, or built by a `reset_options_fn` / `state_transform`.

The score equation is swappable: the default beta shaped kernel around `target` (`sgs_score_fn`) is simply one rule of the bank's state. Whatever you pass as `score_fn` is handed, by name, the arguments it declares — `p` (per state success rates), `counts`, `history` (the raw episode outcomes), `step`, `window`, and the hparams `target`, `kappa`, `eps`, `unvisited_weight`.

The sampling distribution is swappable too: the default follows the paper's Eq. (2) — the beta shape the score produces is floored at `eps`, logits are formed and softmaxed at `temperature` (floored at 1; the paper uses 2.) — and `sgs_sample_fn` draws i.i.d. from those `probs`. A custom `sample_fn` is handed, by name, whatever it declares — `p`, `counts`, `history`, `configs`, `weights` (the raw kernel), `probs` (floored and tempered), `rng`, `size`, `step`, `temperature`, and the hparams — so any distribution of the bank can be plugged in.

The rest of the loop is hookable as well: target and temperature schedules, uniform floor (`uniform_prob`), env-native fallback (`default_prob`), success signal (`success_fn`).

Track the curriculum as it runs — one snapshot of the bank per credited episode, kept in memory and appended to a csv when `log_path` is set:

```python
env = SuccessGuidedSamplingWrapper(
    StandardizeEnvWrapper(gym.make('CartPole-v1')),
    reset_configs = reset_states,
    log_every = 1,                  # snapshot every episode (None turns logging off)
    log_path = 'sgs_progression.csv',
)

env.trace                           # [dict(step, success_rates, probs), ...]
```

LunarLander is the worked example — a difficulty ladder of reset states, three arms (native spawns, uniform bank resets, SGS), paired evaluation and the speedup chart:

```bash
uv run train_lunar_sgs.py --steps=200000 --seeds=0,1,2,3,4   # writes experiments/lunar_sgs/
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
| `time` | Supplies the per-episode timestep. `('time', dict(transform = fourier_time))` |
| `past_action` | Supplies the last action. `('past_action', dict(first_action = None))` |
| `state_norm` | Running mean / std normalization of states, live or deferred until the learner updates. `('state_norm', dict(update_mode = 'deferred'))` |
| `sgs` | Adaptive frontier curriculum over reset configs. `('sgs', dict(reset_configs = configs, target = 0.5, kappa = 2.))` |

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

## Evaluation

`evaluate_actor` rolls a policy through any standardized env and returns per-episode stats, optionally recording videos.

```python
import gymnasium as gym
from env_ssl_wrapper import StandardizeEnvWrapper, evaluate_actor

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))

stats = evaluate_actor(policy, env, episodes = 10, seed = 0)

stats.mean          # mean return
stats.std           # std of returns
stats.min, stats.max
stats.success_rate(475.)          # fraction of episodes above a threshold
stats.summary()                   # dict(episodes, mean, std, min, max)

# deterministic = False samples from distributions instead of collapsing to the mode
# an actor exposing `.dist` takes precedence over calling it directly
# video_path = 'eval.mp4' records the first video_episodes episodes (single env only)
```

## Tests

```bash
uv sync --extra test
uv run pytest

# distributed statistics — every rank ends with the union statistics, bitwise identical
torchrun --nproc_per_node=4 test_state_norm_distributed.py
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

```bibtex
@article{zhang2026balanced,
    title   = {A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control},
    author  = {Octi Zhang and Mateo Guaman Castro and Patrick Yin and Ignacio Dagnino and Abhishek Gupta and Rosario Scalise and Byron Boots},
    journal = {arXiv preprint arXiv:2610.12465},
    year    = {2026}
}
```
