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

Chunks are shaped `(num_envs, chunk_len, *action_shape)` (`(num_envs, chunk_len)` for discrete actions), with the `num_envs` axis optional for a single env — which is what `AsyncVectorEnv` workers receive. With `chunk_len = 1`, a bare `(*action_shape)` / `(num_envs, *action_shape)` action is also accepted. The wrapper can also be applied directly to raw gymnasium environments, single or vectorized — torch action chunks are converted to numpy for the underlying simulator.

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

### Time, Past Action & State

Small wrappers that append temporal context to the observation — the step index, and the actions and states that led into the current state.

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

Keep everything tensor-valued with `has_past_actions_key` — the first step emits zeros, and a `Bool[num_envs]` field says which slots actually have a past action (this is what `DiscretePastActions` from `x-mlps-pytorch` reads for a single past action):

```python
env = PastActionsWrapper(env, has_past_actions_key = 'has_past_actions')   # obs['has_past_actions'] -> (num_envs,) bools
```

Keep more than one action with `num_past_actions` — the history keeps the last n actions, oldest first, in a trailing axis. The most recent action is `[..., -1]`, and `has_past_actions` grows the same axis:

```python
env = PastActionsWrapper(env, num_past_actions = 2, has_past_actions_key = 'has_past_actions')
# obs['past_action']       -> (num_envs, 2) discrete, or (num_envs, 2, *action_shape) continuous
# obs['has_past_actions']  -> (num_envs, 2) bools, False until each slot is filled
```

The default `num_past_actions = 1` keeps the original observation shape. Pass a sequence to keep specific lags only:

```python
env = PastActionsWrapper(env, num_past_actions = (1, 2, 4, 8))
# obs['past_action'] -> (num_envs, 4) discrete, or (num_envs, 4, *action_shape)
```

`PastStatesWrapper` does the same for states. The current state stays in `obs['obs']`; the history holds the states before it, oldest first. Any positive `num_past_states` works:

```python
from env_ssl_wrapper import PastStatesWrapper

env = PastStatesWrapper(env, num_past_states = 4, has_past_states_key = 'has_past_states')
# obs['past_state']      -> (num_envs, 4, *state_shape)
# obs['has_past_states'] -> (num_envs, 4) bools
```

Pass a sequence to keep specific lags only:

```python
env = PastStatesWrapper(env, num_past_states = (1, 2, 4, 8))
# obs['past_state'] -> (num_envs, 4, *state_shape) — lags 8, 4, 2, 1
```

Both wrappers track the batch slot by slot — an env that autoresets mid-rollout gets a fresh value — and both attach to the terminal `info['final_observation']` like the rest of the library.

Run the CartPole demo — the last action fed back in, against a zeros control, same learner, seed and budget:

```bash
uv run train_cartpole.py    # both arms converge to a perfect greedy 500.0
```

### Success-Guided Sampling

Adaptive curriculum over a bank of reset states, keeping episodes near the policy's frontier of competence — the configurations it currently solves about `frontier_success_rate` of the time ([Octi Zhang et al., 2026](https://arxiv.org/abs/2610.12465))

```python
import gymnasium as gym
import torch
from env_ssl_wrapper import StandardizeEnvWrapper, SGSWrapper

reset_states = [torch.zeros(4), torch.ones(4)]  # bank of reset states to sample over

env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
env = SGSWrapper(env, reset_states, frontier_success_rate = 0.5, kernel_sharpness = 2.)

obs, info = env.reset()
# info['sgs_config_id'] -> the config this episode started from (success rates update on episode end)
```

`configs` holds reset states — or callables drawing one (`lambda rng: ...`) — passed through `options`, or built by a `reset_options_fn` / `state_transform`.  `reset_configs` is accepted as an alias.

Banks can nest, and a group may mix leaves with groups, so the tree can be ragged:

```python
reset_states = [
    native_state,                                       # leaf at the root
    dict(name = 'easy', configs = [easy_a, easy_b]),    # named group
    [hard_a, [hard_b, hard_c]],                         # deeper branch
]

env = SGSWrapper(env, reset_states, frontier_success_rate = .5)
```

A group's rate is pooled from its children (count weighted by default; override with `success_fn`), and a group may set `target`, `kappa`, `eps` or `unvisited_weight` for its group children.  Sampling stays one categorical draw over leaves, and `sampler.group_rates` / `group_probs` expose the tree.

For a bank ordered by difficulty, `build_difficulty_bank` composes a ladder from generator factories and appends the native tail, and `difficulty_score_fn` scores against an explicit difficulty — so replicas at one difficulty and non-uniform rungs do not distort the frontier:

```python
from functools import partial
from env_ssl_wrapper import SGSWrapper, build_difficulty_bank, difficulty_score_fn

configs, difficulty, native = build_difficulty_bank(
    (curriculum, runway, circumnav),
    native,
    schedule = (0, 1, 0, 2),   # optional generator cycle
)

env = SGSWrapper(
    env,
    configs,
    frontier_success_rate = .35,
    score_fn = partial(difficulty_score_fn, difficulty = difficulty),
)
```

For a non-autoresetting worker — e.g. a simulator inside `AsyncVectorEnv` with `autoreset_mode='Disabled'` — pass `autoresets = True` to `SGSWrapper`, so it resets the worker itself and the next episode starts from the sampled config.

The score equation is swappable: the default beta shaped kernel around `frontier_success_rate` (paper: `target`; `sgs_score_fn`) is simply one rule of the bank's state. Whatever you pass as `score_fn` is handed, by name, the arguments it declares — `p` (per state success rates), `counts`, `history` (the raw episode outcomes), `step`, `window`, and the hparams `target`, `kappa`, `eps`, `unvisited_weight` (the `target` / `kappa` names are the paper aliases for `frontier_success_rate` / `kernel_sharpness`).

The sampling distribution is swappable too: the default follows the paper's Eq. (2) — the beta shape the score produces is floored at `eps`, logits are formed and softmaxed at `temperature` (floored at 1; the paper uses 2.) — and `sgs_sample_fn` draws i.i.d. from those `probs`. A custom `sample_fn` is handed, by name, whatever it declares — `p`, `counts`, `history`, `configs`, `weights` (the raw kernel), `probs` (floored and tempered), `rng`, `size`, `step`, `temperature`, and the hparams — so any distribution of the bank can be plugged in.

The rest of the loop is hookable as well: frontier-success and temperature schedules, uniform floor (`uniform_prob`), env-native fallback (`default_prob`), success signal (`success_fn`).

Track the curriculum as it runs — one snapshot of the bank per credited episode, kept in memory and appended to a csv when `log_path` is set:

```python
env = SGSWrapper(
    StandardizeEnvWrapper(gym.make('CartPole-v1')),
    reset_states,
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
| `action_transform` | Rescales actions from a canonical `(0, 1)` range to the env's bounds, with optional `overshoot` past them. `('action_transform', dict(auto = True, overshoot = 0.1))` |
| `tensor` | NumPy → torch on a device, torch actions → numpy for the sim. |
| `flatten_obs` | Flattens dict/tuple observations into a single vector. |
| `action_chunk` | Executes actions in open-loop chunks of length k. `('action_chunk', dict(chunk_len = 2, gamma = 0.99))` |
| `time` | Supplies the per-episode timestep. `('time', dict(transform = fourier_time))` |
| `past_action` | Supplies the last action, or the last n with `num_past_actions`. `('past_action', dict(num_past_actions = 2))` |
| `past_state` | Supplies the last state, or the last n with `num_past_states`. `('past_state', dict(num_past_states = 5))` |
| `state_norm` | Running mean / std normalization of states, live or deferred until the learner updates. `('state_norm', dict(update_mode = 'deferred'))` |
| `sgs` | Adaptive frontier curriculum over reset configs. `('sgs', dict(configs = configs, frontier_success_rate = 0.5, kernel_sharpness = 2.))` |

Every env emits the same contract: obs `torch.float32`, rewards `torch.float32`, `terminated`/`truncated` `torch.bool`. `env.seed(n)` works on every sim.

Terminated envs are uniformly padded (zeros / `False` obs; rewards zeroed only after the terminating step, so the terminal transition's reward is never lost), and `info['final_observation']` — the true terminal obs, frozen per env and re-emitted while the env stays done — is always present once any env has terminated, with `info['_final_observation']` masking which envs it applies to. `env.is_done` always reflects the per-env done mask.

### Action Transform

Maps a policy's action range onto the env's action bounds, then clips. `auto = True` reads the bounds off the env and rescales `from_range` (default `(0, 1)`, beta-friendly) onto them, leaving unbounded dimensions untouched:

```python
from env_ssl_wrapper import ActionTransformWrapper

env = ActionTransformWrapper(env, auto = True)
```

`overshoot` lands the mapping that fraction of the half range past each bound, and the clip pulls the excess back onto the bounds — so a bounded policy whose mean saturates a few percent short of its own range (mean-conc-beta at a bound) can still sit exactly on the env's bound:

```python
env = ActionTransformWrapper(env, auto = True, overshoot = 0.1)

# a policy mean at 96% of its range -> 2.115 on (-2, 2) -> clipped onto 2.0
```

The policy keeps emitting raw actions and the clip lives on the env side, so log probs and policy gradients stay exact. `0.05` clears the usual saturation gap; `0.2` to `0.25` measured best on Pendulum-v1 over 16 seeds. The mapping stretches the interior by the same factor, so keep it small when action magnitude carries meaning outside the policy (behavior cloning, Q on actions) — or skip the stretch entirely by declaring the widened range on the policy and clipping only:

```python
# policy emits on (-2.2, 2.2), env takes (-2, 2) — identity inside, clip outside
env = ActionTransformWrapper(env, clip = (-2., 2.))
```

Passing `overshoot` without `auto` raises.

Everything the auto path does can also be written out by hand — the overshoot lives in the `to_range`, and `clip` is applied last:

```python
env = ActionTransformWrapper(
    env,
    transforms = [dict(rescale_from_to = ((-1., 1.), (-2.2, 2.2)))],
    clip = (-2., 2.)
)

# 0.5 -> 1.1, and a saturated 0.9615 -> 2.115 -> 2.0
```

`transforms` takes a list, with `indices` for per-dimension ranges, and `clip` takes per-dimension bounds:

```python
import numpy as np

env = ActionTransformWrapper(
    env,
    transforms = [
        dict(indices = slice(0, 1), rescale_from_to = ((0., 1.), (-1.1, 1.1))),
        dict(indices = slice(1, 2), rescale_from_to = ((0., 1.), (0., 2.2)))
    ],
    clip = (np.array([-1., 0.]), np.array([1., 2.]))
)
```

A bounded policy can hand over the settings in one dict — mean-conc-beta, for instance:

```python
from mean_conc_beta import Beta

beta = Beta(bounds = (-2., 2.), overshoot = 0.2)

env = ActionTransformWrapper(env, **beta.action_transform_kwargs())
```

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

### Seeding and per-slot resets

`reset_env(env, seed)` seeds every slot with the same seed — common random numbers, so vectorized fitness comparisons differ only by policy. `reset_env_at(env, i, seed)` resets one slot; it uses an explicit `reset_at` / `reset_one`, gymnasium 1.x's `options['reset_mask']`, or the env's sub-env list, and returns that slot's obs / info (or `None` when the env has no per-slot reset path). `write_slot_obs` patches the returned obs back into a batched tree.

```python
from env_ssl_wrapper import reset_env, reset_env_at, write_slot_obs

obs, info = reset_env(env, seed = 0)          # every slot replays seed 0
out = reset_env_at(env, 2, seed = 7)         # replay seed 7 on slot 2 only

if out is not None:
    slot_obs, _ = out
    obs = write_slot_obs(obs, slot_obs, 2)
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
