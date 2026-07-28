## env-ssl-wrapper (wip)

Some handy wrappers around envs for now

## Install

```bash
pip install env-ssl-wrapper
```

## Usage

Compose environments seamlessly with `compose_env`:

```python
import torch
import gymnasium as gym
from env_ssl_wrapper import compose_env

env = compose_env(
    gym.make('Pendulum-v1', render_mode = 'rgb_array'),
    ('image', dict(image_size = (64, 64))),
    ('action_transform', dict(
        transforms = dict(rescale_from_to = ((0.0, 1.0), (-2.0, 2.0))),
        clip = (-2.0, 2.0)
    )),
    'auto_batch',
    ('tensor', dict(device = 'cuda' if torch.cuda.is_available() else 'cpu')),
    'done_tracker'
)

# Standard rollout loop — observations are PyTorch GPU tensors,
# and episode lengths are tracked per environment for easy replay buffer insertion

obs, info = env.reset()

while not env.needs_reset:
    actions = policy(obs['image'])
    obs, reward, terminated, truncated, info = env.step(actions)

# Per-environment episode step counts ready for replay buffer
episode_lengths = env.episode_lengths # array of shape (8,)
```

## Wrappers

### Done & Episode Length Tracking (`done_tracker`)

Standardizes `terminated`, `truncated`, and `dones` tracking across vectorized environments while maintaining per-environment `episode_lengths`:

```python
env = compose_env(
    gym.make_vec('CartPole-v1', num_envs = 16),
    ('tensor', dict(device = 'cpu')),
    'done_tracker'
)

obs, info = env.reset() # obs.shape: (16, 4)

while not env.needs_reset:
    actions = model(obs)
    obs, reward, terminated, truncated, info = env.step(actions)

# episode lengths tracked per environment for replay buffer insertion
print(env.episode_lengths) # shape: (16,)
```

### Auto Batching (`auto_batch`)

Ensures single non-vectorized environments output and receive leading batch dimensions seamlessly:

```python
env = compose_env(
    gym.make('CartPole-v1'),
    'auto_batch'
)

obs, info = env.reset() # obs.shape: (1, 4)
```

### Tensor Conversion (`tensor`)

Converts all numpy observations and rewards to PyTorch tensors on your target device, and action tensors back to numpy arrays:

```python
env = compose_env(
    gym.make('CartPole-v1'),
    'auto_batch',
    ('tensor', dict(device = 'cuda'))
)
```

## Citations

```bibtex
@misc{schwarzer2021dataefficientreinforcementlearningselfpredictive,
    title   = {Data-Efficient Reinforcement Learning with Self-Predictive Representations},
    author  = {Max Schwarzer and Ankesh Anand and Rishab Goel and R Devon Hjelm and Aaron Courville and Philip Bachman},
    year    = {2021},
    eprint  = {2007.05929},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2007.05929},
}
```

```bibtex
@misc{schmidt2024learningactactions,
    title   = {Learning to Act without Actions},
    author  = {Dominik Schmidt and Minqi Jiang},
    year    = {2024},
    eprint  = {2312.10812},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2312.10812},
}
```

```bibtex
@misc{eysenbach2023contrastivelearninggoalconditionedreinforcement,
    title   = {Contrastive Learning as Goal-Conditioned Reinforcement Learning},
    author  = {Benjamin Eysenbach and Tianjun Zhang and Ruslan Salakhutdinov and Sergey Levine},
    year    = {2023},
    eprint  = {2206.07568},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2206.07568},
}
```

```bibtex
@misc{ashlag2025stateentropyregularizationrobust,
    title   = {State Entropy Regularization for Robust Reinforcement Learning},
    author  = {Yonatan Ashlag and Uri Koren and Mirco Mutti and Esther Derman and Pierre-Luc Bacon and Shie Mannor},
    year    = {2025},
    eprint  = {2506.07085},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2506.07085},
}
```

```bibtex
@inproceedings{park2026dual,
    title   = {Dual Goal Representations},
    author  = {Seohong Park and Deepinder Mann and Sergey Levine},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    year    = {2026},
    url     = {https://openreview.net/forum?id=aMKFTidLSM}
}
```

```bibtex
@misc{almuzairee2026squintfastvisualreinforcement,
    title   = {Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics},
    author  = {Abdulaziz Almuzairee and Henrik I. Christensen},
    year    = {2026},
    eprint  = {2602.21203},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url     = {https://arxiv.org/abs/2602.21203},
}
```
