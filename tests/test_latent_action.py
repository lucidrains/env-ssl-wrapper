from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from env_ssl_wrapper import (
    ActionBankInterceptor,
    ActionInterceptor,
    LatentActionWrapper,
    MultiprocessingVecEnv,
    PastActionsWrapper,
    StandardizeEnvWrapper,
    StepTranslation,
    Translation,
    evaluate_actor,
)
from env_ssl_wrapper.mocks import DiscreteSpace

# deterministic mocks

class ContStepEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, obs_dim = 4, act_dim = 2, terminate_at = None, truncate_at = None, render_mode = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (obs_dim,), dtype = np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape = (act_dim,), dtype = np.float32)
        self.state = np.zeros(obs_dim, dtype = np.float32)
        self.t = 0

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        base = float(seed or 0) * 0.1
        self.state = np.full(self.obs_dim, base, dtype = np.float32)
        self.t = 0
        return self.state.copy(), {'reset_base': base}

    def step(self, action):
        act = np.asarray(action, dtype = np.float32)
        self.t += 1
        delta = np.pad(act, (0, max(0, self.obs_dim - self.act_dim)))[:self.obs_dim]
        self.state = self.state + delta + 0.01 * self.t
        reward = float(act.sum() + self.t)
        terminated = self.terminate_at == self.t
        truncated = (not terminated) and (self.truncate_at == self.t)
        return self.state.copy(), reward, terminated, truncated, {'prim_t': self.t}

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        return np.full((32, 32, 3), min(255, self.t * 20), dtype = np.uint8)

class DiscStepEnv(gym.Env):
    def __init__(self, num_actions = 4, terminate_at = None, truncate_at = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (3,), dtype = np.float32)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.state = np.zeros(3, dtype = np.float32)
        self.t = 0
        self.applied = []

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        self.t = 0
        self.applied = []
        self.state = np.full(3, float(seed or 0), dtype = np.float32)
        return self.state.copy(), {}

    def step(self, action):
        a = int(action)
        self.applied.append(a)
        self.t += 1
        self.state = self.state + np.array([1., float(a), float(self.t)], dtype = np.float32)
        reward = float(a + 1)
        terminated = self.terminate_at == self.t
        truncated = (not terminated) and (self.truncate_at == self.t)
        return self.state.copy(), reward, terminated, truncated, {'last_prim_a': a}

class LockstepVectorEnv:
    num_envs = 3
    is_vector = True
    action_space = DiscreteSpace(4)

    def __init__(self, terminate_at = (2, 10, 10)):
        self.terminate_at = np.asarray(terminate_at)
        self.t = np.zeros(3, dtype = int)

    def reset(self, **kwargs):
        self.t = np.zeros(3, dtype = int)
        return np.zeros((3, 2), dtype = np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.repeat(self.t[:, None], 2, axis = 1).astype(np.float32)
        reward = np.asarray(action, dtype = np.float32) + 1.
        terminated = self.t == self.terminate_at
        truncated = np.zeros(3, dtype = bool)
        return obs, reward, terminated, truncated, {}

def make_disc_env():
    return DiscStepEnv(num_actions = 4, terminate_at = 5)

# open-loop translation & reward modes

def test_callable_and_module_interceptors():
    proj = nn.Linear(3, 2, bias = False)
    with torch.no_grad():
        proj.weight.copy_(torch.tensor([[1., 0., -1.], [0., 0.5, 0.]]))

    env = LatentActionWrapper(ContStepEnv(), proj, latent_dim = 3, gamma = 0.9, clip_lower_actions = True)
    assert env.action_space.shape == (3,)

    obs, _ = env.reset(seed = 0)
    assert obs.shape == (4,)

    # projected [2.0, 0.5] clips to [1.0, 0.5]; t = 1 -> reward = 1.5 + 1 = 2.5
    obs, reward, term, trunc, info = env.step(np.array([2., 1., 0.], dtype = np.float32))
    assert np.isclose(reward, 2.5)
    assert np.isclose(info['raw_reward'], 2.5)
    assert info['duration'] == 1.0
    assert info['executed_steps'] == 1
    assert info['planned_steps'] == 1
    assert np.isclose(info['discount'], 0.9)
    assert info['aborted'] is False
    assert not term and not trunc

    # fixed-horizon chunk via plain callable: rewards [1, 2, 3], gamma = 0.5 -> 1 + 1 + 0.75 = 2.75
    env_chunk = LatentActionWrapper(ContStepEnv(), lambda z: z[:, None, :2].expand(-1, 3, -1), latent_dim = 2, gamma = 0.5)
    env_chunk.reset(seed = 0)
    _, reward, _, _, info = env_chunk.step([0.5, -0.5])
    assert np.isclose(reward, 2.75)
    assert np.isclose(info['raw_reward'], 6.0)
    assert np.isclose(info['last_reward'], 3.0)
    assert info['duration'] == 3.0
    assert np.isclose(info['discount'], 0.125)

def test_reward_modes_and_mid_chunk_early_stop():
    bank = ActionBankInterceptor(np.array([[0, 1, 2, 3], [3, 3, 3, 3]], dtype = np.int64))

    for mode, expected in (('sum', 1. + 0.9 * 2. + 0.81 * 3.), ('mean', 2.0), ('last', 3.0)):
        env = LatentActionWrapper(DiscStepEnv(terminate_at = 3), bank, gamma = 0.9, reward_mode = mode)
        env.reset()
        _, reward, term, trunc, info = env.step(0)

        assert term is True and trunc is False
        assert info['duration'] == 3.0
        assert info['executed_steps'] == 3
        assert info['planned_steps'] == 4
        assert np.isclose(info['raw_reward'], 6.0)
        assert np.isclose(reward, expected)
        assert 'final_observation' in info

    env_trunc = LatentActionWrapper(DiscStepEnv(truncate_at = 2), bank, gamma = 0.9)
    env_trunc.reset()
    _, _, term, trunc, info = env_trunc.step(1)
    assert term is False and trunc is True
    assert info['duration'] == 2.0
    assert np.isclose(info['raw_reward'], 8.0)

# discrete banks & accretive hierarchies

def test_action_bank_accretive_union_and_override():
    router = ActionBankInterceptor.from_base(4)
    assert router.counts == (4,)
    assert router.hierarchies == (1,)
    assert router.total_actions == 4

    vq2 = np.array([[1, 2], [3, 0]], dtype = np.int64)
    vq4 = np.array([[2, 2, 2, 2], [1, 3, 1, 3], [0, 0, 2, 2]], dtype = np.int64)

    router_u = router.accrete(vq2, mode = 'union', name = 'vq2').accrete(vq4, mode = 'union', name = 'vq4')
    assert router_u.counts == (4, 2, 3)
    assert router_u.hierarchies == (1, 2, 4)
    assert router_u.total_actions == 9

    base = DiscStepEnv()
    env = LatentActionWrapper(base, router_u, gamma = 0.99, trace_steps = True)
    env.reset()

    _, _, _, _, info0 = env.step(0)
    assert info0['horizon'] == 1 and info0['bank_id'] == 0 and info0['duration'] == 1.0

    _, _, _, _, info4 = env.step(4)
    assert info4['horizon'] == 2 and info4['bank_id'] == 1 and info4['duration'] == 2.0
    assert list(info4['latent_trace'].lower_actions) == [1, 2]

    _, _, _, _, info7 = env.step(7)
    assert info7['horizon'] == 4 and info7['bank_id'] == 2 and info7['duration'] == 4.0
    assert base.applied == [0, 1, 2, 1, 3, 1, 3]

    # override replaces the bank table while sharing the underlying lower_buffer
    router_ovr = router_u.accrete(vq2, mode = 'override')
    assert router_ovr.total_actions == 2
    assert router_ovr.num_lower_steps == router_u.num_lower_steps == 7

def test_shared_continuous_variable_horizon_upward_info():
    class SharedLengthInterceptor(ActionInterceptor):
        def __init__(self):
            super().__init__(latent_dim = 3, bounds = (-1., 1.))
            self.register_buffer('horizons', torch.tensor([1., 2., 4.]))

        def translate(self, z):
            pred_len = 1. + 1.5 * (z[:, 0] + 1.)
            idx = (pred_len[:, None] - self.horizons[None, :]).abs().argmin(dim = -1)
            legit_h = self.horizons[idx].long()
            prim = z[:, None, 1:3].expand(-1, 4, -1)
            return Translation(
                actions = prim,
                lengths = legit_h,
                info = dict(pred_length = pred_len, horizon = legit_h),
            )

    ic = SharedLengthInterceptor()
    envs = [ContStepEnv(terminate_at = 3 if i == 0 else 20) for i in range(3)]
    env = LatentActionWrapper(envs, ic, gamma = 0.9)

    obs, _ = env.reset(seed = 10)
    assert obs.shape == (3, 4)

    z = torch.tensor([
        [1.0, 0.2, 0.3],    # planned 4, terminates early at step 3
        [-1.0, -0.1, 0.1],  # planned 1
        [-0.33, 0.0, 0.5],  # planned 2
    ])
    next_obs, _, term, trunc, info = env.step(z)

    assert term.tolist() == [True, False, False]
    assert trunc.tolist() == [False, False, False]
    assert info['planned_steps'].tolist() == [4, 1, 2]
    assert info['executed_steps'].tolist() == [3, 1, 2]
    assert info['duration'].tolist() == [3.0, 1.0, 2.0]
    assert info['horizon'].tolist() == [4, 1, 2]
    assert np.allclose(info['pred_length'], [4.0, 1.0, 2.005], atol = 1e-2)
    assert not np.allclose(next_obs[0], info['final_observation'][0])

# vectorized per-slot smdp

def test_batched_single_forward_pass_on_sync_and_mp_vec_envs():
    call_batch_sizes = []

    class CountingInterceptor(ActionInterceptor):
        def __init__(self):
            super().__init__(num_actions = 3)
            self.seqs = torch.tensor([[0, 0, 0, 0], [1, 2, 0, 0], [3, 2, 1, 0]], dtype = torch.long)
            self.lens = torch.tensor([1, 2, 4], dtype = torch.long)

        def translate(self, action, obs):
            call_batch_sizes.append(int(obs.shape[0]))
            idx = action.long()
            return Translation(actions = self.seqs[idx], lengths = self.lens[idx], info = dict(horizon = self.lens[idx]))

    sync_raw = SyncVectorEnv([make_disc_env for _ in range(4)], autoreset_mode = AutoresetMode.SAME_STEP)
    env_sync = StandardizeEnvWrapper(LatentActionWrapper(sync_raw, CountingInterceptor(), gamma = 0.95))

    obs, _ = env_sync.reset(seed = 0)
    assert torch.is_tensor(obs) and obs.shape == (4, 3)

    _, rew, _, _, info = env_sync.step(torch.tensor([0, 1, 2, 1]))
    assert call_batch_sizes == [4]
    assert torch.is_tensor(rew) and rew.shape == (4,)
    assert info['duration'].tolist() == [1.0, 2.0, 4.0, 2.0]
    assert info['horizon'].tolist() == [1, 2, 4, 2]
    env_sync.close()

    call_batch_sizes.clear()
    mp_raw = MultiprocessingVecEnv(make_disc_env, num_envs = 3, seed = 0)
    env_mp = StandardizeEnvWrapper(LatentActionWrapper(mp_raw, CountingInterceptor(), gamma = 0.95))
    try:
        env_mp.reset()
        _, _, _, _, info = env_mp.step(torch.tensor([2, 0, 1]))
        assert call_batch_sizes == [3]
        assert info['duration'].tolist() == [4.0, 1.0, 2.0]
    finally:
        env_mp.close()

def test_lockstep_vector_env_uniform_vs_heterogeneous():
    uniform_bank = ActionBankInterceptor(np.array([[0, 1, 2], [2, 1, 0]], dtype = np.int64))
    env = LatentActionWrapper(LockstepVectorEnv(terminate_at = (2, 10, 10)), uniform_bank, gamma = 1.0)
    env.reset()

    _, _, term, _, info = env.step([0, 1, 0])
    assert info['executed_steps'].tolist() == [2, 2, 2]
    assert term.tolist() == [True, False, False]

    mixed_bank = ActionBankInterceptor.from_base(4).accrete(np.array([[0, 1, 2]], dtype = np.int64))
    env_mixed = LatentActionWrapper(LockstepVectorEnv(), mixed_bank)
    env_mixed.reset()

    with pytest.raises(ValueError, match = 'per-slot stepping'):
        env_mixed.step([0, 4, 1])

def test_raw_sync_vector_env_and_offline_rollout():
    router = ActionBankInterceptor({1: np.array([[0], [1]]), 3: np.array([[1, 2, 3]])})
    assert router.horizons_np.tolist() == [1, 1, 3]
    assert router.sequences == ((0,), (1,), (1, 2, 3))

    raw_sync = SyncVectorEnv([make_disc_env for _ in range(3)], autoreset_mode = AutoresetMode.SAME_STEP)
    env = LatentActionWrapper(raw_sync, router, gamma = 0.9)
    env.reset(seed = 0)

    _, _, _, _, info = env.step([0, 2, 2])
    assert info['duration'].tolist() == [1.0, 3.0, 3.0]

    _, _, term, _, info = env.step([0, 2, 2])
    assert term.tolist() == [False, True, True]
    assert info['duration'].tolist() == [1.0, 2.0, 2.0]
    assert info['final_observation'].shape == (3, 3)
    raw_sync.close()

    router.clear_buffers()
    fake_obs = torch.zeros((2, 6, 3))
    fake_acts = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 1, 1, 2, 2, 2]])
    fake_dones = torch.tensor([[False, False, True, False, False, False], [False] * 6])
    router.add_lower_rollout(fake_obs, fake_acts, dones = fake_dones)

    c4 = router.lower_chunks(4)
    assert c4.actions.shape == (3, 4)

# steering, mid-chunk abort & closed-loop options

def test_state_conditioned_steering_abort_and_eval_mode():
    class SteerableAbortInterceptor(ActionInterceptor):
        def __init__(self):
            super().__init__(latent_dim = 2)
            self.net = nn.Linear(4, 2)
            with torch.no_grad():
                self.net.weight.zero_()
                self.net.bias.fill_(0.25)

        def translate(self, action, obs, *, eval_mode = False, zero_steer = False):
            base = self.net(obs)
            steer = torch.zeros_like(base) if (zero_steer or action is None) else action
            train_jitter = 0.0 if (eval_mode or not self.training) else 0.02
            chunk = (base + steer + train_jitter)[:, None, :].expand(-1, 4, -1)
            planned_val = torch.full((obs.shape[0],), 10.0, device = obs.device)
            return Translation(
                actions = chunk,
                lengths = torch.full((obs.shape[0],), 4, dtype = torch.long, device = obs.device),
                info = dict(steered = action is not None and not zero_steer, jitter = train_jitter),
                state = planned_val,
            )

        def should_abort(self, obs, remaining_actions, remaining_lengths, *, step, **kwargs):
            return (obs[:, 0] > 0.6) if step == 2 else None

    ic = SteerableAbortInterceptor()
    ic.train()

    env = LatentActionWrapper(ContStepEnv(), ic, gamma = 0.99)
    env.reset(seed = 0)
    _, _, _, _, info_prior = env.step(None)
    assert info_prior['steered'] is False
    assert info_prior['duration'] == 4.0

    env.reset(seed = 0)
    _, _, _, _, info_abort = env.step(np.array([0.5, 0.0], dtype = np.float32))
    assert info_abort['steered'] is True
    assert info_abort['aborted'] is True
    assert info_abort['executed_steps'] == 2
    assert info_abort['planned_steps'] == 4
    assert ic.num_paired_steps == 1

    prev_lower = ic.num_lower_steps
    eval_env = LatentActionWrapper(ContStepEnv(), ic, eval_mode = True)
    eval_env.reset(seed = 0)
    _, _, _, _, info_eval = eval_env.step(np.array([0.1, -0.1], dtype = np.float32), zero_steer = True)
    assert info_eval['jitter'] == 0.0
    assert ic.training is True
    assert ic.num_lower_steps == prev_lower

def test_closed_loop_option_dynamic_termination():
    seen_steps = []

    class ClosedLoopOption(ActionInterceptor):
        def __init__(self):
            super().__init__(num_actions = 2, horizon = 5)

        def step_lower(self, action, obs, step = 0):
            seen_steps.append((step, float(obs[0, 0].item())))
            prim_a = torch.where(action == 0, torch.ones_like(action), torch.full_like(action, 2))
            opt_done = (action == 0) & (obs[:, 0] >= 1.0)
            return StepTranslation(action = prim_a, done = opt_done, info = dict(chosen_opt = action))

    env = LatentActionWrapper(DiscStepEnv(), ClosedLoopOption(), gamma = 0.9)
    env.reset(seed = 0)

    _, _, _, _, info = env.step(0)
    assert info['executed_steps'] == 2
    assert info['planned_steps'] == 5
    assert [s for s, _ in seen_steps] == [0, 1]
    assert np.isclose(seen_steps[1][1] - seen_steps[0][1], 1.0)

# stacked multi-level hierarchies

def test_two_level_stacked_hierarchy_smdp_parity():
    l1_bank = ActionBankInterceptor({
        1: np.array([[3]], dtype = np.int64),
        2: np.array([[1, 2]], dtype = np.int64),
        3: np.array([[0, 2, 1]], dtype = np.int64),
    })
    l2_bank = ActionBankInterceptor({
        2: np.array([[1, 2]], dtype = np.int64),     # [1, 2] then [0, 2, 1] -> 5 prim steps
        3: np.array([[0, 1, 0]], dtype = np.int64),  # [3], [1, 2], [3] -> 4 prim steps
    })

    gamma = 0.93
    lvl1 = LatentActionWrapper([DiscStepEnv(terminate_at = 4 if i == 1 else 50) for i in range(2)], l1_bank, gamma = gamma, autoreset = False)
    lvl2 = LatentActionWrapper(lvl1, l2_bank, gamma = gamma, autoreset = False)
    envs_ref = [DiscStepEnv(terminate_at = 4 if i == 1 else 50) for i in range(2)]

    lvl2.reset(seed = 5)
    for i, e in enumerate(envs_ref):
        e.reset(seed = 5 + i)

    obs_h, disc_r_h, term_h, _, info_h = lvl2.step([0, 0])

    for i in range(2):
        ref_disc, ref_raw, ref_steps = 0., 0., 0
        for t_idx, pa in enumerate([1, 2, 0, 2, 1]):
            ref_obs, r, tm, tr, _ = envs_ref[i].step(pa)
            ref_disc += (gamma ** t_idx) * r
            ref_raw += r
            ref_steps += 1
            if tm or tr:
                break

        assert np.allclose(obs_h[i], ref_obs)
        assert np.isclose(disc_r_h[i], ref_disc, atol = 1e-5)
        assert np.isclose(info_h['raw_reward'][i], ref_raw, atol = 1e-5)
        assert np.isclose(info_h['duration'][i], ref_steps)
        assert np.isclose(info_h['discount'][i], gamma ** ref_steps, atol = 1e-5)
        assert bool(term_h[i]) == tm

    assert info_h['duration'].tolist() == [5.0, 4.0]
    assert info_h['executed_steps'].tolist() == [2, 2]
    assert info_h['lower_info']['duration'].tolist() == [5.0, 4.0]

# stream & paired buffers

def test_episode_boundary_isolation_and_partial_chunks():
    bank = ActionBankInterceptor.from_base(4, buffer_capacity = 64)
    env = LatentActionWrapper(DiscStepEnv(terminate_at = 3), bank, passthrough = True)

    env.reset(seed = 1)
    for a in (0, 1, 2):
        env.step(a)

    env.reset(seed = 2)
    env.step(3)
    env.step(3)

    env.reset(seed = 3)
    for a in (1, 1, 1):
        env.step(a)

    assert bank.lower_buffer.num_chunks(4, allow_partial = False) == 0

    c3 = bank.lower_chunks(3, allow_partial = False)
    assert c3.actions.tolist() == [[0, 1, 2], [1, 1, 1]]
    assert (c3.valid_lens == 3).all() and c3.mask.all()

    cp = bank.lower_chunks(4, stride = 1, allow_partial = True, pad_value = -1)
    for row_a, vlen, m_row in zip(cp.actions, cp.valid_lens.tolist(), cp.mask):
        assert 1 <= vlen <= 3
        assert m_row[:vlen].all() and not m_row[vlen:].any()
        assert (row_a[vlen:] == -1).all()

    mh = bank.sample_lower_chunks(12, chunk_len = (1, 2, 3), pad_value = -9)
    assert mh.actions.shape == (12, 3)
    assert set(mh.valid_lens.tolist()) == {1, 2, 3}

def test_ring_buffer_wraparound_reset_at_and_recent_frac():
    bank = ActionBankInterceptor.from_base(4, buffer_capacity = 64)
    env = LatentActionWrapper([DiscStepEnv() for _ in range(2)], bank, passthrough = True)
    env.reset(seed = 0)

    for _ in range(50):
        env.step([0, 0])

    env.reset_at(0, seed = 99)
    for _ in range(16):
        env.step([3, 2])

    for seq in bank.lower_chunks(4).actions.tolist():
        assert not (0 in seq and 3 in seq), f'chunk crossed reset_at boundary: {seq}'

    for seq in bank.lower_chunks(4, recent_frac = 0.25).actions.tolist():
        assert seq in ([3, 3, 3, 3], [2, 2, 2, 2])

def test_accretive_primitive_chunk_harvesting_across_horizons():
    router = ActionBankInterceptor.from_base(4).accrete(np.array([[1, 2], [3, 0]], dtype = np.int64))
    env = LatentActionWrapper(DiscStepEnv(terminate_at = 5), router, gamma = 0.99)
    env.reset(seed = 0)

    env.step(0)
    env.step(4)
    _, _, term, _, _ = env.step(5)
    assert term is True

    env.reset(seed = 1)
    for _ in range(3):
        env.step(2)

    harvested = router.lower_chunks(4, allow_partial = False)
    assert harvested.actions.tolist() == [[0, 1, 2, 3], [1, 2, 3, 0]]

def test_stage1_passthrough_autoencoder_to_stage2_and_super_chunks():
    torch.manual_seed(0)

    class TanhAutoencoderInterceptor(ActionInterceptor):
        def __init__(self, chunk_len = 3, act_dim = 2, latent_dim = 2):
            super().__init__(latent_dim = latent_dim, passthrough = True, buffer_capacity = 4096)
            self.chunk_len = chunk_len
            flat_dim = chunk_len * act_dim
            self.encoder = nn.Sequential(nn.Linear(flat_dim, 32), nn.SiLU(), nn.Linear(32, latent_dim), nn.Tanh())
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, flat_dim), nn.Tanh())

        def translate(self, z):
            return rearrange(self.decoder(z), 'b (h a) -> b h a', h = self.chunk_len)

        def loss(self, batch_size = 64):
            batch = self.sample_lower_chunks(batch_size, self.chunk_len)
            z = self.encoder(rearrange(batch.actions, 'b h a -> b (h a)'))
            return F.mse_loss(self.translate(z), batch.actions)

    ic = TanhAutoencoderInterceptor()
    env = LatentActionWrapper(StandardizeEnvWrapper(ContStepEnv()), ic, gamma = 0.99)

    assert env.action_space.shape == (2,)
    env.reset(seed = 0)
    patterns = (torch.tensor([[0.5, -0.5]]), torch.tensor([[-0.5, 0.5]]))
    for step_i in range(90):
        env.step(patterns[(step_i // 3) % 2])

    opt = torch.optim.Adam(ic.parameters(), lr = 1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = ic.loss(batch_size = 32)
        loss.backward()
        opt.step()

    assert float(loss.item()) < 0.02

    env.passthrough = False
    env.reset(seed = 1)
    for _ in range(10):
        _, _, _, _, info = env.step(torch.randn(1, 2).clamp(-1., 1.))
        assert info['duration'].item() == 3.0

    super_batch = ic.sample_higher_chunks(8, chunk_len = 4, allow_partial = True)
    assert super_batch.actions.shape == (8, 4, 2)
    assert (super_batch.durations[:, 0] == 3.0).all()

def test_online_paired_learning_and_dict_observations():
    torch.manual_seed(0)
    skill_vecs = torch.tensor([[0.8, 0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]])
    hook_counts = []

    class SkillInterceptor(ActionInterceptor):
        def __init__(self):
            super().__init__(num_actions = 4, buffer_capacity = 2048)
            self.disc = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 4))

        def translate(self, skill_id, obs):
            assert isinstance(obs, dict) and 'obs' in obs and 'past_action' in obs
            return skill_vecs[skill_id.long()][:, None, :].expand(-1, 2, -1)

        def on_step(self, batch):
            hook_counts.append(int(batch.higher_action.shape[0]))

    ic = SkillInterceptor()
    base = PastActionsWrapper(StandardizeEnvWrapper(ContStepEnv(obs_dim = 4, act_dim = 2)))
    env = LatentActionWrapper(base, ic, gamma = 0.99)

    env.reset(seed = 0)
    for i in range(80):
        env.step(torch.tensor([i % 4]))

    assert len(hook_counts) == 80
    paired = ic.paired_data()
    assert isinstance(paired.obs, dict) and paired.lower_actions.shape == (80, 2, 2) and paired.step_mask.all()

    opt = torch.optim.Adam(ic.disc.parameters(), lr = 2e-2)
    for _ in range(100):
        b = ic.sample_paired(32)
        loss = F.cross_entropy(ic.disc(b.next_obs['obs'] - b.obs['obs']), b.higher_action.long())
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        all_b = ic.paired_data()
        acc = (ic.disc(all_b.next_obs['obs'] - all_b.obs['obs']).argmax(dim = -1) == all_b.higher_action.long()).float().mean().item()
    assert acc > 0.95

def test_transdimensional_dict_higher_actions():
    class TransDimOptionInterceptor(ActionInterceptor):
        def __init__(self):
            space = gym.spaces.Dict({
                'h': gym.spaces.Discrete(2),
                'z': gym.spaces.Box(-1., 1., shape = (2,), dtype = np.float32),
            })
            super().__init__(action_space = space, buffer_capacity = 512)
            self.register_buffer('bank_horizons', torch.tensor([1, 3], dtype = torch.long))

        def translate(self, action, obs):
            h_idx = action['h'].long()
            lengths = self.bank_horizons[h_idx]
            scale = torch.where(h_idx[:, None] == 0, 1.0, 0.5)
            prim = (action['z'] * scale)[:, None, :].expand(-1, 3, -1)
            return Translation(actions = prim, lengths = lengths, info = dict(horizon = lengths))

    env_single = LatentActionWrapper(ContStepEnv(), TransDimOptionInterceptor(), gamma = 0.9)
    env_single.reset(seed = 0)
    _, _, _, _, info_s = env_single.step({'h': 1, 'z': np.array([0.4, -0.4], dtype = np.float32)})
    assert info_s['duration'] == 3.0 and info_s['horizon'] == 3

    ic_vec = TransDimOptionInterceptor()
    vec_env = StandardizeEnvWrapper(LatentActionWrapper([ContStepEnv() for _ in range(2)], ic_vec, gamma = 0.9))
    vec_env.reset(seed = 0)

    for step_i in range(6):
        act = {'h': torch.tensor([step_i % 2, (step_i + 1) % 2]), 'z': torch.tensor([[0.2, -0.2], [-0.3, 0.3]])}
        _, _, _, _, inf = vec_env.step(act)
        assert inf['duration'].tolist() == ([1., 3.] if step_i % 2 == 0 else [3., 1.])

    h_chunks = ic_vec.sample_higher_chunks(4, chunk_len = 2)
    assert h_chunks.actions['h'].shape == (4, 2) and h_chunks.actions['h'].dtype == torch.long
    assert h_chunks.actions['z'].shape == (4, 2, 2) and h_chunks.actions['z'].dtype == torch.float32

    paired = ic_vec.sample_paired(4)
    assert paired.higher_action['h'].shape == (4,) and paired.higher_action['z'].shape == (4, 2)

# composition, evaluation & readme

def test_evaluate_actor_and_substep_frames(tmp_path):
    env = StandardizeEnvWrapper(ContStepEnv(terminate_at = 6, render_mode = 'rgb_array'))
    zero_chunk_ic = ActionInterceptor(lambda z: torch.zeros((z.shape[0], 3, 2), device = z.device), num_actions = 1)
    wrapped = LatentActionWrapper(env, zero_chunk_ic, gamma = 1.0, eval_mode = True, record_frames = True)

    wrapped.reset(seed = 0)
    _, _, _, _, step_info = wrapped.step(torch.tensor([0]))
    assert len(step_info['substep_frames']) == 3

    vid_file = tmp_path / 'macro_eval.mp4'
    stats = evaluate_actor(
        lambda obs: torch.zeros((obs.shape[0],), dtype = torch.long),
        wrapped,
        episodes = 2,
        seed = 0,
        video_path = str(vid_file),
        video_episodes = 1,
    )

    assert np.allclose(stats.returns.numpy(), [21.0, 21.0])
    assert np.allclose(stats.lengths.numpy(), [6.0, 6.0])
    assert vid_file.exists()

def test_standardize_composition_state_norm_time_limit_and_sgs():
    router = ActionBankInterceptor({1: np.array([[0], [1]]), 2: np.array([[0, 1], [1, 0]])})
    sync_cart = SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(3)], autoreset_mode = AutoresetMode.SAME_STEP)
    env = StandardizeEnvWrapper(
        LatentActionWrapper(sync_cart, router, gamma = 0.99),
        normalize_state = dict(update_mode = 'deferred'),
        max_timesteps = 15,
    )

    obs, _ = env.reset(seed = 42)
    assert obs.shape == (3, 4)

    for _ in range(8):
        obs, rew, _, _, _ = env.step(torch.tensor([0, 2, 3]))
        assert obs.shape == (3, 4) and rew.shape == (3,)

    stats = env.commit_stats()
    assert stats['obs'].count >= 1 + 3 * 9

    reset_bank = [torch.zeros(4), torch.full((4,), 0.02)]
    sgs_base = StandardizeEnvWrapper(
        gym.make('CartPole-v1'),
        reset_configs = reset_bank,
        sgs_kwargs = dict(set_state_fn = lambda env, state, cfg_id, obs: obs),
    )
    sgs_env = LatentActionWrapper(sgs_base, ActionBankInterceptor(np.array([[0, 0], [1, 1]])), gamma = 0.99)
    _, info_sgs = sgs_env.reset(seed = 0)
    assert 'sgs_config_id' in info_sgs

    _, _, _, _, info_step = sgs_env.step(torch.tensor([0]))
    assert 'sgs_config_id' in info_step and info_step['duration'].item() >= 1.0

    sync_cart.close()
    sgs_env.close()

def test_readme_latent_action_snippets():
    # 1. linear projection
    decoder = nn.Linear(4, 1)
    env = StandardizeEnvWrapper(gym.make('Pendulum-v1'))
    env = LatentActionWrapper(env, decoder, latent_dim = 4, bounds = (-1., 1.), clip_lower_actions = True)
    obs, _ = env.reset(seed = 0)
    obs, reward, _, _, info = env.step(torch.randn(1, 4))
    assert obs.shape == (1, 3) and reward.shape == (1,) and info['duration'].item() == 1.0
    env.close()

    # 2. discrete skill bank + accretion
    skills = ActionBankInterceptor.from_base(2).accrete(torch.tensor([[0, 0], [1, 1]]))
    env = LatentActionWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), skills, gamma = 0.99)
    env.reset(seed = 0)
    obs, _, _, _, info = env.step(torch.tensor([2]))
    assert info['horizon'].item() == 2 and info['bank_id'].item() == 1
    env.close()

    # 3. two-stage passthrough -> latent
    class ChunkAutoencoder(ActionInterceptor):
        def __init__(self):
            super().__init__(latent_dim = 4, passthrough = True)
            self.decoder = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        def translate(self, z):
            return rearrange(self.decoder(z), 'b (h a) -> b h a', h = 4)

    interceptor = ChunkAutoencoder()
    env = LatentActionWrapper(StandardizeEnvWrapper(gym.make('Pendulum-v1')), interceptor, gamma = 0.99)
    env.reset(seed = 0)
    for _ in range(16):
        env.step(torch.randn(1, 1))

    batch = interceptor.sample_lower_chunks(batch_size = 8, chunk_len = 4)
    assert batch.actions.shape == (8, 4, 1)

    env.passthrough = False
    env.step(torch.randn(1, 4))
    paired = interceptor.sample_paired(batch_size = 1)
    assert paired.lower_actions.shape == (1, 4, 1)
    env.close()

    # 4. state-conditioned adaptive skill with mid-chunk abort
    class AdaptiveSkill(ActionInterceptor):
        def __init__(self):
            super().__init__(latent_dim = 3)
            self.net = nn.Linear(3 + 3, 4)

        def translate(self, z, obs):
            actions = rearrange(self.net(torch.cat((z, obs), dim = -1)), 'b (h a) -> b h a', h = 4)
            lengths = torch.where(z[:, 0] > 0., 4, 2)
            return Translation(actions = actions, lengths = lengths, info = dict(chosen_h = lengths))

        def should_abort(self, obs, remaining_actions, remaining_lengths, *, step, **kwargs):
            return obs[:, 0].abs() > 0.9

    env = LatentActionWrapper(StandardizeEnvWrapper(gym.make('Pendulum-v1')), AdaptiveSkill(), gamma = 0.99, clip_lower_actions = True)
    env.reset(seed = 0)
    _, _, _, _, info = env.step(torch.ones(1, 3))
    assert info['chosen_h'].item() == 4
    env.close()

def test_compose_env_latent_action_and_mp_step_at():
    from env_ssl_wrapper import compose_env

    decoder = nn.Linear(3, 2)
    env = compose_env(
        ContStepEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        ('latent_action', dict(interceptor = decoder, latent_dim = 3)),
    )
    obs, _ = env.reset(seed = 0)
    assert obs.shape == (1, 4)
    obs, rew, _, _, info = env.step(torch.zeros(1, 3))
    assert obs.shape == (1, 4) and info['duration'].item() == 1.0

    with MultiprocessingVecEnv(make_disc_env, num_envs = 3, seed = 0) as mp_env:
        mp_env.reset()
        obs_sub, rew_sub, term_sub, trunc_sub, _ = mp_env.step_at([0, 2], np.array([1, 2]))
        assert obs_sub.shape == (2, 3) and rew_sub.shape == (2,)
        assert term_sub.shape == (2,) and trunc_sub.shape == (2,)

def test_latent_action_reward_mode_recording_stacked_gamma_and_edge_cases():
    # reward_mode = 'mean' is stored consistently in higher_buffer and paired_buffer
    bank = ActionBankInterceptor(np.array([[0, 2]], dtype = np.int64))
    env = LatentActionWrapper(DiscStepEnv(), bank, gamma = 0.9, reward_mode = 'mean')
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert np.isclose(reward, 2.0)
    assert np.isclose(bank.paired_data().reward.item(), 2.0)
    assert np.isclose(bank.paired_data().raw_reward.item(), 4.0)
    assert np.isclose(bank.higher_chunks(1).rewards.item(), 2.0)

    # outer gamma = 1.0 preserves inner layer's discounted reward when stacked
    l1 = ActionBankInterceptor(np.array([[0, 1]], dtype = np.int64))
    l2 = ActionBankInterceptor(np.array([[0, 0]], dtype = np.int64))
    inner = LatentActionWrapper(DiscStepEnv(), l1, gamma = 0.5)
    outer = LatentActionWrapper(inner, l2, gamma = 1.0)
    outer.reset()
    _, rew_outer, _, _, _ = outer.step(0)
    # each inner step returns 1 + 0.5 * 2 = 2.0; two inner steps with outer gamma = 1.0 -> 4.0
    assert np.isclose(rew_outer, 4.0)

    # unspecified higher space on discrete inner env still accepts unbatched continuous latent vectors
    env_unspec = LatentActionWrapper(DiscStepEnv(), lambda z: (z[:, 0] > 0).long())
    env_unspec.reset()
    _, _, _, _, info_u = env_unspec.step(np.array([1.0, -1.0, 0.5], dtype = np.float32))
    assert info_u['duration'] == 1.0

    # mixed float and int banks raise AssertionError instead of silently truncating
    with pytest.raises(AssertionError):
        ActionBankInterceptor([np.array([[0, 1]], dtype = np.int64), np.array([[0.5, 0.5]], dtype = np.float32)])

    # ragged / non-numeric inner info fields do not crash stacked info aggregation
    class RaggedInfoEnv(DiscStepEnv):
        def step(self, action):
            obs, r, tm, tr, info = super().step(action)
            info['ragged'] = [1] * self.t
            return obs, r, tm, tr, info

    lvl1 = LatentActionWrapper(RaggedInfoEnv(), ActionBankInterceptor(np.array([[0, 1]], dtype = np.int64)), gamma = 0.9)
    lvl2 = LatentActionWrapper(lvl1, ActionBankInterceptor(np.array([[0, 0]], dtype = np.int64)), gamma = 0.9)
    lvl2.reset()
    _, _, _, _, info_rag = lvl2.step(0)
    assert 'lower_info' in info_rag and 'ragged' not in info_rag['lower_info']
