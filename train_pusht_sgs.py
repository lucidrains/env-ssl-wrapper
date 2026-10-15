# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium",
#     "gym-pusht",
#     "imageio",
#     "imageio[ffmpeg]",
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "torch",
#     "x-mlps-pytorch",
#     "x-ppo>=0.0.7",
# ]
# ///
#
# Success-Guided Sampling on Push-T - an adaptive curriculum over reset states
#
#   uv run train_pusht_sgs.py                              # 2 arms x 3 seeds x 1M steps
#   uv run train_pusht_sgs.py --steps=100000 --seeds=0     # quick look
#   uv run train_pusht_sgs.py --method sgs --seed 0        # a single run (used by --parallel workers)
#
# the bank holds 16 reset configurations: 10 near-goal (easy) + 6 standoff front (hard).
# both arms share the learner, the step budget and the evaluation protocol - only the
# reset distribution differs
#
#   uniform - uniform sampling over the bank of reset states
#   sgs     - success-guided sampling over that same bank, chasing the frontier
#
# everything lands in experiments/pusht_sgs/: results.json, per-run eval curves, the
# progression of the bank per run (sgs_progression_*.csv), policy checkpoints, the
# learning-curve chart and rendered solves of the best SGS policy

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import fire
import gymnasium as gym
import gym_pusht  # noqa: F401
from gym_pusht.envs.pusht import PushTEnv
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Beta
import torch.nn as nn
import torch.nn.functional as F
from x_mlps_pytorch import MLP
from x_ppo import calc_gae, ppo_actor_loss

from env_ssl_wrapper import (
    SGSSampler,
    StandardizeEnvWrapper,
    SuccessGuidedSamplingWrapper,
    uniform_score_fn,
)

# fast exact PushT step (OpenCV convex intersection replaces shapely for 3.5x speedup)

RECT1 = np.array([(-60., 0.), (60., 0.), (60., 30.), (-60., 30.)], dtype = np.float32)
RECT2 = np.array([(-15., 30.), (15., 30.), (15., 120.), (-15., 120.)], dtype = np.float32)
KP_LOC = torch.tensor([[-60., 15.], [60., 15.], [0., 0.], [0., 120.]])

def get_rects(x, y, th):
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, s], [-s, c]], dtype = np.float32)
    pos = np.array([x, y], dtype = np.float32)
    return RECT1 @ rot + pos, RECT2 @ rot + pos

GOAL_TH = float(np.pi / 4)
G_R1, G_R2 = get_rects(256., 256., GOAL_TH)
GOAL_COM = np.array([256. - 45. * np.sin(GOAL_TH), 256. + 45. * np.cos(GOAL_TH)], dtype = np.float32)

def fast_step(self, action):
    ax, ay = float(action[0]), float(action[1])
    dt, kp, kv = self.dt, self.k_p, self.k_v
    for _ in range(int(1 / (dt * self.control_hz))):
        px, py = self.agent.position
        vx, vy = self.agent.velocity
        self.agent.velocity = (vx + (kp * (ax - px) - kv * vx) * dt, vy + (kp * (ay - py) - kv * vy) * dt)
        self.space.step(dt)
    bx, by = self.block.position
    ba = self.block.angle
    r1, r2 = get_rects(bx, by, ba)
    area = sum(cv2.intersectConvexConvex(g, r)[0] for g in (G_R1, G_R2) for r in (r1, r2))
    cov = float(area) / 6300.
    is_success = cov > self.success_threshold
    obs = np.array([self.agent.position[0], self.agent.position[1], bx, by, ba % (2 * np.pi)], dtype = np.float64)
    return obs, min(max(cov / self.success_threshold, 0.), 1.), is_success, False, dict(is_success = is_success, coverage = cov)

def make_pusht(max_steps = 50, seed = None):
    env = gym.make('gym_pusht/PushT-v0')
    env.unwrapped.step = fast_step.__get__(env.unwrapped, PushTEnv)
    env = StandardizeEnvWrapper(env, auto_batch = False, device = None, done_tracker = False, pad_episodes = False, max_timesteps = max_steps)

    if seed is not None:
        env.reset(seed = seed)
        env.action_space.seed(seed)

    return env

def build_configs():
    configs, c4, s4 = [], np.cos(GOAL_TH), np.sin(GOAL_TH)
    # 10 near-goal configs (easy) + 6 standoff reaching & steering frontier configs (hard)
    specs = [
        (3., 0.00, 0., 21.), (4., -0.01, 0., 21.), (5., 0.01, 0., 21.), (6., -0.02, 0., 21.), (7., 0.02, 0., 21.),
        (8., -0.02, 0., 22.), (9., 0.03, 0., 22.), (10., -0.03, 0., 22.), (12., 0.04, 0., 22.), (14., -0.04, 0., 22.),
        (34., 0.16, -24., 48.), (40., -0.17, 24., 52.), (46., 0.19, -26., 56.),
        (54., -0.20, 26., 60.), (62., 0.22, -28., 64.), (70., -0.24, 28., 68.),
    ]
    for d, dth, lat, clear in specs:
        du, dv = 0.47 * d * np.sin(dth), -d * np.cos(0.5 * dth)
        u_a, v_a = np.clip(-3000. * dth / d + lat, -48., 48.), -clear
        t_com = GOAL_COM + np.array([du * c4 - dv * s4, du * s4 + dv * c4])
        th = float(GOAL_TH + dth)
        bx, by = t_com[0] + 45. * np.sin(th), t_com[1] - 45. * np.cos(th)
        ax, ay = bx + np.cos(th) * u_a - np.sin(th) * v_a, by + np.sin(th) * u_a + np.cos(th) * v_a
        configs.append(np.array([ax, ay, bx - 45. * np.sin(th), by - 45. * (1. - np.cos(th)), th], dtype = np.float32))
    return configs

NUM_EASY = 10

def get_kp(pos: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(th).unsqueeze(-1), torch.sin(th).unsqueeze(-1)
    u, v = KP_LOC[:, 0], KP_LOC[:, 1]
    return torch.cat([pos[..., 0:1] + c * u - s * v, pos[..., 1:2] + s * u + c * v], dim = -1)

KP_GOAL = get_kp(torch.tensor([[256., 256.]]), torch.tensor([GOAL_TH])).squeeze(0)

def featurize(raw: torch.Tensor) -> torch.Tensor:
    raw = raw.float()
    pa, pb, th = raw[..., 0:2], raw[..., 2:4], raw[..., 4]
    c, s = torch.cos(th), torch.sin(th)
    cb = torch.stack([pb[..., 0] - 45. * s, pb[..., 1] + 45. * c], dim = -1)
    g_com = torch.tensor(GOAL_COM)
    d_ab, d_cg = cb - pa, g_com - cb
    e1 = d_ab / (d_ab.norm(dim = -1, keepdim = True) + 1e-6)
    e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim = -1)
    cg_ab = torch.stack([(d_cg * e1).sum(-1), (d_cg * e2).sum(-1)], dim = -1) / 64.
    loc_a = torch.stack([c * (pa - pb)[..., 0] + s * (pa - pb)[..., 1], -s * (pa - pb)[..., 0] + c * (pa - pb)[..., 1]], dim = -1) / 128.
    loc_g = torch.stack([c * d_cg[..., 0] + s * d_cg[..., 1], -s * d_cg[..., 0] + c * d_cg[..., 1]], dim = -1) / 128.
    dth = th - GOAL_TH
    kp_err = (get_kp(pb, th) - KP_GOAL) / 128.
    return torch.cat([cg_ab, d_ab / 128., d_cg / 128., loc_a, loc_g, torch.stack([s, c, torch.sin(dth), torch.cos(dth)], dim = -1), kp_err], dim = -1)

def to_env_action(raw: np.ndarray, act: np.ndarray, scale: float = 55.) -> np.ndarray:
    pa, pb, th = raw[..., 0:2], raw[..., 2:4], raw[..., 4]
    cb = np.stack([pb[..., 0] - 45. * np.sin(th), pb[..., 1] + 45. * np.cos(th)], axis = -1)
    d_ab = cb - pa
    e1 = d_ab / (np.linalg.norm(d_ab, axis = -1, keepdims = True) + 1e-6)
    e2 = np.stack([-e1[..., 1], e1[..., 0]], axis = -1)
    u = (act * 2. - 1.) * scale
    return np.clip(pa + u[..., 0:1] * e1 + u[..., 1:2] * e2, 15., 497.).astype(np.float32)

def make_sampler(method = 'sgs', seed = 0, progression_path = None, target = 0.15, kappa = 3.0, eps = 0.06, window = 10, unvisited_weight = 0.4, uniform_prob = None):
    if uniform_prob is None:
        uniform_prob = 0.10 if method == 'sgs' else 0.0

    return SGSSampler(
        configs = build_configs(),
        target = target,
        kappa = kappa,
        eps = eps,
        window = window,
        unvisited_weight = unvisited_weight,
        uniform_prob = uniform_prob,
        score_fn = uniform_score_fn if method == 'uniform' else None,
        log_every = 1,
        log_path = progression_path,
        seed = seed,
    )

# ppo training with SuccessGuidedSamplingWrapper

def train(
    method = 'sgs',
    steps = 1_000_000,
    num_envs = 16,
    rollout = 128,
    seed = 0,
    max_timesteps = 100,
    eval_every = 8,
    eval_episodes = 6,
    lr = 3e-4,
    gamma = 0.98,
    lam = 0.95,
    epochs = 6,
    minibatch = 256,
    target = 0.15,
    kappa = 3.0,
    uniform_prob = None,
    # reward = 0.1 * coverage_reward + success_bonus * success, value head unbounded.
    # NOTE: with small bonuses (and a saturating value head) a hovering episode
    # (block parked just under the coverage threshold for the whole horizon) returns
    # more than a solve, and the learner converges to hovering instead of the task
    success_bonus = 10.0,
    critic_sigmoid = False,
    value_clip = 0.0,
    entropy_coef = 5e-4,
    run_dir = None,
    progression_path = None,
    checkpoint_path = None,
    verbose = True,
):
    assert method in ('uniform', 'sgs', 'native'), f'unknown method {method}'

    torch.manual_seed(seed)
    np.random.seed(seed)

    configs = build_configs()
    bank_envs = [make_pusht(max_timesteps) for _ in configs]
    sampler = None if method == 'native' else make_sampler(method, seed = seed, progression_path = progression_path, target = target, kappa = kappa, uniform_prob = uniform_prob)

    if method == 'native':
        envs = [make_pusht(max_timesteps) for _ in range(num_envs)]
    else:
        envs = [SuccessGuidedSamplingWrapper(make_pusht(max_timesteps), reset_configs = sampler) for _ in range(num_envs)]

    actor = MLP(22, 128, 128, 4, activation = nn.SiLU())
    critic = MLP(22, 128, 128, 1, activation = nn.SiLU(), squeeze_out = True)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr = lr)
    succ_buf: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen = 20)

    def get_dist(obs_t):
        out = actor(featurize(obs_t))
        m, c = out[..., :2].sigmoid() * 0.90 + 0.05, 4. + (out[..., 2:] - 1.).sigmoid() * 60.
        return Beta(1. + m * c, 1. + (1. - m) * c), m

    def value_of(obs_t):
        v = critic(featurize(obs_t))
        return v.sigmoid() if critic_sigmoid else v

    @torch.no_grad()
    def evaluate(eval_episodes = 3, use_mean = False, seed_base = 0):
        # per-config success rates on the training bank, from the same fixed start states
        # for every arm and seed. use_mean evaluates the beta mean (greedy); otherwise
        # actions are sampled, matching the stochastic policy the learner optimizes

        succ = np.zeros((eval_episodes, len(configs)), dtype = np.float32)

        for ep in range(eval_episodes):
            obs = np.stack([e.reset(seed = seed_base + 1000 * ep + i, options = dict(reset_to_state = c))[0] for i, (e, c) in enumerate(zip(bank_envs, configs))]).astype(np.float32)
            done = np.zeros(len(configs), dtype = bool)

            for _ in range(max_timesteps):
                if done.all():
                    break

                dist, m = get_dist(torch.from_numpy(obs))

                with torch.random.fork_rng():
                    torch.manual_seed(seed_base + 7 * ep + 1)
                    act_np = (m if use_mean else dist.sample().clamp(0.01, 0.99)).numpy()

                for i in np.where(~done)[0]:
                    obs[i], _, term, trunc, _ = bank_envs[i].step(to_env_action(obs[i], act_np[i]))
                    if term or trunc:
                        succ[ep, i], done[i] = float(term), True

        return succ.mean(0)

    obs = np.stack([e.reset()[0] for e in envs]).astype(np.float32)
    ep_s, ep_a = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]
    recent_succ = deque(maxlen = 100)
    num_episodes = 0
    t0 = time.time()

    history = dict(steps = [], time = [], eval_all = [], eval_frontier = [], eval_all_sampled = [], eval_frontier_sampled = [], frontier_mass = [], episodes = [], train_succ = [], per_config = [])
    num_updates = steps // (num_envs * rollout)
    best_frontier = -1.0

    for upd in range(1, num_updates + 1):
        S, A, LP, R, V, M = [], [], [], [], [], []

        for _ in range(rollout):
            obs_t = torch.from_numpy(obs)
            with torch.no_grad():
                dist, _ = get_dist(obs_t)
                act = dist.sample().clamp(0.01, 0.99)
                lp, val = dist.log_prob(act).sum(-1), value_of(obs_t)
            env_act = to_env_action(obs, act.numpy())
            next_obs, rew, mask = np.empty_like(obs), np.zeros(num_envs, dtype = np.float32), np.ones(num_envs, dtype = np.float32)
            for i, e in enumerate(envs):
                no, cov_r, term, trunc, _ = e.step(env_act[i])
                ep_s[i].append(obs[i]); ep_a[i].append(act[i].numpy())
                rew[i] = 0.1 * float(cov_r) + (success_bonus if term else 0.)
                if trunc and not term:
                    with torch.no_grad():
                        rew[i] += gamma * float(value_of(torch.from_numpy(no.astype(np.float32))))
                if term or trunc:
                    num_episodes += 1
                    recent_succ.append(float(term))
                    if term:
                        sa, aa = np.stack(ep_s[i]), np.stack(ep_a[i])
                        succ_buf.append((sa, aa, gamma ** np.arange(len(sa) - 1, -1, -1, dtype = np.float32)))
                    ep_s[i].clear(); ep_a[i].clear(); mask[i] = 0.
                    no, _ = e.reset()
                next_obs[i] = no
            S.append(obs_t); A.append(act); LP.append(lp); R.append(torch.from_numpy(rew)); V.append(val); M.append(torch.from_numpy(mask))
            obs = next_obs

        with torch.no_grad():
            last_v = value_of(torch.from_numpy(obs))
            rets, advs = calc_gae(torch.stack(R).T, torch.stack(V).T, torch.stack(M).T, next_value = last_v, gamma = gamma, lam = lam, use_accelerated = False, return_advantages = True)
            rets, advs = rets.T.reshape(-1), advs.T.reshape(-1)
            if value_clip > 0.:
                rets = rets.clamp(0., value_clip)
            advs = (advs - advs.mean()) / advs.std().clamp(min = 0.08)

        Sf, Af, LPf = torch.cat(S), torch.cat(A), torch.cat(LP)
        for _ in range(epochs):
            for idx in torch.randperm(len(Sf)).split(minibatch):
                dist, _ = get_dist(Sf[idx])
                pg = ppo_actor_loss(dist.log_prob(Af[idx]).sum(-1), LPf[idx], advs[idx], 0.2, normalize_advantages = False).mean()
                vf_target = rets[idx].clamp(0., 1.) if critic_sigmoid else rets[idx]
                vf = F.mse_loss(value_of(Sf[idx]), vf_target)
                loss = pg + 0.5 * vf - entropy_coef * dist.entropy().sum(-1).mean()
                if succ_buf:
                    bs = torch.from_numpy(np.concatenate([v[0] for v in succ_buf]))
                    ba = torch.from_numpy(np.concatenate([v[1] for v in succ_buf])).clamp(0.01, 0.99)
                    br = torch.from_numpy(np.concatenate([v[2] for v in succ_buf]))
                    bdist, bmean = get_dist(bs)
                    loss = loss + 0.25 * ((-bdist.log_prob(ba).sum(-1) + 6. * (bmean - ba).pow(2).sum(-1)) * br).mean() + 0.15 * F.mse_loss(value_of(bs), success_bonus * br)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5); opt.step()

        if upd % eval_every == 0 or upd == num_updates:
            per_config = evaluate(eval_episodes = eval_episodes, use_mean = True)
            per_config_sampled = evaluate(eval_episodes = eval_episodes, use_mean = False)

            f_mass = float(sampler.probs[NUM_EASY:].sum()) if method == 'sgs' else float('nan')

            history['steps'].append(upd * num_envs * rollout)
            history['time'].append(time.time() - t0)
            history['eval_all'].append(float(per_config.mean()))
            history['eval_frontier'].append(float(per_config[NUM_EASY:].mean()))
            history['eval_all_sampled'].append(float(per_config_sampled.mean()))
            history['eval_frontier_sampled'].append(float(per_config_sampled[NUM_EASY:].mean()))
            history['frontier_mass'].append(f_mass)
            history['episodes'].append(num_episodes)
            history['train_succ'].append(float(np.mean(recent_succ)) if recent_succ else float('nan'))
            history['per_config'].append([float(x) for x in per_config])

            if per_config[NUM_EASY:].mean() > best_frontier and checkpoint_path:
                best_frontier = float(per_config[NUM_EASY:].mean())
                torch.save(dict(actor = actor.state_dict(), method = method, seed = seed, steps = upd * num_envs * rollout, eval_frontier = best_frontier), checkpoint_path)

            if verbose:
                print(f'[{method:>7}] step {upd * num_envs * rollout:8d} | eval_all: {per_config.mean():5.1%} | eval_frontier: {per_config[NUM_EASY:].mean():5.1%} | '
                      f'frontier_prob: {f_mass:5.1%} | train_succ: {history["train_succ"][-1]:5.1%} | eps: {num_episodes:5d} | time: {time.time() - t0:5.1f}s', flush = True)

    for e in [*envs, *bank_envs]:
        close = getattr(e, 'close', None)
        if callable(close):
            close()

    final_ckpt = None
    if checkpoint_path:
        final_ckpt = str(checkpoint_path).replace('.pt', '_final.pt')
        torch.save(dict(actor = actor.state_dict(), method = method, seed = seed, steps = steps), final_ckpt)

    return dict(method = method, seed = seed, steps = steps, max_timesteps = max_timesteps, history = history, time = time.time() - t0, checkpoint = final_ckpt)

# report

STYLE = {
    'native':  dict(color = '#7f7f7f', label = 'native spawns (no curriculum)'),
    'uniform': dict(color = '#3b7dd8', label = 'uniform bank resets'),
    'sgs':     dict(color = '#d1495b', label = 'SGS (success-guided resets)'),
}

PANELS = [
    ('eval_all',               'eval success on the whole bank (greedy)'),
    ('eval_frontier',          'eval success on the 6 frontier states (greedy)'),
    ('eval_frontier_sampled',  'eval success on the 6 frontier states (sampled)'),
    ('frontier_mass',          'SGS sampling mass on the frontier configs'),
    ('episodes',               'training episodes collected (what the step budget buys)'),
    ('train_succ',             'rolling training success (last 100 episodes)'),
]

def runs_of(runs, method, steps):
    return [r for r in runs if r['method'] == method and r['steps'] == steps]

def plot(runs, steps, path = 'pusht_sgs.png'):
    methods = [m for m in ('native', 'uniform', 'sgs') if runs_of(runs, m, steps)]

    fig, axes = plt.subplots(2, 3, figsize = (18.5, 9.2), dpi = 130)

    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for method in methods:
            method_runs = runs_of(runs, method, steps)
            x = np.asarray(method_runs[0]['history']['steps'], dtype = float)
            y = np.stack([r['history'][key] for r in method_runs])

            if key == 'frontier_mass' and not np.isfinite(y).any():
                continue

            y = y * 100.
            mean, std = y.mean(0), y.std(0)
            ax.plot(x / 1000., mean, color = STYLE[method]['color'], linewidth = 2., label = STYLE[method]['label'])

            if len(method_runs) > 1:
                ax.fill_between(x / 1000., mean - std, mean + std, color = STYLE[method]['color'], alpha = 0.18, linewidth = 0)

        ax.set_xlabel('environment steps (thousands)')
        ax.set_ylabel('%')
        ax.set_title(title, fontsize = 10.5)
        ax.grid(alpha = 0.25)

        if key != 'episodes':
            ax.set_ylim(-3., 103.)

    axes[0, 0].legend(loc = 'lower right', frameon = True, fontsize = 9)

    lines = []
    for key, label in (('eval_all', 'bank'), ('eval_frontier', 'frontier'), ('eval_frontier_sampled', 'frontier (sampled)')):
        row = []
        for method in methods:
            final = np.stack([r['history'][key][-1] for r in runs_of(runs, method, steps)])
            row.append(f'{method} {final.mean():5.1%} +/- {final.std():4.1%}')
        lines.append(f'final {label:<18} ' + '  |  '.join(row))

    num_seeds = len(runs_of(runs, methods[0], steps))
    fig.suptitle(
        f'Success-Guided Sampling on Push-T - PPO, same learner, {len(methods)} arms x {num_seeds} seeds, {steps // 1000}k env steps each\n' + '\n'.join(lines),
        fontsize = 11.5,
    )
    fig.tight_layout(rect = (0., 0., 1., 0.92))
    fig.savefig(path)
    print(f'saved chart to {path}')

    return path

# rendering

def load_actor(path):
    ckpt = torch.load(path, map_location = 'cpu', weights_only = False)
    actor = MLP(22, 128, 128, 4, activation = nn.SiLU())
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    return actor, ckpt

@torch.no_grad()
def record_solves(actor, run_dir, max_timesteps = 50, seed = 0, max_videos = 6):
    configs = build_configs()
    results = []

    for idx in range(NUM_EASY, len(configs)):
        record = idx < NUM_EASY + max_videos
        env = gym.make('gym_pusht/PushT-v0', render_mode = 'rgb_array')
        env.unwrapped.step = fast_step.__get__(env.unwrapped, PushTEnv)

        obs, _ = env.reset(seed = seed + idx, options = dict(reset_to_state = configs[idx]))
        raw = np.asarray(obs, dtype = np.float32)
        frames, success, best = [], False, 0.0

        for _ in range(max_timesteps):
            out = actor(featurize(torch.from_numpy(raw[None])))
            act = (out[..., :2].sigmoid() * 0.90 + 0.05).numpy()[0]
            action = to_env_action(raw, act)
            obs, reward, term, trunc, info = env.step(action)
            raw = np.asarray(obs, dtype = np.float32)
            best = max(best, float(info.get('coverage', 0.)))
            if record:
                frames.append(env.render())
            if term or trunc:
                success = bool(term)
                break

        env.close()
        results.append(dict(config = idx, success = success, best_coverage = best))

        if record and frames:
            path = run_dir / f'solve_cfg_{idx:02d}_{"ok" if success else "fail"}.gif'
            imageio.mimsave(path, frames, fps = 12, loop = 0)

    return results

# matrix runner

def as_list(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(',') if v.strip()]

def main(
    steps: int = 1_000_000,
    seeds: str = '0,1,2',
    methods: str = 'uniform,sgs',
    num_envs: int = 16,
    rollout: int = 128,
    max_timesteps: int = 100,
    eval_every: int = 8,
    eval_episodes: int = 6,
    lr: float = 3e-4,
    target: float = 0.15,
    kappa: float = 3.0,
    success_bonus: float = 10.0,
    critic_sigmoid: bool = False,
    value_clip: float = 0.0,
    entropy_coef: float = 5e-4,
    parallel: int = 1,
    single: bool = False,
    run: str = 'pusht_sgs',
    render: bool = True,
    rerun: bool = False,
):
    run_dir = Path('experiments') / str(run)
    run_dir.mkdir(parents = True, exist_ok = True)

    method_list, seed_list = [str(m) for m in as_list(methods)], [int(s) for s in as_list(seeds)]

    def combo_kwargs(method, seed):
        return dict(
            method = method, steps = steps, num_envs = num_envs, rollout = rollout, seed = seed,
            max_timesteps = max_timesteps, eval_every = eval_every, eval_episodes = eval_episodes,
            lr = lr, target = target, kappa = kappa,
            success_bonus = success_bonus, critic_sigmoid = critic_sigmoid, value_clip = value_clip, entropy_coef = entropy_coef,
            run_dir = run_dir,
            progression_path = run_dir / f'sgs_progression_{method}_{seed}.csv',
            checkpoint_path = run_dir / f'actor_{method}_{seed}.pt',
        )

    def result_path(method, seed):
        return run_dir / f'result_{method}_{seed}.json'

    if single:
        method, seed = method_list[0], seed_list[0]
        result = train(**combo_kwargs(method, seed))
        result_path(method, seed).write_text(json.dumps(result))
        return result

    combos = [(m, s) for m in method_list for s in seed_list]

    if parallel > 1:
        pending = [(m, s) for m, s in combos if rerun or not result_path(m, s).exists()]
        procs = {}

        while pending or procs:
            while pending and len(procs) < parallel:
                method, seed = pending.pop(0)
                log = open(run_dir / f'worker_{method}_{seed}.log', 'w')
                cmd = [sys.executable, str(Path(__file__).resolve()), '--single', 'True', '--methods', method, '--seeds', str(seed),
                       '--steps', str(steps), '--num_envs', str(num_envs), '--rollout', str(rollout), '--max_timesteps', str(max_timesteps),
                       '--eval_every', str(eval_every), '--eval_episodes', str(eval_episodes), '--lr', str(lr), '--target', str(target),
                       '--kappa', str(kappa), '--success_bonus', str(success_bonus), '--critic_sigmoid', str(critic_sigmoid),
                       '--value_clip', str(value_clip), '--entropy_coef', str(entropy_coef),
                       '--run', str(run), '--render', 'False']
                env_vars = dict(os.environ, OMP_NUM_THREADS = '1', MKL_NUM_THREADS = '1', PUSHT_WORKER = '1')
                procs[(method, seed)] = (subprocess.Popen(cmd, stdout = log, stderr = subprocess.STDOUT, env = env_vars), log)
                print(f'=== launched {method} seed {seed} (pid {procs[(method, seed)][0].pid}) ===', flush = True)

            done = [k for k, (p, _) in procs.items() if p.poll() is not None]
            for k in done:
                p, log = procs.pop(k)
                log.close()
                print(f'=== {k[0]} seed {k[1]} finished (exit {p.returncode}) ===', flush = True)
            if not done:
                time.sleep(2)
    else:
        for method, seed in combos:
            if not rerun and result_path(method, seed).exists():
                print(f'skipping {method} seed {seed} (cached)')
                continue
            print(f'=== {method} | seed {seed} ===', flush = True)
            result = train(**combo_kwargs(method, seed))
            result_path(method, seed).write_text(json.dumps(result))
            print(f'=== {method} | seed {seed} done in {result["time"]:.1f}s ===\n', flush = True)

    runs = [json.loads(p.read_text()) for p in sorted(run_dir.glob('result_*.json')) if json.loads(p.read_text())['steps'] == steps]
    (run_dir / 'results.json').write_text(json.dumps(runs))

    if len(runs):
        plot(runs, steps = steps, path = run_dir / 'pusht_sgs.png')

        sgs_runs = runs_of(runs, 'sgs', steps)

        if render and sgs_runs:
            best = max(sgs_runs, key = lambda r: r['history']['eval_frontier'][-1])
            ckpt = run_dir / f'actor_sgs_{best["seed"]}.pt'
            if ckpt.exists():
                actor, meta = load_actor(ckpt)
                print(f'=== rendering best SGS policy (seed {best["seed"]}, frontier {meta.get("eval_frontier", float("nan")):.1%}) ===', flush = True)
                results = record_solves(actor, run_dir, max_timesteps = max_timesteps)
                print('frontier solves:', results, flush = True)

    print(f'\nresults in {run_dir}/')
    return runs

if __name__ == '__main__':
    fire.Fire(main)
