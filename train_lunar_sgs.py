# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium[box2d]",
#     "imageio",
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "torch",
#     "torch-einops-utils",
#     "x-mlps-pytorch",
#     "x-ppo>=0.0.7",
# ]
# ///
#
# Success-Guided Sampling on LunarLander - an adaptive curriculum over reset states
#
#   uv run train_lunar_sgs.py                            # 3 arms x 5 seeds x 200k steps
#   uv run train_lunar_sgs.py --steps=20000 --seeds=0    # quick look
#
# three arms share the learner, the step budget and the evaluation protocol - only the
# reset distribution differs
#
#   native  - the environment's own random spawn (regular RL)
#   uniform - uniform sampling over a bank of reset states
#   sgs     - success-guided sampling over that same bank, chasing the frontier
#
# everything lands in experiments/lunar_sgs/: results.json, the progression of the bank
# per run (sgs_progression_*.csv), actor checkpoints, the speedup chart and the landings
# of the best SGS policy

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import fire
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw
from torch.distributions import Categorical
from torch_einops_utils import cast_tensor, safe_stack, z_score
from x_mlps_pytorch import MLP
from x_ppo import calc_gae, ppo_actor_loss

from gymnasium.envs.box2d.lunar_lander import (
    FPS, LEG_AWAY, LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W, LunarLander,
)

from env_ssl_wrapper import (
    SGSSampler,
    StandardizeEnvWrapper,
    SuccessGuidedSamplingWrapper,
    uniform_score_fn,
)
from env_ssl_wrapper.helpers import divisible_by

# lunar lander with reset-to-state
#
# LunarLander-v3 regenerates its world on reset and only ever spawns at the top of the
# screen. the lander body, both legs and the contact flags are reachable through Box2D
# though, so any point of the 8d state space can be spawned into a physically identical
# environment - same dynamics, same reward, same observations

HALF_W = VIEWPORT_W / SCALE / 2
HALF_H = VIEWPORT_H / SCALE / 2
FOOT = LEG_DOWN / SCALE

class ResetStateLunarLander(LunarLander):
    """ identical physics to LunarLander-v3, plus reset(options = dict(reset_to_state = s)) """

    def reset(self, *, seed = None, options = None):
        state = (options or {}).get('reset_to_state')
        obs, info = super().reset(seed = seed, options = options)

        if state is None:
            return obs, info

        self.set_state(np.asarray(state, dtype = np.float64))
        return self.get_state(), info

    def get_state(self) -> np.ndarray:
        # same observation mapping as LunarLander.step
        pos, vel = self.lander.position, self.lander.linearVelocity

        return np.array([
            (pos.x - HALF_W) / HALF_W,
            (pos.y - (self.helipad_y + FOOT)) / HALF_H,
            vel.x * HALF_W / FPS,
            vel.y * HALF_H / FPS,
            self.lander.angle,
            20. * self.lander.angularVelocity / FPS,
            1. if self.legs[0].ground_contact else 0.,
            1. if self.legs[1].ground_contact else 0.,
        ], dtype = np.float32)

    def set_state(self, s):
        x = float(s[0] * HALF_W + HALF_W)
        y = float(s[1] * HALF_H + self.helipad_y + FOOT)
        vx, vy = float(s[2] * FPS / HALF_W), float(s[3] * FPS / HALF_H)
        th, w = float(s[4]), float(s[5] * FPS / 20.)

        self.world.ClearForces()  # drop the random spawn impulse
        self.game_over, self.prev_shaping = False, None

        self.lander.position = (x, y)
        self.lander.angle = th
        self.lander.linearVelocity = (vx, vy)
        self.lander.angularVelocity = w

        for leg, i in zip(self.legs, (-1, +1)):  # snap the legs onto the new pose
            lth = th + i * 0.05
            lx = x - (np.cos(lth) * i * LEG_AWAY / SCALE - np.sin(lth) * FOOT)
            ly = y - (np.sin(lth) * i * LEG_AWAY / SCALE + np.cos(lth) * FOOT)
            rx, ry = lx - x, ly - y
            leg.position = (lx, ly)
            leg.angle = lth
            leg.linearVelocity = (vx - w * ry, vy + w * rx)
            leg.angularVelocity = w
            leg.ground_contact = bool(s[6 + (i > 0)])

def make_env(max_timesteps = 1000):
    return StandardizeEnvWrapper(
        ResetStateLunarLander(),
        auto_batch = False, device = None, done_tracker = False,
        pad_episodes = False, max_timesteps = max_timesteps,
    )

def landed(obs, reward, terminated, truncated, info):
    return terminated * (reward >= 50.)  # +100 terminal on a soft landing, -100 on a crash

# reset state bank
#
# a difficulty ladder with a foothold - it starts at near-resting states that even a
# do-nothing policy solves, then ramps up fall speed, offset, tilt and spin, so there is
# always a rung just above what the policy can do. the held out bank is never sampled by
# any arm, and tests whether competence transfers

def build_bank(k = 20):
    bank = []

    for i in range(k):
        u = i / (k - 1)
        s = np.zeros(8, dtype = np.float32)
        sgn_x = 1. if i % 2 == 0 else -1.
        sgn_v = 1. if (i // 2) % 2 == 0 else -1.
        sgn_a = 1. if i % 3 == 0 else (-1. if i % 3 == 1 else 1.)

        s[0] = 0.35 * max(u - 0.15, 0.) / 0.85 * sgn_x             # x offset
        s[1] = 0.01 + 0.70 * u ** 1.4                               # altitude above the pad
        s[2] = 0.35 * max(u - 0.15, 0.) / 0.85 * sgn_v             # vx
        s[3] = -0.45 * max(u - 0.10, 0.) / 0.90                    # vy, descending
        s[4] = 0.35 * max(u - 0.25, 0.) / 0.75 * sgn_a             # tilt
        s[5] = 0.30 * max(u - 0.40, 0.) / 0.60 * (-sgn_x)          # angular velocity
        bank.append(s)

    return bank

def build_heldout(k = 8):
    bank = []

    for j in range(k):
        u = 0.25 + 0.75 * j / (k - 1)
        s = np.zeros(8, dtype = np.float32)
        sgn_x = -1. if j % 2 == 0 else 1.
        sgn_v = -1. if (j // 2) % 2 == 0 else 1.

        s[0] = 0.32 * u * sgn_x
        s[1] = 0.05 + 0.65 * u ** 1.3
        s[2] = 0.30 * u * sgn_v
        s[3] = -0.40 * max(u - 0.05, 0.) / 0.95
        s[4] = 0.30 * u * sgn_x
        s[5] = 0.25 * u * sgn_v
        bank.append(s)

    return bank

# sgs curriculum config
#
# target           competence level the frontier is pulled toward
# kappa            sharpness of the score around the target
# eps              smoothing floor on the score weights
# window           episodes of memory per reset state
# unvisited_weight pull toward states that were never tried
# uniform_prob     exploration floor under the curriculum distribution

SGS_CONFIG = dict(
    target = 0.4,
    kappa = 3.,
    eps = 0.06,
    window = 10,
    unvisited_weight = 0.4,
    uniform_prob = 0.1,
)

def make_sampler(arm = 'sgs', seed = 0, progression_path = None):
    if arm == 'native':
        return None

    kwargs = dict(SGS_CONFIG)

    if arm == 'uniform':  # same bank and success tracking, flat score and no exploration floor
        kwargs.update(score_fn = uniform_score_fn, uniform_prob = 0.)

    # log_every = 1 keeps one snapshot per episode in memory, and appends to the csv

    return SGSSampler(
        configs = build_bank(),
        log_every = 1,
        log_path = progression_path,
        seed = seed,
        **kwargs,
    )

# ppo

def train(
    arm = 'sgs',
    steps = 200_000,
    num_envs = 16,
    rollout = 128,
    seed = 0,
    max_timesteps = 1000,
    eval_every = 4,
    num_native_eval = 12,
    num_eval_repeats = 3,
    lr = 1e-3,
    gamma = 0.999,
    lam = 0.95,
    epochs = 10,
    minibatch = 256,
    entropy_coef = 0.01,
    progression_path = None,
    checkpoint_path = None,
    verbose = True,
):
    assert arm in ('native', 'uniform', 'sgs'), f'unknown arm {arm}'
    assert steps >= num_envs * rollout, f'steps must cover at least one update of {num_envs} x {rollout}'

    torch.manual_seed(seed)
    np.random.seed(seed)

    bank, heldout = build_bank(), build_heldout()
    num_updates = steps // (num_envs * rollout)  # the budget is floored to whole updates
    sampler = make_sampler(arm, seed = seed, progression_path = progression_path)

    def make_train_env():
        env = make_env(max_timesteps)
        return env if arm == 'native' else SuccessGuidedSamplingWrapper(env, reset_configs = sampler, success_fn = landed)

    envs = [make_train_env() for _ in range(num_envs)]
    bank_envs = [make_env(max_timesteps) for _ in bank]
    held_envs = [make_env(max_timesteps) for _ in heldout]
    native_envs = [make_env(max_timesteps) for _ in range(num_native_eval)]

    actor = MLP(8, 128, 128, 4, activation = nn.SiLU())
    critic = MLP(8, 128, 128, 1, activation = nn.SiLU(), squeeze_out = True)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr = lr)

    def get_dist(obs_t):
        return Categorical(logits = actor(obs_t))

    @torch.no_grad()
    def greedy_rollouts(env_list, starts, seed_base, round_id = 0, repeats = 1):
        # scored with the same states and the same env randomness for every arm and seed,
        # re-seeded per round so the comparison stays paired - hence the fixed seed bases

        n = len(env_list)
        succ, rets = np.zeros(n), np.zeros(n)

        for rep in range(repeats):
            obs = torch.zeros(n, 8)
            done = np.zeros(n, dtype = bool)

            for i, e in enumerate(env_list):
                reset_out = e.reset(seed = seed_base + 1000 * round_id + 100 * rep + i, options = starts[i] if starts else None)
                obs[i] = cast_tensor(reset_out[0])

            for _ in range(max_timesteps):
                if done.all():
                    break

                acts = actor(obs).argmax(-1).numpy()

                for i in range(n):
                    if done[i]:
                        continue

                    o, r, term, trunc, _ = env_list[i].step(int(acts[i]))
                    rets[i] += float(r) / repeats

                    if term or trunc:
                        succ[i] += float(term and r >= 50.) / repeats
                        done[i] = True
                    else:
                        obs[i] = cast_tensor(o)

        return succ, rets

    bank_starts = [dict(reset_to_state = c) for c in bank]
    held_starts = [dict(reset_to_state = c) for c in heldout]
    history = dict(steps = [], time = [], bank = [], heldout = [], native_succ = [], native_ret = [], frontier_mass = [], episodes = [], train_succ = [])

    obs = np.zeros((num_envs, 8), dtype = np.float32)
    for i, e in enumerate(envs):
        obs[i] = e.reset(seed = 1234 * (seed + 1) + i)[0]

    recent_succ = deque(maxlen = 100)  # rolling training success, logs only
    num_episodes = 0
    t0 = time.time()

    for upd in range(1, num_updates + 1):
        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []

        for _ in range(rollout):
            obs_t = cast_tensor(obs)

            with torch.no_grad():
                dist = get_dist(obs_t)
                act = dist.sample()
                lp, val = dist.log_prob(act), critic(obs_t)

            next_obs, rew = np.empty_like(obs), np.zeros(num_envs, dtype = np.float32)
            mask = np.ones(num_envs, dtype = np.float32)

            for i, e in enumerate(envs):
                no, r, term, trunc, _ = e.step(int(act[i]))
                rew[i] = float(r)

                if term or trunc:
                    num_episodes += 1
                    recent_succ.append(float(term and r >= 50.))
                    mask[i] = 0. if term else 1.  # truncated episodes still bootstrap
                    no, _ = e.reset()

                next_obs[i] = no

            states.append(obs_t); actions.append(act); log_probs.append(lp)
            rewards.append(cast_tensor(rew)); values.append(val); masks.append(cast_tensor(mask))
            obs = next_obs

        # generalized advantage estimation - (n b) rollouts -> (b n), then flattened

        with torch.no_grad():
            last_v = critic(cast_tensor(obs))
            rets, advs = calc_gae(
                rearrange(safe_stack(rewards), 'n b -> b n'),
                rearrange(safe_stack(values), 'n b -> b n'),
                rearrange(safe_stack(masks), 'n b -> b n'),
                next_value = last_v, gamma = gamma, lam = lam,
                use_accelerated = False, return_advantages = True,
            )
            rets, advs = (rearrange(x, 'b n -> (n b)') for x in (rets, advs))  # time major, matching torch.cat over rollout steps
            advs = z_score(advs)

        all_states, all_actions, all_log_probs = torch.cat(states), torch.cat(actions), torch.cat(log_probs)

        for _ in range(epochs):
            for idx in torch.randperm(all_states.shape[0]).split(minibatch):
                dist = get_dist(all_states[idx])
                pg = ppo_actor_loss(dist.log_prob(all_actions[idx]), all_log_probs[idx], advs[idx], 0.2, normalize_advantages = False).mean()
                vf = F.mse_loss(critic(all_states[idx]), rets[idx])
                loss = pg + 0.5 * vf - entropy_coef * dist.entropy().mean()

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5)
                opt.step()

        if upd % eval_every == 0 or upd == num_updates:
            round_id = upd // eval_every
            bank_succ, _ = greedy_rollouts(bank_envs, bank_starts, 977_000, round_id, repeats = num_eval_repeats)
            held_succ, _ = greedy_rollouts(held_envs, held_starts, 733_000, round_id, repeats = num_eval_repeats)
            native_succ, native_ret = greedy_rollouts(native_envs, None, 511_000, round_id)

            frontier_mass = float('nan')
            if sampler is not None:
                p, w = sampler.success_rates, sampler.probs
                frontier_mass = float(w[torch.isfinite(p) & (p > 0.1) & (p < 0.9)].sum())

            history['steps'].append(upd * num_envs * rollout)
            history['time'].append(time.time() - t0)
            history['bank'].append(float(bank_succ.mean()))
            history['heldout'].append(float(held_succ.mean()))
            history['native_succ'].append(float(native_succ.mean()))
            history['native_ret'].append(float(native_ret.mean()))
            history['frontier_mass'].append(frontier_mass)
            history['episodes'].append(num_episodes)
            history['train_succ'].append(float(np.mean(recent_succ)) if recent_succ else float('nan'))

            if verbose:
                train_sr = history['train_succ'][-1]
                print(
                    f'[{arm:>7}] step {upd * num_envs * rollout:7d} | '
                    f'bank: {bank_succ.mean():5.1%} | holdout: {held_succ.mean():5.1%} | '
                    f'native: {native_succ.mean():5.1%} | train: {train_sr:5.1%} | '
                    f'frontier: {frontier_mass:5.1%} | time: {time.time() - t0:5.1f}s',
                    flush = True,
                )

    for e in [*envs, *bank_envs, *held_envs, *native_envs]:
        close = getattr(e, 'close', None)
        if callable(close):
            close()

    if checkpoint_path:
        torch.save(dict(actor = actor.state_dict(), arm = arm, seed = seed, steps = steps, final_bank = history['bank'][-1]), checkpoint_path)

    return dict(arm = arm, seed = seed, steps = steps, history = history, time = time.time() - t0)

# report

STYLE = {
    'native':  dict(color = '#7f7f7f', label = 'regular RL (native random resets)'),
    'uniform': dict(color = '#3b7dd8', label = 'uniform resets over the same bank'),
    'sgs':     dict(color = '#d1495b', label = 'SGS (success-guided resets)'),
}

# key, scale tag, panel title - scale picks the y label and the factor applied to the metric

PANELS = [
    ('bank',          'pct',    'landing success on the training bank (20 reset states)'),
    ('heldout',       'pct',    'landing success on held-out reset states (8)'),
    ('native_succ',   'pct',    'landing success on native random spawns'),
    ('native_ret',    'return', 'mean episode return on native spawns'),
    ('episodes',      'count',  'training episodes collected (what the step budget buys)'),
    ('frontier_mass', 'pct',    'sampling mass on the frontier (0.1 < success rate < 0.9)'),
]

YLABELS = {'pct': 'success rate (%)', 'return': 'return', 'count': 'episodes'}
FACTORS = {'pct': 100., 'return': 1., 'count': 1.}

def steps_to(curve, target, x):
    # first step at which the curve crosses target, linearly interpolated

    if len(curve) and curve[0] >= target:
        return x[0]

    for i in range(1, len(curve)):
        if np.isnan(curve[i - 1]):
            continue
        if curve[i - 1] < target <= curve[i]:
            frac = (target - curve[i - 1]) / max(curve[i] - curve[i - 1], 1e-12)
            return x[i - 1] + frac * (x[i] - x[i - 1])

    return None

def runs_of(runs, arm, steps):
    return [r for r in runs if r['arm'] == arm and r['steps'] == steps]

def plot(runs, steps = 200_000, path = 'lunar_sgs.png', thresholds = (0.5, 0.8)):
    arms = [a for a in ('native', 'uniform', 'sgs') if runs_of(runs, a, steps)]
    trained = float(np.asarray(runs_of(runs, arms[0], steps)[0]['history']['steps'])[-1])  # the budget floors to whole updates

    fig, axes = plt.subplots(2, 3, figsize = (18.5, 9.2), dpi = 130)
    crossings = {}

    for ax, (key, scale, title) in zip(axes.ravel(), PANELS):
        for arm in arms:
            arm_runs = runs_of(runs, arm, steps)
            x = np.asarray(arm_runs[0]['history']['steps'], dtype = float)
            y = np.stack([r['history'][key] for r in arm_runs])

            if key == 'native_ret':  # per episode return means are very noisy - smooth per seed
                pad = np.pad(np.nan_to_num(y, nan = 0.), ((0, 0), (1, 1)), mode = 'edge')
                y = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.

            if key == 'frontier_mass' and not np.isfinite(y).any():
                continue

            mean, std = FACTORS[scale] * y.mean(0), FACTORS[scale] * y.std(0)
            ax.plot(x / 1000., mean, color = STYLE[arm]['color'], linewidth = 2., label = STYLE[arm]['label'])

            if len(arm_runs) > 1:
                ax.fill_between(x / 1000., mean - std, mean + std, color = STYLE[arm]['color'], alpha = 0.18, linewidth = 0)

            if key == 'bank':
                crossings[arm] = steps_to(y.mean(0), thresholds[0], x)

        ax.set_xlabel('environment steps (thousands)')
        ax.set_ylabel(YLABELS[scale])
        ax.set_title(title, fontsize = 10.5)
        ax.set_xlim(0, trained / 1000.); ax.grid(alpha = 0.25)

        if scale == 'pct':
            ax.set_ylim(-3., 103.)
            for thr in thresholds:
                ax.axhline(100. * thr, color = 'black', linestyle = ':', linewidth = 0.7, alpha = 0.3)

        if key == 'native_ret':
            ax.set_ylim(-800., 400.)
            ax.axhline(200., color = 'black', linestyle = ':', linewidth = 1.)
            ax.text(steps / 1000. * 0.99, 205., 'solved (200)', ha = 'right', va = 'bottom', fontsize = 8, alpha = 0.7)

    # mark the steps at which each arm reaches the first target on the primary metric

    ax = axes[0, 0]
    notes = []

    for arm, step in crossings.items():
        if step is None:
            continue
        ax.axvline(step / 1000., color = STYLE[arm]['color'], linestyle = '--', linewidth = 1., alpha = 0.8)
        notes.append(f'{arm}: {step / 1000.:.0f}k')

    if notes:
        ax.text(0.02, 0.97, f'steps to {thresholds[0]:.0%} bank success\n' + '\n'.join(notes), transform = ax.transAxes, fontsize = 8.5,
                va = 'top', ha = 'left', bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.85, edgecolor = '0.7'))

    ax.legend(loc = 'lower right', frameon = True, fontsize = 9)

    # header table: steps to each target, and the mean success over the whole budget

    lines = []

    for thr in thresholds:
        row = []

        for arm in arms:
            arm_runs = runs_of(runs, arm, steps)
            x = np.asarray(arm_runs[0]['history']['steps'], dtype = float)
            curve = np.stack([r['history']['bank'] for r in arm_runs]).mean(0)
            step = steps_to(curve, thr, x)
            row.append(f'{arm} {step / 1000.:.0f}k' if step is not None else f'{arm} never')

        lines.append(f'steps to {thr:.0%} bank success   ' + '  |  '.join(row))

    row = []

    for arm in arms:
        bank = np.stack([r['history']['bank'] for r in runs_of(runs, arm, steps)])
        row.append(f'{arm} {bank.mean():5.1%} ({bank[:, -1].mean():5.1%} final)')

    lines.append('mean bank success over the budget (final)   ' + '  |  '.join(row))

    num_seeds = len(runs_of(runs, arms[0], steps))
    fig.suptitle(
        f'Success-Guided Sampling on LunarLander - PPO, same learner, {len(arms)} arms x {num_seeds} seeds, {trained / 1000:.0f}k env steps each\n' + '\n'.join(lines),
        fontsize = 11.5,
    )
    fig.tight_layout(rect = (0., 0., 1., 0.92))
    fig.savefig(path)
    print(f'saved chart to {path}')

    return path

# rendering

def load_actor(path):
    checkpoint = torch.load(path, map_location = 'cpu')
    actor = MLP(8, 128, 128, 4, activation = nn.SiLU())
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    return actor, checkpoint

@torch.no_grad()
def record_landing(actor, state, seed = 0, max_steps = 1000, render = True):
    # greedy rollout from one reset state - frames are only built when rendering

    env = ResetStateLunarLander(render_mode = 'rgb_array' if render else None)
    obs, _ = env.reset(seed = seed, options = dict(reset_to_state = np.asarray(state, dtype = np.float32)))
    frames, states, total, success = [env.render()] if render else [], [np.asarray(obs, dtype = float)], 0., False

    for _ in range(max_steps):
        act = int(actor(cast_tensor(obs)).argmax(-1))
        obs, r, term, trunc, _ = env.step(act)
        total += float(r)
        states.append(np.asarray(obs, dtype = float))

        if render:
            frames.append(env.render())

        if term or trunc:
            success = bool(term and r >= 50.)
            break

    env.close()
    return dict(success = success, reward = total, steps = len(states) - 1, states = np.stack(states), frames = frames)

def label_frame(frame, text, size = 15):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, size + 8)], fill = (255, 255, 255))
    draw.text((6, 4), text, fill = (0, 0, 0))
    return np.asarray(img)

def scale_frame(frame, scale):
    img = Image.fromarray(frame)
    return np.asarray(img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS))

def montage(landings, path, columns = 2, tile_scale = 0.5, fps = 25):
    # a tile per landing, last frame held so all tiles have the same length, then
    # (tiles frames h w k) -> (frames rows*h cols*w k)

    rows = (len(landings) + columns - 1) // columns
    length = max(len(l['frames']) for l in landings)
    tiles = []

    for landing in landings:
        label = f'bank reset #{landing["config"]} - landed in {landing["steps"]} steps'
        tile = [scale_frame(label_frame(frame, f'{label}   t = {k / FPS:.2f}s'), tile_scale) for k, frame in enumerate(landing['frames'])]
        tiles.append([*tile, *[tile[-1]] * (length - len(tile))])

    blank = np.full_like(tiles[0][0], 255)
    tiles = [*tiles, *[[blank] * length] * (rows * columns - len(tiles))]  # fill the grid

    frames = rearrange(np.stack([np.stack(tile, 0) for tile in tiles]), '(r c) f h w k -> f (r h) (c w) k', r = rows)
    imageio.mimsave(path, frames, fps = fps, loop = 0)
    print(f'saved gif to {path} ({len(frames)} frames)')

def contact_sheet(landings, path, snapshots = 5):
    rows = len(landings)
    fig, axes = plt.subplots(rows, snapshots, figsize = (3.1 * snapshots, 2.3 * rows), dpi = 120, squeeze = False)

    for row, landing in enumerate(landings):
        frames, states = landing['frames'], landing['states']

        for col, k in enumerate(np.linspace(0, len(frames) - 1, snapshots).round().astype(int)):
            ax = axes[row, col]
            ax.imshow(frames[k])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'#{landing["config"]}  t = {k / FPS:.2f}s  alt {states[k][1]:.2f}  vy {states[k][3]:+.2f}', fontsize = 7.5)

        axes[row, 0].set_ylabel(f'landed in {landing["steps"]} steps\nreward {landing["reward"]:+.0f}', fontsize = 8)

    fig.suptitle('LunarLander landed by the SGS-trained policy (greedy rollouts from bank reset states)', fontsize = 11)
    fig.tight_layout(rect = (0., 0., 1., 0.96))
    fig.savefig(path)
    print(f'saved contact sheet to {path}')

def path_plot(landings, path, num_states = 20):
    fig, ax = plt.subplots(figsize = (7.5, 4.2), dpi = 140)

    for landing in landings:
        s = landing['states']
        ax.plot(s[:, 0], s[:, 1], linewidth = 1.8, label = f'bank reset #{landing["config"]}  (difficulty {landing["config"] / (num_states - 1):.2f})')
        ax.scatter(s[0, 0], s[0, 1], s = 22, marker = 'o', zorder = 5, edgecolor = 'black', linewidth = 0.6)
        ax.scatter(s[-1, 0], s[-1, 1], s = 40, marker = '*', zorder = 5, color = 'gold', edgecolor = 'black', linewidth = 0.6)

    ax.axhline(0., color = 'black', linewidth = 1.)
    ax.axvspan(-0.4, 0.4, color = 'gold', alpha = 0.25)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.05, 1.45)
    ax.set_xlabel('x offset from pad center'); ax.set_ylabel('altitude above the pad')
    ax.set_title('descent paths of the landed trajectories\n(circles = reset state, stars = touchdown, shaded band = pad)', fontsize = 10)
    ax.grid(alpha = 0.25); ax.legend(fontsize = 8, loc = 'upper left')
    fig.tight_layout()
    fig.savefig(path)
    print(f'saved path plot to {path}')

def render_landings(actor, info, run_dir, num_landings = 4):
    # greedily score the bank, then animate the hardest landings

    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # rgb_array rendering needs no display

    bank = build_bank()
    scored = [record_landing(actor, state, render = False)['success'] for state in bank]
    landed_indices = [i for i, ok in enumerate(scored) if ok]
    print(f'best SGS policy (seed {info["seed"]}) lands {len(landed_indices)}/{len(bank)} bank states greedily')

    landings = []

    for i in landed_indices[-num_landings:]:
        landing = dict(config = i, **record_landing(actor, bank[i], seed = 0))
        print(f'  reset #{i:2d} -> landed in {landing["steps"]} steps, reward {landing["reward"]:+.0f}')
        landings.append(landing)

    montage(landings, run_dir / 'lunar_sgs_landings.gif')
    contact_sheet(landings, run_dir / 'lunar_sgs_trajectories.png')
    path_plot(landings, run_dir / 'lunar_sgs_paths.png', num_states = len(bank))

# entry point

def as_list(value):
    # fire hands back ints, tuples or comma separated strings depending on the call
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(',') if v.strip()]

def main(
    steps: int = 200_000,
    seeds: str = '0,1,2,3,4',
    arms: str = 'native,uniform,sgs',
    num_envs: int = 16,
    rollout: int = 128,
    eval_every: int = 4,
    lr: float = 1e-3,
    gamma: float = 0.999,
    epochs: int = 10,
    run: str = 'lunar_sgs',
    render: bool = True,
    rerun: bool = False,
):
    run_dir = Path('experiments') / str(run)
    run_dir.mkdir(parents = True, exist_ok = True)
    results_path = run_dir / 'results.json'

    arm_list, seed_list = [str(a) for a in as_list(arms)], [int(s) for s in as_list(seeds)]
    runs = json.loads(results_path.read_text()) if results_path.exists() and not rerun else []

    for arm in arm_list:
        for seed in seed_list:
            if any(r['arm'] == arm and r['seed'] == seed and r['steps'] == steps for r in runs):
                print(f'skipping {arm} seed {seed} (cached)')
                continue

            print(f'=== {arm} | seed {seed} ===')
            result = train(
                arm = arm, steps = steps, num_envs = num_envs, rollout = rollout, seed = seed,
                eval_every = eval_every, lr = lr, gamma = gamma, epochs = epochs,
                progression_path = run_dir / f'sgs_progression_{arm}_{seed}.csv',
                checkpoint_path = run_dir / f'actor_{arm}_{seed}.pt',
            )
            runs.append(result)
            results_path.write_text(json.dumps(runs))
            print(f'=== {arm} | seed {seed} done in {result["time"]:.1f}s ===\n')

    plot(runs, steps = steps, path = run_dir / 'lunar_sgs.png')

    sgs_runs = runs_of(runs, 'sgs', steps)

    if render and sgs_runs:
        best = max(sgs_runs, key = lambda r: r['history']['bank'][-1])
        render_landings(load_actor(run_dir / f'actor_sgs_{best["seed"]}.pt')[0], best, run_dir)

    print(f'\nresults in {run_dir}/')
    return runs

if __name__ == '__main__':
    fire.Fire(main)
