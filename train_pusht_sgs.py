# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium",
#     "gym-pusht",
#     "numpy",
#     "opencv-python",
#     "torch",
#     "x-mlps-pytorch",
#     "x-ppo>=0.0.7",
# ]
# ///

from __future__ import annotations

from collections import deque
import time
import cv2
import fire
import gymnasium as gym
import gym_pusht  # noqa: F401
from gym_pusht.envs.pusht import PushTEnv
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
    return obs, min(max(cov / self.success_threshold, 0.), 1.), is_success, False, dict(is_success = is_success)

def make_pusht(max_steps = 50):
    env = gym.make('gym_pusht/PushT-v0')
    env.unwrapped.step = fast_step.__get__(env.unwrapped, PushTEnv)
    return StandardizeEnvWrapper(env, auto_batch = False, device = None, done_tracker = False, pad_episodes = False, max_timesteps = max_steps)

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

# ppo training with SuccessGuidedSamplingWrapper

def train(method = 'sgs', steps = 32_768, num_envs = 16, rollout = 128, seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    configs = build_configs()

    sampler = SGSSampler(
        configs = configs,
        target = 0.15,
        kappa = 3.0,
        eps = 0.06,
        window = 10,
        unvisited_weight = 0.4,
        uniform_prob = 0.10 if method == 'sgs' else 0.0,
        score_fn = uniform_score_fn if method == 'uniform' else None,
        seed = seed,
    )
    envs = [SuccessGuidedSamplingWrapper(make_pusht(), reset_configs = sampler) for _ in range(num_envs)]
    eval_envs = [make_pusht() for _ in configs]

    actor = MLP(22, 128, 128, 4, activation = nn.SiLU())
    critic = MLP(22, 128, 128, 1, activation = nn.SiLU(), squeeze_out = True)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr = 3e-4)
    succ_buf: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen = 20)

    def get_dist(obs_t):
        out = actor(featurize(obs_t))
        m, c = out[..., :2].sigmoid() * 0.90 + 0.05, 4. + (out[..., 2:] - 1.).sigmoid() * 60.
        return Beta(1. + m * c, 1. + (1. - m) * c), m

    @torch.no_grad()
    def evaluate(eval_episodes = 3):
        # evaluate sampled actions, matching the stochastic policy the learner optimizes
        # (the beta mean can sit off-frontier and read ~0% while sampled attempts succeed)

        succ = np.zeros((eval_episodes, len(configs)), dtype = np.float32)

        for ep in range(eval_episodes):
            obs = np.stack([e.reset(options = dict(reset_to_state = c))[0] for e, c in zip(eval_envs, configs)]).astype(np.float32)
            done = np.zeros(len(configs), dtype = bool)

            for _ in range(50):
                if done.all():
                    break

                dist, _ = get_dist(torch.from_numpy(obs))
                act_np = dist.sample().clamp(0.01, 0.99).numpy()

                for i in np.where(~done)[0]:
                    obs[i], _, term, trunc, _ = eval_envs[i].step(to_env_action(obs[i], act_np[i]))
                    if term or trunc:
                        succ[ep, i], done[i] = float(term), True

        per_config = succ.mean(0)
        return float(per_config.mean()), float(per_config[10:].mean())

    obs = np.stack([e.reset()[0] for e in envs]).astype(np.float32)
    ep_s, ep_a = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]
    t0 = time.time()

    for upd in range(1, steps // (num_envs * rollout) + 1):
        S, A, LP, R, V, M = [], [], [], [], [], []
        for _ in range(rollout):
            obs_t = torch.from_numpy(obs)
            with torch.no_grad():
                dist, _ = get_dist(obs_t)
                act = dist.sample().clamp(0.01, 0.99)
                lp, val = dist.log_prob(act).sum(-1), critic(featurize(obs_t)).sigmoid()
            env_act = to_env_action(obs, act.numpy())
            next_obs, rew, mask = np.empty_like(obs), np.zeros(num_envs, dtype = np.float32), np.ones(num_envs, dtype = np.float32)
            for i, e in enumerate(envs):
                no, cov_r, term, trunc, _ = e.step(env_act[i])
                ep_s[i].append(obs[i]); ep_a[i].append(act[i].numpy())
                rew[i] = 0.1 * float(cov_r) + float(term)
                if trunc and not term:
                    with torch.no_grad():
                        rew[i] += 0.98 * float(critic(featurize(torch.from_numpy(no.astype(np.float32)))).sigmoid())
                if term or trunc:
                    if term:
                        sa, aa = np.stack(ep_s[i]), np.stack(ep_a[i])
                        succ_buf.append((sa, aa, 0.98 ** np.arange(len(sa) - 1, -1, -1, dtype = np.float32)))
                    ep_s[i].clear(); ep_a[i].clear(); mask[i] = 0.
                    no, _ = e.reset()
                next_obs[i] = no
            S.append(obs_t); A.append(act); LP.append(lp); R.append(torch.from_numpy(rew)); V.append(val); M.append(torch.from_numpy(mask))
            obs = next_obs

        with torch.no_grad():
            last_v = critic(featurize(torch.from_numpy(obs))).sigmoid()
            rets, advs = calc_gae(torch.stack(R).T, torch.stack(V).T, torch.stack(M).T, next_value = last_v, gamma = 0.98, lam = 0.95, use_accelerated = False, return_advantages = True)
            rets, advs = rets.T.reshape(-1).clamp(0., 1.2), advs.T.reshape(-1)
            advs = (advs - advs.mean()) / advs.std().clamp(min = 0.08)

        Sf, Af, LPf = torch.cat(S), torch.cat(A), torch.cat(LP)
        for _ in range(6):
            for idx in torch.randperm(len(Sf)).split(256):
                dist, _ = get_dist(Sf[idx])
                pg = ppo_actor_loss(dist.log_prob(Af[idx]).sum(-1), LPf[idx], advs[idx], 0.2, normalize_advantages = False).mean()
                vf = F.mse_loss(critic(featurize(Sf[idx])).sigmoid(), rets[idx].clamp(0., 1.))
                loss = pg + 0.5 * vf - 5e-4 * dist.entropy().sum(-1).mean()
                if succ_buf:
                    bs = torch.from_numpy(np.concatenate([v[0] for v in succ_buf]))
                    ba = torch.from_numpy(np.concatenate([v[1] for v in succ_buf])).clamp(0.01, 0.99)
                    br = torch.from_numpy(np.concatenate([v[2] for v in succ_buf]))
                    bdist, bmean = get_dist(bs)
                    loss = loss + 0.25 * ((-bdist.log_prob(ba).sum(-1) + 6. * (bmean - ba).pow(2).sum(-1)) * br).mean() + 0.15 * F.mse_loss(critic(featurize(bs)).sigmoid(), br)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5); opt.step()

        if upd % 4 == 0:
            all_sr, hard_sr = evaluate()
            f_mass = float(sampler.probs[10:].sum())
            print(f'[{method:>7}] step {upd * num_envs * rollout:6d} | eval_all: {all_sr:5.1%} | eval_frontier: {hard_sr:5.1%} | frontier_prob: {f_mass:5.1%} | time: {time.time() - t0:4.1f}s', flush = True)

def main(steps: int = 32_768, seed: int = 0):
    for m in ('uniform', 'sgs'):
        train(method = m, steps = steps, seed = seed)

if __name__ == '__main__':
    fire.Fire(main)
