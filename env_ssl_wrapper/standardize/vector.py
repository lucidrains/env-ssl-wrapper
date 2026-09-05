from __future__ import annotations

from multiprocessing import get_context
import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_structure, tree_unflatten

from .helpers import any_true, dones_of, exists, get_attr, instantiate_env, safe_close, truthy_attr
from .spaces import action_dim_of
from .standardize_wrapper import StandardizeWrapper

# leaf helpers

def _zero_leaf(x):
    if is_tensor(x):
        return torch.zeros_like(x)
    return np.zeros_like(x)

def _stack_leaves(leaves):
    if all(map(is_tensor, leaves)):
        return torch.stack(leaves)
    return np.stack(leaves)

def _stack_trees(trees):
    first = trees[0]

    if is_tensor(first):
        return torch.stack(trees)

    if isinstance(first, np.ndarray):
        return np.stack(trees)

    if isinstance(first, dict):
        return {key: _stack_trees([t[key] for t in trees]) for key in first}

    if isinstance(first, tuple):
        return tuple(_stack_trees([t[i] for t in trees]) for i in range(len(first)))

    leaves = [tree_flatten(tree)[0] for tree in trees]
    stacked = [_stack_leaves(col) for col in zip(*leaves)]
    return tree_unflatten(stacked, tree_structure(trees[0]))

def _merge_infos(infos):
    infos = [info if isinstance(info, dict) else {} for info in infos]

    if not infos:
        return {}

    keys = set(infos[0])
    for info in infos[1:]:
        keys &= set(info)

    out = {}
    for key in keys:
        vals = [info[key] for info in infos]
        try:
            out[key] = _stack_trees(vals)
        except Exception:
            out[key] = np.array(vals, dtype = object)

    return out

# worker protocol

class WorkerError(RuntimeError):
    pass

def _exec(env, cmd, payload):
    if cmd == 'step':
        obs, reward, terminated, truncated, info = env.step(payload)

        if any_true(dones_of(terminated, truncated)):
            if truthy_attr(get_attr(env.adapter, 'autoresets')):
                final_obs = info.get('final_observation', obs)
            else:
                obs, final_obs = env.reset()[0], obs
        else:
            final_obs = None

        return obs, reward, terminated, truncated, info, final_obs

    if cmd == 'reset':
        kwargs = payload or {}
        return env.reset(**kwargs)

    if cmd == 'seed':
        try:
            env.seed(payload)
        except Exception:
            pass
        return None

    if cmd == 'get_attr':
        return get_attr(env, payload)

    if cmd == 'close':
        safe_close(env)
        return None

def _worker_main(conn, env_factory):
    try:
        env = StandardizeWrapper(instantiate_env(env_factory))
    except Exception as e:
        conn.send(WorkerError(f'worker failed to initialize env: {e}'))
        return

    conn.send(None)

    while True:
        try:
            cmd, payload = conn.recv()
        except (EOFError, BrokenPipeError):
            return

        try:
            result = _exec(env, cmd, payload)
        except Exception as e:
            result = WorkerError(str(e))

        conn.send(result)

        if cmd == 'close':
            return

# pipe helpers

class _InlineConn:
    def __init__(self, env):
        self.env = env

    def send(self, msg):
        try:
            self._result = _exec(self.env, *msg)
        except Exception as e:
            self._result = WorkerError(str(e))

    def recv(self):
        return self._result

def _safe_send(conn, msg):
    """Send that swallows broken-pipe — for broadcast patterns where _recv_all reports the error."""
    try:
        conn.send(msg)
    except (EOFError, BrokenPipeError):
        pass

def _send_recv(conn):
    """Single-channel send+recv — raises immediately on dead worker."""
    try:
        result = conn.recv()
    except (EOFError, BrokenPipeError) as e:
        raise WorkerError(f'worker process terminated unexpectedly: {e}')

    if isinstance(result, WorkerError):
        raise result

    return result

def _recv_all(conns):
    """Drain all channels, collecting the first error — never abandons undrained pipes."""
    results = []
    error = None

    for conn in conns:
        try:
            result = conn.recv()
        except (EOFError, BrokenPipeError) as e:
            result = WorkerError(f'worker process terminated unexpectedly: {e}')

        if isinstance(result, WorkerError):
            error = error or result
        else:
            results.append(result)

    if exists(error):
        raise error

    return results

def _shutdown(conns, procs):
    for conn in conns:
        try:
            conn.send(('close', None))
        except (EOFError, BrokenPipeError, OSError):
            pass

    for proc in procs:
        if proc.is_alive():
            proc.join(timeout = 0.2)

        if proc.is_alive():
            proc.terminate()

    for conn in conns:
        try:
            conn.close()
        except Exception:
            pass

def _split_actions(actions, num_envs):
    if isinstance(actions, dict):
        assert all(len(v) == num_envs for v in actions.values()), f'expected {num_envs} actions per key'
        return [{k: v[i] for k, v in actions.items()} for i in range(num_envs)]

    if isinstance(actions, tuple) and not is_tensor(actions) and not isinstance(actions, np.ndarray):
        assert all(len(v) == num_envs for v in actions), f'expected {num_envs} actions per element'
        return [tuple(elem[i] for elem in actions) for i in range(num_envs)]

    assert len(actions) == num_envs, f'expected {num_envs} actions, but got {len(actions)}'
    return actions

# class

class MultiprocessingVecEnv:
    autoresets = True
    is_vector = True

    def __init__(
        self,
        env_factory,
        num_envs: int = 8,
        seed: int = 0
    ):
        self.num_envs = num_envs

        if num_envs == 1:
            self._conns = [_InlineConn(StandardizeWrapper(instantiate_env(env_factory)))]
            self._procs = []
        else:
            self._conns, self._procs = self._start_processes(env_factory, num_envs)

        self.action_space = self._get_attr('action_space')
        self.observation_space = self._get_attr('observation_space')
        self.action_dim = action_dim_of(self)

        self.seed(seed)

    @staticmethod
    def _start_processes(env_factory, num_envs):
        ctx = get_context('spawn')
        conns, procs = [], []

        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target = _worker_main, args = (child, env_factory), daemon = True)
            proc.start()
            child.close()
            conns.append(parent)
            procs.append(proc)

        try:
            _recv_all(conns)
        except WorkerError as e:
            _shutdown(conns, procs)
            raise RuntimeError(str(e))

        return conns, procs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def single_action_space(self):
        return self.action_space

    @property
    def single_observation_space(self):
        return self.observation_space

    def _get_attr(self, name):
        conn = self._conns[0]
        conn.send(('get_attr', name))
        return _send_recv(conn)

    def seed(self, seed: int | None):
        if not exists(seed):
            return

        for conn, env_seed in zip(self._conns, range(seed, seed + self.num_envs)):
            _safe_send(conn, ('seed', env_seed))

        _recv_all(self._conns)

    def reset(self, seed = None, **kwargs):
        if exists(seed):
            self.seed(seed)

        for conn in self._conns:
            _safe_send(conn, ('reset', kwargs))

        results = _recv_all(self._conns)
        return _stack_trees([obs for obs, _ in results]), _merge_infos([info for _, info in results])

    def step(self, actions):
        actions = _split_actions(actions, self.num_envs)

        for conn, action in zip(self._conns, actions):
            _safe_send(conn, ('step', action))

        results = _recv_all(self._conns)

        obs = [result[0] for result in results]
        reward = np.asarray([result[1] for result in results])
        terminated = np.asarray([result[2] for result in results], dtype = bool)
        truncated = np.asarray([result[3] for result in results], dtype = bool)
        infos = [result[4] for result in results]
        final_obs = [result[5] for result in results]

        info = _merge_infos(infos)
        done = terminated | truncated

        if done.any():
            final_obs = [
                final if exists(final) else tree_map(_zero_leaf, obs_i)
                for final, obs_i in zip(final_obs, obs)
            ]

            info['final_observation'] = _stack_trees(final_obs)
            info['_final_observation'] = done

        return _stack_trees(obs), reward, terminated, truncated, info

    def close(self):
        _shutdown(self._conns, self._procs)
