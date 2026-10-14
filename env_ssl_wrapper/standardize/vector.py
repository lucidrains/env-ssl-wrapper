from __future__ import annotations

from multiprocessing import get_context
import numpy as np
from torch import is_tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .helpers import (
    any_true,
    default,
    dones_of,
    exists,
    get_adapter,
    get_attr,
    instantiate_env,
    safe_close,
    stack_trees,
    to_numpy,
    truthy_attr,
    zero_leaf,
)
from .spaces import action_dim_of, action_space_is_discrete, obs_dim_of
from .standardize_wrapper import StandardizeWrapper

# leaf helpers

def merge_infos(infos):
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
            out[key] = stack_trees(vals)
        except Exception:
            out[key] = np.array(vals, dtype = object)

    return out

# worker protocol

class WorkerError(RuntimeError):
    pass

def exec_worker_cmd(env, cmd, payload):
    if cmd == 'step':
        if not get_adapter(env).torch_native:
            payload = tree_map(lambda a: to_numpy(a) if is_tensor(a) else a, payload)

        if action_space_is_discrete(env.action_space):
            if hasattr(payload, 'item') and (not hasattr(payload, 'ndim') or payload.ndim <= 1):
                payload = payload.item()
            elif isinstance(payload, (list, tuple)) and len(payload) == 1:
                payload = payload[0]

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

def worker_main(conn, env_factory):
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
            result = exec_worker_cmd(env, cmd, payload)
        except Exception as e:
            result = WorkerError(str(e))

        conn.send(result)

        if cmd == 'close':
            return

# pipe helpers

class InlineConn:
    def __init__(self, env):
        self.env = env

    def send(self, msg):
        try:
            self._result = exec_worker_cmd(self.env, *msg)
        except Exception as e:
            self._result = WorkerError(str(e))

    def recv(self):
        return self._result

    def close(self):
        pass

def safe_send(conn, msg):
    # swallows broken-pipe for broadcast patterns where recv_all reports the error
    try:
        conn.send(msg)
    except (EOFError, BrokenPipeError):
        pass

def send_recv(conn):
    # single-channel send + recv — raises immediately on dead worker
    try:
        result = conn.recv()
    except (EOFError, BrokenPipeError) as e:
        raise WorkerError(f'worker process terminated unexpectedly: {e}')

    if isinstance(result, WorkerError):
        raise result

    return result

def recv_all(conns):
    # drain all channels, collecting the first error — never abandons undrained pipes
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

def shutdown(conns, procs):
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

def split_actions(actions, num_envs):
    if isinstance(actions, list) and len(actions) == num_envs:
        return actions

    leaves, spec = tree_flatten(actions)
    assert all(len(leaf) == num_envs for leaf in leaves), (
        f'expected {num_envs} actions, but got {len(leaves[0]) if leaves else 0}'
    )
    return [tree_unflatten([leaf[i] for leaf in leaves], spec) for i in range(num_envs)]

_split_actions = split_actions
_merge_infos = merge_infos

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
            self._conns = [InlineConn(StandardizeWrapper(instantiate_env(env_factory)))]
            self._procs = []
        else:
            self._conns, self._procs = self._start_processes(env_factory, num_envs)

        self.action_space = self._get_attr('action_space')
        self.observation_space = self._get_attr('observation_space')
        self.action_dim = default(action_dim_of(self), self._get_attr('action_dim'))
        self.obs_dim = default(obs_dim_of(self), self._get_attr('obs_dim'))

        self.seed(seed)

    @staticmethod
    def _start_processes(env_factory, num_envs):
        ctx = get_context('spawn')
        conns, procs = [], []

        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target = worker_main, args = (child, env_factory), daemon = True)
            proc.start()
            child.close()
            conns.append(parent)
            procs.append(proc)

        try:
            recv_all(conns)
        except WorkerError as e:
            shutdown(conns, procs)
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
        return send_recv(conn)

    def seed(self, seed: int | None):
        if not exists(seed):
            return

        for conn, env_seed in zip(self._conns, range(seed, seed + self.num_envs)):
            safe_send(conn, ('seed', env_seed))

        recv_all(self._conns)

    def reset_at(self, index: int, seed = None, **kwargs):
        worker_kwargs = dict(kwargs)
        if exists(seed):
            worker_kwargs['seed'] = seed
        safe_send(self._conns[index], ('reset', worker_kwargs))
        return send_recv(self._conns[index])

    def reset(self, seed = None, **kwargs):
        options = kwargs.get('options')

        for i, conn in enumerate(self._conns):
            worker_kwargs = dict(kwargs)
            if exists(seed):
                worker_kwargs['seed'] = seed + i

            if isinstance(options, (list, tuple)) and len(options) == self.num_envs:
                worker_kwargs['options'] = options[i]
            elif isinstance(options, dict):
                worker_kwargs['options'] = {
                    k: (v[i] if isinstance(v, (list, tuple)) and len(v) == self.num_envs else v)
                    for k, v in options.items()
                }

            safe_send(conn, ('reset', worker_kwargs))

        results = recv_all(self._conns)

        if not exists(self.observation_space):
            self.observation_space = self._get_attr('observation_space')
            self.obs_dim = default(self.obs_dim, obs_dim_of(self))

        return stack_trees([obs for obs, _ in results]), merge_infos([info for _, info in results])

    def step(self, actions):
        actions = tree_map(lambda a: to_numpy(a) if is_tensor(a) else a, actions)
        actions = split_actions(actions, self.num_envs)

        for conn, action in zip(self._conns, actions):
            safe_send(conn, ('step', action))

        results = recv_all(self._conns)

        obs = [result[0] for result in results]
        reward = np.asarray([result[1] for result in results])
        terminated = np.asarray([result[2] for result in results], dtype = bool)
        truncated = np.asarray([result[3] for result in results], dtype = bool)
        infos = [result[4] for result in results]
        final_obs = [result[5] for result in results]

        info = merge_infos(infos)
        done = terminated | truncated

        if done.any():
            final_obs = [
                final if exists(final) else tree_map(zero_leaf, obs_i)
                for final, obs_i in zip(final_obs, obs)
            ]

            info['final_observation'] = stack_trees(final_obs)
            info['_final_observation'] = done

        return stack_trees(obs), reward, terminated, truncated, info

    def close(self):
        shutdown(self._conns, self._procs)
