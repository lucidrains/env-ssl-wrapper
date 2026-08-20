import pytest
import numpy as np
import torch

from env_ssl_wrapper.action_transform_wrapper import ActionTransformWrapper
from env_ssl_wrapper.mocks import GymnasiumMockEnv, GymnasiumDiscreteMockEnv, DMControlMockEnv, Space

class MockEnv:
    def step(self, action):
        self.last_action = action
        return np.zeros(4), 0.0, False, False, {}

def test_action_transform_wrapper():
    env = MockEnv()

    transforms = [
        dict(indices=slice(0, 2), rescale_from_to=((0., 1.), (-1., 1.))),
        dict(indices=slice(2, 4), rescale_from_to=((0., 1.), (0., 2.)))
    ]

    wrapper = ActionTransformWrapper(env, transforms=transforms, clip=(-1., 1.5))

    # test numpy

    wrapper.step(np.array([0., 1., 0.5, 2.], dtype=np.float32))
    assert np.allclose(env.last_action, np.array([-1., 1., 1., 1.5]))

    # test torch

    wrapper.step(torch.tensor([0., 1., 0.5, 2.], dtype=torch.float32))
    assert torch.allclose(env.last_action, torch.tensor([-1., 1., 1., 1.5]))

    # test tree map with dict

    wrapper.step(dict(
        continuous = torch.tensor([0., 1., 0.5, 2.], dtype=torch.float32),
        discrete = torch.tensor([5, 10], dtype=torch.int64)
    ))

    assert torch.allclose(env.last_action['continuous'], torch.tensor([-1., 1., 1., 1.5]))
    assert torch.allclose(env.last_action['discrete'], torch.tensor([5, 10]))

    # test implicit indices

    implicit_transforms = [
        dict(rescale_from_to=((0., 1.), (-1., 1.))),
        dict(rescale_from_to=((0., 1.), (0., 2.)))
    ]

    wrapper_implicit = ActionTransformWrapper(env, transforms=implicit_transforms)
    wrapper_implicit.step(np.array([0., 1.], dtype=np.float32))

    assert np.allclose(env.last_action, np.array([-1., 2.]))

# a single transform without indices applies to the whole action vector

def test_single_transform_applies_to_whole_action():
    env = MockEnv()
    transform = dict(rescale_from_to = ((0., 1.), (-1., 1.)))

    wrapper = ActionTransformWrapper(env, transforms = transform)
    wrapper.step(np.array([0., 1.], dtype = np.float32))

    assert np.allclose(env.last_action, np.array([-1., 1.]))

# auto rescale — policy emits in canonical (0, 1) (beta-friendly),
# wrapper maps to any env's action bounds, leaving unbounded dims untouched

class BoundedMockEnv(GymnasiumMockEnv):
    def __init__(self, low, high):
        super().__init__()
        self.low = np.asarray(low, dtype = float)
        self.high = np.asarray(high, dtype = float)

    @property
    def action_space(self):
        return Space((self.action_dim,), self.low, self.high)

def test_auto_rescale_uniform_bounds():
    env = ActionTransformWrapper(BoundedMockEnv(-2., 2.), auto = True)
    env.step(np.array([0.25, 1.0], dtype = np.float32))

    assert np.allclose(env.unwrapped.last_action, np.array([-1.0, 2.0]))

def test_auto_rescale_per_dim_bounds():
    env = ActionTransformWrapper(BoundedMockEnv([-2., 0.], [0., 4.]), auto = True)
    env.step(np.array([0.5, 0.5], dtype = np.float32))

    assert np.allclose(env.unwrapped.last_action, np.array([-1.0, 2.0]))

def test_auto_rescale_unbounded_dims_passthrough():
    env = ActionTransformWrapper(BoundedMockEnv([-np.inf, -2.], [np.inf, 2.]), auto = True)
    env.step(np.array([0.5, 0.5], dtype = np.float32))

    assert np.allclose(env.unwrapped.last_action, np.array([0.5, 0.0]))

def test_auto_rescale_torch():
    env = ActionTransformWrapper(BoundedMockEnv(-2., 2.), auto = True)
    env.step(torch.tensor([0.25, 1.0]))

    assert torch.allclose(torch.from_numpy(env.unwrapped.last_action), torch.tensor([-1.0, 2.0]))

def test_auto_rescale_discrete_passthrough():
    env = ActionTransformWrapper(GymnasiumDiscreteMockEnv(), auto = True)
    env.step(np.int64(1))

    assert env.unwrapped.last_action == 1

def test_auto_rescale_dm_control_action_spec():
    env = ActionTransformWrapper(DMControlMockEnv(), auto = True)
    env.step(np.array([0.0, 1.0], dtype = np.float32))

    assert np.allclose(env.unwrapped.last_action, np.array([-1.0, 1.0]))
