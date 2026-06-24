import pytest
import numpy as np
import torch

from env_ssl_wrapper.action_transform_wrapper import ActionTransformWrapper

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
