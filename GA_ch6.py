import numpy as np
import torch
import gym


env = gym.make("CartPole-v0")


def model(x, unpacked_params):
    l1, b1, l2, b2, l3, b3 = unpacked_params
    y = torch.nn.functional.linear(x, l1, b1)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l2, b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l3, b3)
    y = torch.log_softmax(y, dim=0)
    return y


def unpack_params(params, layers=[(25, 4), (10, 25), (2, 10)]):
    unpacked_params = []
    e = 0
    for i, l in enumerate(layers):
        s, e = e, e+np.prod(l)
        weights = params[s:e].view(l)
        s, e = e, e + l[0]
        bias = params[s:e]
        unpacked_params.extend([weights, bias])
    return unpacked_params


def spawn_population(N=50, size=407):
    pop = []
    for i in range(N):
        vec = torch.randn(size) / 2.0
        fit = 0
        p = {'params': vec, 'fitness': fit}
        pop.append(p)
    return pop


def recombine(x1, x2):
    x1 = x1['params']
    x2 = x2['params']
    l = x1.shape[0]
    split_pt = np.random.randint(l)
    child1 = torch.zeros(l)
    child2 = torch.zeros(l)
    child1[:split_pt] = x1[:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[:split_pt] = x2[:split_pt]
    child2[split_pt:] = x1[split_pt:]

    c1 = {'params': child1, 'fitness': 0.0}
    c2 = {'params': child2, 'fitness': 0.0}

    return c1, c2


def mutate(x, rate=0.01):
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0])
    idx = np.random.randint(low=0, high=x_.shape[0], size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0
    x['params'] = x_
    return x


def test_model(agent):
    done = False
    state = torch.from_numpy(env.reset()).float()
    score = 0
    while not done:
        params = unpack_params(agent['params'])
        probs = model(state, params)
        action = torch.distributions.Categorical(probs=probs).sample()
        state_, reward, done, info = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1
    return score

