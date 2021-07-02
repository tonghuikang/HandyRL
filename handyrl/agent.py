# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player)
        else:
            return random.choice(env.legal_actions(player))


class AgentBoilergoose(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_boilergoose'):
            return env.agent_boilergoose(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentCrazyGoose(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_crazy_goose'):
            return env.agent_crazy_goose(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentGreedy(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_greedy'):
            return env.agent_greedy(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentRiskAdverseGreedy(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_risk_adverse_greedy'):
            return env.agent_risk_adverse_greedy(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentPubHRL(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_pubhrl'):
            return env.agent_pubhrl(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentPubHRLLatest(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_pubhrl_latest'):
            return env.agent_pubhrl_latest(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentSimpleBFS(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_simple_bfs'):
            return env.agent_simple_bfs(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentSimpleToward(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_simple_toward'):
            return env.agent_simple_toward(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


class AgentStraightforwardBFS(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'agent_straightforward_bfs'):
            return env.agent_straightforward_bfs(player)
        else:
            assert False
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        print('v = %f' % v)
        print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, observation=False, temperature=0.0):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.observation = observation
        self.temperature = temperature

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        outputs = self.plan(env.observation(player))
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p = p - mask * 1e32

        if show:
            print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            outputs = self.plan(env.observation(player))
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v if v is not None else [0.0]


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o:
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [o]
        for k, vl in outputs:
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model, observation=False):
        super().__init__(model, observation=observation, temperature=1.0)
