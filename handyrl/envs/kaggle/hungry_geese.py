# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import random, math
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# You need to install kaggle_environments, requests
from kaggle_environments import make

from ...environment import BaseEnvironment


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn, padding_mode="circular", stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, padding_mode=padding_mode, padding=1, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, h):
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        f1, f2, f3, f4 = 16, 24, 32, 40

        self.conv0 = TorusConv2d(5, f1, (3, 3), True)

        self.blocks_conv1 = nn.ModuleList([TorusConv2d(f1, f1, (3, 3), True, padding_mode="circular") for _ in range(2)])
        self.down1 = TorusConv2d(f1, f2, (3, 3), True, padding_mode="circular", stride=2)
        self.down1a = nn.Conv2d(f1, f2, kernel_size=1, stride=2, bias=False)

        self.blocks_conv2 = nn.ModuleList([TorusConv2d(f2, f2, (3, 3), True, padding_mode="circular") for _ in range(2)])
        self.down2 = TorusConv2d(f2, f3, (3, 3), True, padding_mode="circular", stride=2)
        self.down2a = nn.Conv2d(f2, f3, kernel_size=1, stride=2, bias=False)

        self.blocks_conv3 = nn.ModuleList([TorusConv2d(f3, f3, (3, 3), True, padding_mode="circular") for _ in range(2)])
        self.down3 = TorusConv2d(f3, f4, (3, 3), True, padding_mode="circular", stride=2)
        self.down3a = nn.Conv2d(f3, f4, kernel_size=1, stride=2, bias=False)

        self.head_p = nn.Linear(f3, 4, bias=False)
        self.head_v = nn.Linear(f3 * 2, 1, bias=False)


    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))

        for block in self.blocks_conv1:
            h = F.relu_(h + block(h))
        h = F.relu_(self.down1a(h) + self.down1(h))

        for block in self.blocks_conv2:
            h = F.relu_(h + block(h))
        h = F.relu_(self.down2a(h) + self.down2(h))

        for block in self.blocks_conv3:
            h = F.relu_(h + block(h))

        h_head = h[:,:,0,0]
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
    
        return {'policy': p, 'value': v}


class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    NUM_AGENTS = 4

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.reset()

    def reset(self, args={}):
        obs = self.env.reset(num_agents=self.NUM_AGENTS)
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_from is None or pos_to is None:
            return None
        x, y = pos_from // 11, pos_from % 11
        for i, d in enumerate(self.DIRECTION):
            nx, ny = (x + d[0]) % 7, (y + d[1]) % 11
            if nx * 11 + ny == pos_to:
                return i
        return None

    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]['observation']
        colors = ['\033[33m', '\033[34m', '\033[32m', '\033[31m']
        color_end = '\033[0m'

        def check_cell(pos):
            for i, geese in enumerate(obs['geese']):
                if pos in geese:
                    if pos == geese[0]:
                        return i, 'h'
                    if pos == geese[-1]:
                        return i, 't'
                    index = geese.index(pos)
                    pos_prev = geese[index - 1] if index > 0 else None
                    pos_next = geese[index + 1] if index < len(geese) - 1 else None
                    directions = [self.direction(pos, pos_prev), self.direction(pos, pos_next)]
                    return i, directions
            if pos in obs['food']:
                return 'f'
            return None

        def cell_string(cell):
            if cell is None:
                return '.'
            elif cell == 'f':
                return 'f'
            else:
                index, directions = cell
                if directions == 'h':
                    return colors[index] + '@' + color_end
                elif directions == 't':
                    return colors[index] + '*' + color_end
                elif max(directions) < 2:
                    return colors[index] + '|' + color_end
                elif min(directions) >= 2:
                    return colors[index] + '-' + color_end
                else:
                    return colors[index] + '+' + color_end

        cell_status = [check_cell(pos) for pos in range(7 * 11)]

        s = 'turn %d\n' % len(self.obs_list)
        for x in range(7):
            for y in range(11):
                pos = x * 11 + y
                s += cell_string(cell_status[pos])
            s += '\n'
        for i, geese in enumerate(obs['geese']):
            s += colors[i] + str(len(geese) or '-') + color_end + ' '
        return s

    def step(self, actions):
        # state transition
        obs = self.env.step([self.action2str(actions.get(p, None) or 0) for p in self.players()])
        self.update((obs, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs['status'] == 'ACTIVE':
                return False
        return True

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        action_map = {'N': Action.NORTH, 'S': Action.SOUTH, 'W': Action.WEST, 'E': Action.EAST}

        agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
        agent.last_action = action_map[self.ACTION[self.last_actions[player]][0]] if player in self.last_actions else None
        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def net(self):
        return GeeseNet

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS+1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos] -= 1
            # whole position
            for i,pos in enumerate(geese[::-1], start=4):
                b[0 + (p - player) % self.NUM_AGENTS, pos] += 2/math.log(i,2)

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for pos in obs_prev['geese'][player][:1]:
                b[0, pos] -= 2

        # food
        for pos in obs['food']:
            b[4, pos] = 1

        b = b.reshape(-1, 7, 11)

        cx, cy = divmod(obs['geese'][player][0], 11)
        b = np.roll(b, (-cx,-cy), axis=(1,2))

        # if len(obs['geese'][player]) > 3:
        #     print(obs['geese'][player])
        #     print(np.round(b[4], 2))
        #     print(np.round(b[0], 2))
        #     assert False

        return b


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = {p: e.legal_actions(p) for p in e.turns()}
            print([[e.action2str(a, p) for a in alist] for p, alist in actions.items()])
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e)
        print(e.outcome())
