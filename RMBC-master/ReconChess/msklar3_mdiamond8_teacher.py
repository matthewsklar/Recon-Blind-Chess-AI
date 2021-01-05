import os.path as path

import chess
import numpy as np
import torch
import torch.optim as optim

import msklar3_mdiamond8 as agent
import msklar3_mdiamond8_config as config
import msklar3_mdiamond8_knight_rush_agent as knight_rush_agents
import msklar3_mdiamond8_nn as nn
import random_agent
from msklar3_mdiamond8_memory import GameMemory, TurnMemory
from msklar3_mdiamond8_play_game import play_local_game
import random

WINNERS = {
    'WHITE' : 0,
    'BLACK' : 0,
}

class Teacher():
    def __init__(self, agent, opponent, games_per_epoch):
        self.game_history = []  # list of game history objects
        self.agent = agent
        if random.random() < 0.5:
            self.opponent = knight_rush_agents.KnightRush()
        else:
            self.opponent = opponent
        self.games_per_epoch = games_per_epoch
        self.win  = 0
        self.loss = 0

        dirname = path.dirname(__file__)
        self.network = torch.load(path.join(dirname, 'msklar3_mdiamond8_network.torch'))    
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

    def epoch(self):
        print('loss ---------------------')
        for _ in range(self.games_per_epoch):
            self.play_game()

        self.train()

        self.game_history = []
        torch.save(self.network, 'network.torch')

        print('loss epoch stats')
        print('loss { wins: ', self.win, ' loss: ', self.loss, ', win_percent: ', self.win / (self.win + self.loss))

        self.win = 0
        self.loss = 0

        self.agent = agent.MagnusDLuffy()

    '''
    Play a game and obtain game history data
    '''
    def play_game(self):
        results = play_local_game(self.agent, self.opponent, ['white','black'])
        print(results[0])
        print("All game stats", WINNERS)
        if results[0] == True:
            WINNERS['WHITE'] += 1

            if self.agent.color == chess.WHITE:
                self.win += 1
            else:
                self.loss += 1
        elif results[0] == False:
            WINNERS['BLACK'] += 1

            if self.agent.color == chess.WHITE:
                self.loss += 1
            else:
                self.win += 1

        if self.agent.color == chess.WHITE:
            self.add_game(results[2])
        elif self.agent.color == chess.BLACK:
            self.add_game(results[3])

    def add_game(self, game):
        if isinstance(game, GameMemory):
            self.game_history.append(game)

    def turn_history(self):
        turns = []

        for game in self.game_history:
            for t in game.turns:
                if game.v != None:
                    turns.append((t, game.v))

        return turns

    '''
    Train on subset of terms in game history memory
    '''
    def train(self):
        turn_list = self.turn_history()
        training_turn_count = min(config.TRAINING_TURNS_PER_EPOCH, len(turn_list))
        training_turns = np.random.randint(0, len(turn_list), size=(training_turn_count,))

        pi_list = []
        v_list = []

        for turn in training_turns:
            turn_data = turn_list[turn]

            pi, v = self.network.forward(turn_data[0].state[0], turn_data[0].state[1])

            pi_list.append(pi)
            v_list.append(v)

        self.optimizer.zero_grad()

        loss = self.network.loss(v, turn_data[1], pi, turn_data[0].p)
        print('loss:', loss)
        loss.backward(retain_graph=True)

        self.optimizer.step()

if __name__ == "__main__":
    teacher = Teacher(agent.MagnusDLuffy(), random_agent.Random(), config.GAMES_PER_EPOCH)

    for _ in range(config.EPOCHS):
        teacher.epoch()

    print(WINNERS)
