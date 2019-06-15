
from .stats import play_strategies
from ..utils import get_rng, Distribution
from enum import Enum
import math
import itertools
from collections import namedtuple


class Player:

    def __init__(self, name, strategy, rating=None):
        self.name = name
        self.strategy = strategy
        self.rating = rating
        self.rating_change = 0

        self.wins = 0
        self.losses = 0
        self.draws = 0


class PlayerOrder:

    Fixed = 0
    Random = 1
    BothSides = 2


ResultRecord = namedtuple("Record", ["player1", "player2", "result"])


class RandomPairing:

    def __init__(rounds, rng=None, seed=None):
        self.rng = rng = get_rng(rng=rng, seed=seed)
        self.rounds = rounds

    def generate_pairing(self, players):
        for _ in range(rounds):
            yield self.rng.choice(players, size=2)


class AllPlayAllPairing:

    def generate_pairing(self, players):
        return itertools.combinations(players, 2)


class PlayerDatabase:

    def __init__(self,
                 game,
                 player_order: PlayerOrder = PlayerOrder.Fixed,
                 max_moves=None,
                 rng=None, seed=None):
        assert game.players == 2
        self.players = {}
        self.results = []
        self.game = game
        self.max_moves = max_moves
        self.player_order = player_order
        self.rng = get_rng(rng=rng, seed=seed)

    def add_player(self, name, strategy, rating=1500):
        assert name not in self.players
        self.players[name] = Player(name, strategy, rating)

    def _compute_elo(self, player1, player2, elo_k, result_p1, result_p2):
        r1 = 10 ** (player1.rating / 400)
        r2 = 10 ** (player2.rating / 400)
        rs = r1 + r2
        e1 = r1 / rs
        e2 = r2 / rs
        player1.rating_change += elo_k * (result_p1 - e1)
        player2.rating_change += elo_k * (result_p2 - e2)

    def play_tournament(self, pairing_generator, elo_k=1):
        players = list(self.players.values())
        if len(players) < 2:
            raise Exception("Not enough players")

        for player1, player2 in pairing_generator.generate_pairing(players):
            self._play_match(player1, player2, elo_k)

        for player in players:
            player.rating = max(100, player.rating + player.rating_change)
            player.rating_change = 0

    def get_player_table(self):
        return [(p.name, p.rating, p.wins, p.draws, p.losses)
                for p in self.players.values()]

    def _play_match(self, player1, player2, elo_k):
        value = play_strategies(self.game, [player1.strategy, player2.strategy], rng=self.rng).payoff
        if abs(value[0] - value[1]) < 0.00001:
            player1.draws += 1
            player2.draws += 1
            result_p1 = 0.5
            result_p2 = 0.5
            result = 0
        elif value[0] > value[1]:
            player1.wins += 1
            player2.losses += 1
            result_p1 = 1
            result_p2 = 0
            result = 1
        else:
            player2.wins += 1
            player1.losses += 1
            result_p1 = 0
            result_p2 = 1
            result = -1
        self.results.append(ResultRecord(player1, player2, result))
        self._compute_elo(player1, player2, elo_k, result_p1, result_p2)