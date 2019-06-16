
from .stats import play_strategies
from ..utils import get_rng, Distribution
from enum import Enum
import math
import itertools


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



class Tournament:

    def __init__(self, game, player_order: PlayerOrder = PlayerOrder.Fixed, max_moves=None, rng=None, seed=None):
        assert game.players == 2
        self.players = {}
        self.game = game
        self.max_moves = max_moves
        self.player_order = player_order
        self.rng = get_rng(rng=rng, seed=seed)


    def add_player(self, name, strategy, rating=1500):
        assert name not in self.players
        self.players[name] = Player(name, strategy, rating)

    def get_table(self):
        return [(player.name, player.rating, player.wins, player.losses, player.draws)
                for player in self.players.values()]

    def _compute_elo(self, player1, player2, elo_k, result):
        r1 = 10 ** (player1.rating / 400)
        r2 = 10 ** (player2.rating / 400)
        rs = r1 + r2
        e1 = r1 / rs
        e2 = r2 / rs
        player1.rating_change += elo_k * (result - e1)
        player2.rating_change += elo_k * (1 - result - e2)

    def update_ratings(self):
        for player in self.players.values():
            player.rating = max(100, player.rating + player.rating_change)
            player.rating_change = 0

    def play_random_matches(self, count, elo_k=32):
        players = list(self.players.values())
        assert len(players) >= 2

        for _ in range(count):
            player1, player2 = self.rng.choice(players, size=2)
            self._play_match(player1, player2, elo_k)
        self.update_ratings()

    def play_all_pairs(self, elo_k=32):
        players = list(self.players.values())
        for player1, player2 in itertools.combinations(players, 2):
            self._play_match(player1, player2, elo_k)
        self.update_ratings()

    def _play_match(self, player1, player2, elo_k):
        #if self.player_order == PlayerOrder.Fixed:
        #    value = play_strategies(self.game, [player1.strategy, player2.strategy]).value
        #else:
        #    raise Exception("Invalid player order")

        value = play_strategies(self.game, [player1.strategy, player2.strategy], rng=self.rng).payoff
        if abs(value[0] - value[1]) < 0.00001:
            player1.draws += 1
            player2.draws += 1
            result = 0
        elif value[0] > value[1]:
            player1.wins += 1
            player2.losses += 1
            result = 1
        else:
            player2.wins += 1
            player1.losses += 1
            result = -1
        self._compute_elo(player1, player2, elo_k, result)