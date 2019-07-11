
from .stats import play_strategies
from ..utils import get_rng, Distribution
from enum import Enum
import math
import itertools
import random
import json
import tqdm
import numpy as np
from collections import namedtuple, Counter


"""
class RandomPairing:

    def __init__(self, rounds, rng=None, seed=None):
        self.rng = rng = get_rng(rng=rng, seed=seed)
        self.rounds = rounds

    def generate_pairing(self, players):
        for _ in range(self.rounds):
            yield self.rng.choice(players, size=2)



class AllPlayAllPairing:

    def __init__(self, both_sides=False, randomize=False, rng=None, seed=None):
        self.both_sides = both_sides
        if randomize:
            self.rng = get_rng(rng, seed)
        else:
            self.rng = None

    def generate_pairing(self, players):
        def shuffle(pair):
            pair = list(pair)
            rng.shuffle(pair)
            return pair

        if self.both_sides:
            return itertools.permutations(players, 2)
        else:
            rng = self.rng
            it = itertools.combinations(players, 2)
            if rng is None:
                return it
            return (shuffle(pair) for pair in it)
"""


class AllPlayAllPairing:

    def __init__(self):
        pass

    def generate_pairing(self, players):
        return itertools.product(*players)


ResultRecord = namedtuple("Record", ["tournament_id", "players", "payoff"])

class GameResults:

    def __init__(self):
        self.records = []

    def to_dicts(self):
        return [{"tournament_id": r.tournament_id,
                 "players": r.players,
                 "payoff": tuple(r.payoff)}
            for r in self.records
        ]

    def save(self, filename):
        data = self.to_dicts()
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        results = GameResults()
        results.records = [
            ResultRecord(d["tournament_id"], tuple(d["players"]), np.array(d["payoff"]))
            for d in data
        ]
        return results

    def players(self, position=None):
        if position is None:
            return set(p for r in self.records for p in r.players)
        else:
            return set(r.players[position] for r in self.records)

    def add_result(self, tournament_id, players, payoff):
        self.records.append(ResultRecord(tournament_id, players, payoff))

    def tournament_pairings(self, tournament_id):
        return [r.players for r in self.records if r.tournament_id == tournament_id]

    """
    def _compute_elo(self, player1_rating, player2_rating, elo_k, result):
        if result == 0:
            result_p1 = 0.5
            result_p2 = 0.5
        elif result > 0:
            result_p1 = 1
            result_p2 = 0
        else:
            result_p1 = 0
            result_p2 = 1

        r1 = 10 ** (player1_rating / 400)
        r2 = 10 ** (player2_rating / 400)
        rs = r1 + r2
        e1 = r1 / rs
        e2 = r2 / rs
        return elo_k * (result_p1 - e1), elo_k * (result_p2 - e2)
    """

    def player_stats(self):
        import pandas as pd
        players = sorted(self.players())

        frame = pd.DataFrame(index=players)
        frame["payoff"] = 0
        frame["plays"] = 0

        for r in self.records:
            for player, payoff in zip(r.players, r.payoff):
                p = frame.loc[player]
                p.payoff += payoff
                p.plays += 1

        frame["name"] = frame.index
        return frame


        """
        def get_tournamen_id(record):
            return record.tournament_id

        self.records.sort(key=get_tournamen_id)
        frames = []
        for tournament_id, group in itertools.groupby(self.records, key=get_tournamen_id):
            elo_change = {}
            for r in group:
                p1 = frame.loc[r.player1]
                p2 = frame.loc[r.player2]
                elo_change1, elo_change2 = self._compute_elo(p1["rating"], p2["rating"], elo_k, r.result)

                elo_change.setdefault(r.player1, 0)
                elo_change.setdefault(r.player2, 0)
                elo_change[r.player1] += elo_change1
                elo_change[r.player2] += elo_change2

                if r.result == 0:
                    frame.loc[r.player1, "draws"] += 1
                    frame.loc[r.player2, "draws"] += 1
                elif r.result > 0:
                    frame.loc[r.player1, "wins"] += 1
                    frame.loc[r.player2, "losses"] += 1
                    #p1.wins += 1
                    #p2.losses += 1
                else:
                    frame.loc[r.player1, "losses"] += 1
                    frame.loc[r.player2, "wins"] += 1
                    #p2.wins += 1
                    #p1.losses += 1

            for p, v in elo_change.items():
                frame.loc[p, "rating"] = max(100, frame.loc[p, "rating"] + v)

            #print(frame)

            new_frame = frame.copy()
            new_frame["tournament_id"] = tournament_id
            new_frame["player"] = new_frame.index
            new_frame.reset_index(inplace=True, drop=True)
            frames.append(new_frame)
        return pd.concat(frames)
        """

Player = namedtuple("Player", ["name", "strategy", "player_position"])


class PlayerList:

    def __init__(self,
                 game,
                 max_moves=None,
                 game_results=None,
                 rng=None,
                 seed=None):

        self.game = game
        self.max_moves = max_moves
        self.rng = get_rng(rng=rng, seed=seed)
        self.players = [[] for _ in range(game.players)]
        self.game_results = game_results or GameResults()


    def add_player(self, name, strategy, position):
        assert 0 <= position < self.game.players
        self.players[position].append(Player(name, strategy, position))


    def play_tournament(self, tournament_id, pairing_generator, skip_existing=False):
        assert isinstance(tournament_id, int)
        game_results = self.game_results

        if any(not ps for ps in self.players):
            raise Exception("Not enough players")

        pairing = pairing_generator.generate_pairing(self.players)

        if skip_existing:
            raise Exception("TODO")
            """
            p = game_results.tournament_pairings(tournament_id)
            existing = Counter(p)
            pairing_counter = Counter(pairing)
            pairing = (pairing_counter - existing).elements()
            """

        for ps in tqdm.tqdm(list(pairing)):
            result = self._play_match(ps)
            game_results.add_result(tournament_id, [p.name for p in ps], result)


    def _play_match(self, players):
        return play_strategies(self.game, [p.strategy for p in players], rng=self.rng).payoff

