from typing import Callable

import numpy as np

from ..situation import Situation
from ..utils import get_rng, Distribution


class PlayInfo:

    def __init__(self, situation, distributions):
        self.situation = situation
        self.distributions = distributions

    def replay(self):
        game = self.game
        s = game.start()
        print(self.situation.history, self.distributions)
        for a, d in zip(self.situation.history, self.distributions):
            print(a)
            yield s, a, d
            s = game.play(s, a)
        yield s, None, None

    @property
    def game(self):
        return self.situation.game


def play_strategies(game,
                    strategies,
                    *,
                    rng=None,
                    seed=None,
                    start: Situation = None,
                    stop_when: Callable = None,
                    max_moves: int = None,
                    after_move_callback=None,
                    return_play_info=None):
    """
    Generate a play based on given strategies (one per player), return the last state.

    Starts from a given state or `self.start()`. Plays until a terminal state is hit,
    `stop_when(hist)` is True or for at most `max_moves` actions (whenever given).
    """
    moves = 0
    rng = get_rng(rng=rng, seed=seed)
    if len(strategies) != game.players:
        raise ValueError("One strategy per player required")
    if start is None:
        start = game.start()
    sit = start
    if return_play_info:
        distributions = []
    while not sit.is_terminal():
        if stop_when is not None and stop_when(sit):
            break
        if max_moves is not None and moves >= max_moves:
            break
        if sit.is_chance():
            dist = Distribution(sit.actions, sit.chance)
        else:
            p = sit.player
            dist = strategies[p].get_policy(sit)
        if return_play_info:
            distributions.append(dist)
        action = dist.sample(rng)
        sit = game.play(sit, action)
        moves += 1
        if after_move_callback:
            after_move_callback(sit)

    if return_play_info:
        return PlayInfo(sit, distributions)
    else:
        return sit


def sample_payoff(game, strategies, iterations=100, *, start=None, rng=None, seed=None):
    """
    Play the game `iterations` times using `strategies`.

    Returns `(mean payoffs, payoff variances)` as two numpy arrays.
    """
    rng = get_rng(rng, seed)
    if start is None:
        start = game.start()
    payoffs = [
        play_strategies(game, strategies, start=start, rng=rng).payoff for i in range(iterations)
    ]
    return (np.mean(payoffs, axis=0), np.var(payoffs, axis=0))
