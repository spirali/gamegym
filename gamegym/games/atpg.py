from typing import Any, Tuple

import numpy as np

from ..game import PerfectInformationGame
from ..situation import Action, Situation, StateInfo
from ..adapter import Adapter, TensorAdapter, TextAdapter, TensorShape
from ..utils import Distribution
from ..ui.cliutils import draw_board


class Asymetric3PlayerGomoku(PerfectInformationGame):
    """
    A artificial asymetric variant of Gomoku

    The state is encoded as `(np.array board, tuple of available actions)`.
    """

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        board_actions = tuple((r, c) for c in range(self.w) for r in range(self.h))
        self.player0_actions = board_actions
        dirs1 = ("n", "s", "w", "e")
        dirs2 = ("nw", "ne", "sw", "se")
        dirs = dirs1 + dirs2
        self.dir_delta = {
            "n": (0, -1),
            "s": (0, 1),
            "e": (-1, 0),
            "w": (1, 0),
            "nw": (-1, -1),
            "ne": (1, -1),
            "se": (1, 1),
            "sw": (-1, 1)
        }
        self.player1_actions = dirs
        player2_actions =  tuple((p, d) for p in board_actions for d in dirs1)
        self.player2_actions = player2_actions
        actions = board_actions + dirs + player2_actions

        self.max_moves = 20

        super().__init__(3, actions)

    def initial_state(self) -> StateInfo:
        """
        Return the initial internal state and active player.
        """
        board = np.zeros((self.h, self.w), dtype=np.int8) - 1  # -1: empty, 0,1: player 0/1
        state = (board, self.player0_actions)  # board, free coordinates, places of first two players
        return StateInfo.new_player(state, 0, self.player0_actions)

    def _check_empty(self, board, x, y):
        return x >= 0 and x < self.w and y >= 0 and y < self.h and board[x, y] == -1

    def update_state(self, situation: Situation, action: Action) -> StateInfo:
        """
        Return the updated internal state, active player and per-player observations.
        """
        player = situation.player
        board, free_fields = situation.state

        new_board = board.copy()
        if player == 0:
            assert action in free_fields
            assert new_board[action] == -1
            new_pos = action
            new_pos_value = 0
        elif player == 1:
            x, y = situation.history[-1]
            dx, dy = self.dir_delta[action]
            new_pos = (x + dx, y + dy)
            assert new_board[new_pos] == -1
            new_pos_value = 1
        else:
            assert player == 2
            pos, d = action
            dx, dy = self.dir_delta[d]
            v = new_board[pos]
            assert v == 0 or v == 1
            new_board[pos] = 2
            new_pos = (pos[0] + dx, pos[1] + dy)
            new_pos_value = v

        new_board[new_pos] = new_pos_value
        new_free_fields = tuple(a for a in free_fields if a != new_pos)
        new_state = (new_board, new_free_fields)

        if len(situation.history) == self.max_moves:
            return StateInfo.new_terminal(new_state, (-1, -1, 1))

        x, y = new_pos
        v = new_board[new_pos]
        chain = 4 - v
        if ((self._extent(new_board, x, y, -1, -1, new_pos_value) >= chain) or
            (self._extent(new_board, x, y, 1, -1, new_pos_value) >= chain) or
            (self._extent(new_board, x, y, 0, 1, new_pos_value) >= chain) or
            (self._extent(new_board, x, y, 1, 0, new_pos_value) >= chain)):

            payoff3 = len(situation.history) / self.max_moves
            if v == 0:
                payoff = (1, -1, payoff3)
            else:
                payoff = (-1, 1, payoff3)
            return StateInfo.new_terminal(new_state, payoff)

        new_player = (player + 1) % 3

        if new_player == 1:
            new_actions = []
            for k, (dx, dy) in self.dir_delta.items():
                if self._check_empty(new_board, x + dx, y + dy):
                    new_actions.append(k)
            new_actions = tuple(new_actions)
            if not new_actions:
                new_player = 2

        if new_player == 2:
            new_actions = []
            for a in self.player2_actions:
                pos, d = a
                v = new_board[pos]
                if v == -1 or v == 2:
                    continue
                x, y = pos
                dx, dy = self.dir_delta[d]
                if self._check_empty(new_board, x + dx, y + dy):
                    new_actions.append(a)
            new_actions = tuple(new_actions)
            if not new_actions:
                new_player = 0

        if new_player == 0:
            new_actions = new_free_fields
        return StateInfo.new_player(new_state, new_player, new_actions)

    # def get_features(self, situation: Situation, _for_player: int = None) -> tuple:
    #     """
    #     Return the features as a tuple of numpy arrays.

    #     The features are: `(active player pieces, other player pieces, active player no)`.
    #     """
    #     board = situation.state[0]
    #     player = situation.player
    #     active_board = (board == player).astype(np.float32)
    #     other_board = (board == 1 - player).astype(np.float32)
    #     return (active_board, other_board, np.array([player], np.float32))

    # def get_features_shape(self) -> tuple:
    #     """
    #     Return the shapes of the features as tuple of tensor shapes (tuples).
    #     """
    #     return ((self.h, self.w), (self.h, self.w), (1, ))

    def _extent(self, b: np.ndarray, r: int, c: int, dr: int, dc: int, v: int) -> int:
        """
        Return the length of a chain of the values at `b[r, c]` in the direction `(dr, dc)`.
        Does not include `b[r, c]` itself.
        """
        l = 1
        rr = r + dr
        cc = c + dc
        while rr >= 0 and cc >= 0 and rr < self.h and cc < self.w and b[rr, cc] == v:
            l += 1
            rr += dr
            cc += dc

        rr = r - dr
        cc = c - dc
        while rr >= 0 and cc >= 0 and rr < self.h and cc < self.w and b[rr, cc] == v:
            l += 1
            rr -= dr
            cc -= dc
        return l

    def __repr__(self) -> str:
        return "<{} {}x{}>".format(
            self.__class__.__name__, self.w, self.h)

    def show_board(self, situation, swap_players=False, colors=False) -> str:
        """
        Return a string with a pretty-printed board
        """
        if swap_players:
            scolors = ["yellow", "red", "blue", "green"]
            symbols =  '.ox#'
        else:
            scolors = ["yellow", "blue", "red", "green"]
            symbols = '.xo#'

        if not colors:
            scolors = None

        return "Player: {}\n{}".format(situation.player, draw_board(situation.state[0] + 1, symbols, scolors))

    # def show_situation(self, situation, swap_players=False) -> str:
    #     """
    #     Return a string with a pretty-printed board and one-line game information.
    #     """
    #     ps = ["player 0 (x)", "player 1 (o)"]
    #     cs = {-1: '.', 0: 'x', 1: 'o'}
    #     if swap_players:
    #         ps = ps[1], ps[0]
    #         cs = {-1: '.', 0: 'o', 1: 'x'}

    #     if situation.is_terminal():
    #         if situation.payoff[0] > 0.0:
    #             info = ps[0] + " won"
    #         elif situation.payoff[0] < 0.0:
    #             info = ps[1] + " won"
    #         else:
    #             info = "draw"
    #     else:
    #         info = ps[situation.player] + " active"

    #     lines = [''.join(cs[x] for x in l) for l in situation.state[0]]
    #     return "\n".join(lines) + "\n{} turn {}, {}".format(self, len(situation.history) + 1, info)


    #class TextAdapter(TextAdapter):
    #    SYMMETRIZABLE = True
    #    IGNORE_WHITESPACE = False
    #    IGNORE_MULTI_WHITESPACE = True
    #    IGNORE_PARENS = True

    # def observe_data(self, sit, _player):
    #     swap = self.symmetrize and sit.player == 1
    #     return sit.game.show_board(sit, colors=self.colors)


    class TensorAdapter(TensorAdapter):

        SYMMETRIZABLE = False

        def observe_data(self, situation, player):
            """
            Extract features from a given game situation from the point of view of the active player.
            """
            board = situation.state[0]
            if player == 1:
                last_move = np.zeros_like(board)
                last_move[situation.history[-1]] = 1
                return (np.stack([board != 0, board != 1, board != 2, last_move], axis=2),)
            else:
                return (np.stack([board != 0, board != 1, board != 2], axis=2),)

        def _generate_shapes(self):
            board = [(self.game.w, self.game.h, 3)]
            board_and_last_move = [(self.game.w, self.game.h, 4)]

            actions = self.game.player0_actions
            array = np.empty(len(actions), dtype=object)
            for i in range(len(actions)):
                array[i] = actions[i]
            player0 = np.reshape(array, (self.game.w, self.game.h))

            actions = self.game.player1_actions
            player1 = np.array(actions, dtype=object)

            actions = self.game.player2_actions
            array = np.empty(len(actions), dtype=object)
            for i in range(len(actions)):
                array[i] = actions[i]
            player2 = np.reshape(array, (self.game.w, self.game.h, 4))

            return [
                TensorShape(board, player0),
                TensorShape(board_and_last_move, player1),
                TensorShape(board, player2),
            ]

        def shape_index(self, situation: Situation):
            return situation.player