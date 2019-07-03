
from gamegym.games.atpg import Asymetric3PlayerGomoku
from gamegym.algorithms.mcts import search, buffer, alphazero, model
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku
from gamegym.algorithms.stats import play_strategies
from gamegym.ui.tree import export_play_tree, export_az_play_tree
from gamegym.strategy import UniformStrategy


def test_atpg_random_play():
    g = Asymetric3PlayerGomoku(5, 5, 4)
    s = UniformStrategy()
    sit = play_strategies(g, [s, s, s], after_move_callback=lambda sit: print(g.show_board(sit, colors=True)))
    print(sit.payoff)
