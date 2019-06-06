

from gamegym.ui.tree import TreeNode, export_tree_to_html, export_play_tree
from gamegym.algorithms.stats import play_strategies
from gamegym.strategy import UniformStrategy
from gamegym.games import Gomoku
from gamegym.games import Goofspiel


def test_tree(tmpdir):
    output = str(tmpdir.join("output.html"))
    print(output)

    #root = TreeNode("Root")
    #c1 = root.child("Child 1")
    #c2 = root.child("Child 2")
    #c1.child("Child 1.1")
    #c1.child("Child 1.2")
    #c1.child("Child 1.3")
    #c2.child("Child 2.1")

    #export_tree_to_html(root, output)

    g = Gomoku(3, 4, 3)
    #g = Goofspiel(4, Goofspiel.Scoring.ZEROSUM)
    s = UniformStrategy()
    play_info = play_strategies(g, [s, s], return_play_info=True)
    export_play_tree(play_info, output)