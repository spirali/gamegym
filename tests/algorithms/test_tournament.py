
from gamegym.games.matrix import MatrixZeroSumGame
from gamegym.algorithms.tournament import Tournament
from gamegym import Distribution
from gamegym.strategy import ConstStrategy, UniformStrategy

import pandas as pd


simple_game = MatrixZeroSumGame([[0, -1, -1], [1, 0, -1], [1, 1, 0]])


def test_tournament():
    tr = Tournament(simple_game)


    #tr.add_player("always-0", ConstStrategy(Distribution.single_value(0)))

    tr.add_player("always-1", ConstStrategy(Distribution.single_value(1)))
    tr.add_player("uniform", UniformStrategy())
    #tr.add_player("0-or-1", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.5, 0.5, 0))))
    #tr.add_player("0-or-2", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.5, 0, 0.5))))
    #tr.add_player("0.1/0.6/0.3", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.1, 0.6, 0.3))))
    tr.add_player("0.2/0.1/0.7", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.1, 0.2, 0.7))))
    tr.add_player("0.00/0.6/0.4", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.0, 0.6, 0.4))))

    frames = []
    for step in range(400):
        #tr.play_random_matches(16)
        tr.play_all_pairs()
        table = tr.get_table()
        frame = pd.DataFrame(table, columns=["player", "rating", "wins", "losses", "draws"])
        frame["step"] = step
        frames.append(frame)

    frame = pd.concat(frames)

    import seaborn as sns
    import matplotlib.pyplot as plt

    frame["f"] = frame["wins"] / frame["losses"]

    #sns.lineplot(x="step", y="f", hue="player", data=frame)
    sns.lineplot(x="step", y="rating", hue="player", data=frame)
    #sns.lineplot(x="step", y="losses", hue="player", data=frame)
    plt.show()




