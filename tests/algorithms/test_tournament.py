
from gamegym.games.matrix import MatrixZeroSumGame
from gamegym.algorithms.tournament import PlayerDatabase, AllPlayAllPairing
from gamegym import Distribution
from gamegym.strategy import ConstStrategy, UniformStrategy

import pandas as pd


simple_game = MatrixZeroSumGame([[0, -1, -1], [1, 0, -1], [1, 1, 0]])


def test_tournament():
    pl = PlayerDatabase(simple_game)

    #pl.add_player("always-0", ConstStrategy(Distribution.single_value(0)))

    pl.add_player("always-1", ConstStrategy(Distribution.single_value(1)))
    pl.add_player("uniform", UniformStrategy())
    #pl.add_player("0-or-1", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.5, 0.5, 0))))
    #pl.add_player("0-or-2", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.5, 0, 0.5))))
    #pl.add_player("0.1/0.6/0.3", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.1, 0.6, 0.3))))
    pl.add_player("0.2/0.1/0.7", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.1, 0.2, 0.7))))
    pl.add_player("0.00/0.6/0.4", ConstStrategy(Distribution(vals=[0, 1, 2], probs=(0.0, 0.6, 0.4))))

    frames = []

    pairing = AllPlayAllPairing()

    for step in range(1, 50):
        pl.play_tournament(pairing, 1)
        table = pl.get_player_table()
        frame = pd.DataFrame(table, columns=["player", "rating", "wins", "draws", "losses"])
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




