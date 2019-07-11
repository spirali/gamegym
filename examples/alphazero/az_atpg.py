from gamegym.algorithms.mcts import search, buffer, alphazero, model as mcts_model
from gamegym.utils import Distribution
from gamegym.games import atpg
from gamegym.algorithms.stats import play_strategies
from gamegym.ui.tree import export_play_tree, export_az_play_tree
from gamegym.algorithms import tournament
from gamegym.strategy import UniformStrategy
from gamegym.ui.azcli import AlphaZeroCli

import argparse
import numpy as np
import pytest
import keras
import tensorflow as tf
import os
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crossentropy_logits(target, output):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)


def make_model(game):
    def common_part(adapter, player):
        game = adapter.game
        inputs = keras.layers.Input(adapter.shapes[player].input_shape[0])
        x = inputs
        x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh")(x)
        x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh")(x)

        y = keras.layers.Conv2D(8, (3, 3), padding="same", activation="tanh")(x)
        y = keras.layers.MaxPool2D(pool_size=(game.w, game.h), padding="same")(y)
        out_values = keras.layers.Dense(3, activation="tanh", name="out_values")(keras.layers.Flatten()(y))

        return inputs, out_values, x

    def build_player_0_model(adapter):
        player = 0
        game = adapter.game

        action_shape = adapter.shapes[player].shaped_actions[0].shape

        inputs, out_values, x = common_part(adapter, player)

        y = keras.layers.Conv2D(1, (3, 3), padding="same", activation="tanh")(x)
        out_policy = keras.layers.Reshape(action_shape)(y)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[out_values, out_policy])

        model.summary()

        def crossentropy_logits(target, output):
            return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                            logits=output)

        model.compile(
            loss=['mean_squared_error', crossentropy_logits],
            optimizer='adam')
        return model


    def build_player_1_model(adapter):
        player = 1
        game = adapter.game

        action_shape = adapter.shapes[player].shaped_actions[0].shape

        inputs, out_values, x = common_part(adapter, player)

        y = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh")(x)
        y = keras.layers.MaxPool2D(pool_size=(game.w, game.h), padding="same")(y)
        out_policy = keras.layers.Dense(8, activation="tanh")(keras.layers.Flatten()(y))
        model = keras.models.Model(
            inputs=inputs,
            outputs=[out_values, out_policy])

        model.summary()

        def crossentropy_logits(target, output):
            return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                            logits=output)

        model.compile(
            loss=['mean_squared_error', crossentropy_logits],
            optimizer='adam')
        return model


    def build_player_2_model(adapter):
        player = 2
        game = adapter.game

        action_shape = adapter.shapes[player].shaped_actions[0].shape

        inputs, out_values, x = common_part(adapter, player)

        y = keras.layers.Conv2D(4, (3, 3), padding="same", activation="tanh")(x)
        out_policy = keras.layers.Reshape(action_shape)(y)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[out_values, out_policy])

        model.summary()

        def crossentropy_logits(target, output):
            return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                            logits=output)

        model.compile(
            loss=['mean_squared_error', crossentropy_logits],
            optimizer='adam')
        return model

    adapter = game.TensorAdapter(game)
    m0 = build_player_0_model(adapter)
    m1 = build_player_1_model(adapter)
    m2 = build_player_2_model(adapter)
    return [m0, m1, m2]

"""
    def run_train():

        m0 = build_player_0_model(adapter)
        m1 = build_player_1_model(adapter)
        m2 = build_player_2_model(adapter)
        model = mcts_model.KerasModel(False, adapter, False, [m0, m1, m2])
        model.name = "conv1"

        az = alphazero.AlphaZero(
            game, model,
            max_moves=20, num_simulations=64, batch_size=64, replay_buffer_size=2800)
        az.prefill_replay_buffer()

        WRITE_STEP = 50
        for i in range(800 + 1):
            print("******************** STEP {} *************************".format(i))
            az.do_step()
            if i % WRITE_STEP == 0 and i > 0:
                az.train_model(1)
                model.keras_models[0].save("models/{}-0-{}-{}.model".format(GAME_NAME, model.name, i))
                model.keras_models[1].save("models/{}-1-{}-{}.model".format(GAME_NAME, model.name, i))
                model.keras_models[2].save("models/{}-2-{}-{}.model".format(GAME_NAME, model.name, i))


    def run_play():
        adapter = Gomoku.TensorAdapter(game, symmetrize=True)

        filename = "{}-results.dat".format(GAME_NAME)
        if os.path.isfile(filename):
            results = tournament.GameResults.load(filename)
        else:
            results = None
        pdb = tournament.PlayerList(game, game_results=results)

        for path in os.listdir("models"):
            if not path.startswith(GAME_NAME):
                continue
            name = path[len(GAME_NAME) + 1:-6]
            print("Loading", path)
            keras_model = keras.models.load_model(os.path.join("models", path), custom_objects={"crossentropy_logits": crossentropy_logits})
            model = MyModel(MyModel.SYMMETRIC_MODEL, adapter, True, keras_model)
            pdb.add_player(name, model.make_strategy(num_simulations=64))

        #pdb.add_player("uniform", UniformStrategy())

        pairing = tournament.AllPlayAllPairing(both_sides=True)
        frames = []

        for step in tqdm.tqdm(range(10)):
            pdb.play_tournament(step, pairing, skip_existing=True)

        pdb.game_results.save("results-{}.dat".format(GAME_NAME))

        #frame = pd.concat(frames)
        #sns.lineplot(x="step", y="rating", hue="player", data=frame)
        #plt.show()

        #table = pdb.get_player_table()
        #frame = pd.DataFrame(table, columns=["player", "rating", "wins", "draws", "losses"])
        #frame["net"] = frame["player"].apply(lambda x: split_name(x)[0])
        #frame["iter"] = frame["player"].apply(lambda x: split_name(x)[1])
        #sns.lineplot(x="iter", y="rating", hue="net", data=frame)
        #plt.show()

    def run_show():
        results = tournament.GameResults.load("results-{}.dat".format(GAME_NAME))
        frame = results.player_stats()
        frame["fr"] = frame["wins"] / frame["losses"]
        sns.lineplot(x="tournament_id", y="rating", hue="player", data=frame)
        plt.show()

        sns.lineplot(x="tournament_id", y="fr", hue="player", data=frame)
        plt.show()


        t = frame["tournament_id"].max()
        xframe = frame[frame["tournament_id"] == t]
        xframe.reset_index(drop=True)
        xframe["net"] = xframe["player"].apply(lambda x: split_name(x)[0])
        xframe["iter"] = xframe["player"].apply(lambda x: split_name(x)[1])
        sns.lineplot(x="iter", y="rating", hue="net", data=xframe)
        plt.show()


    def run_sample():
        models = []
        for i in range(3):
            path = "atpg-5-{}-conv1-800.model".format(i)
            keras_model = keras.models.load_model(os.path.join("models", path), custom_objects={"crossentropy_logits": crossentropy_logits})
            models.append(keras_model)

        model = mcts_model.KerasModel(False, adapter, True, models)
        s = model.make_strategy(num_simulations=64)

        sit = play_strategies(game, [s, s, s], after_move_callback=lambda sit: print(game.show_board(sit, colors=True)))



    def split_name(name):
        i = len(name) - 1
        while i >= 0 and name[i].isdigit():
            i -= 1
        i += 1
        return name[:i], int(name[i:])

if __name__ == "__main__":
    print(split_name("abc-123-321"))
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    args = ap.parse_args()

    if args.mode == "train":
        run_train()
    #elif args.mode == "play":
    #    run_play()
    #elif args.mode == "show":
    #    run_show()
    elif args.mode == "sample":
        run_sample()
    #else:
    #    print("Invalid mode")
"""

if __name__ == "__main__":
    cli = AlphaZeroCli(keras_custom_objects={"crossentropy_logits": crossentropy_logits},
                       max_moves=20, num_simulations=64, batch_size=128, replay_buffer_size=3000)
    cli.register_game("atpg-5", atpg.Asymetric3PlayerGomoku(5, 5))
    cli.register_model("model1", make_model)
    cli.main()