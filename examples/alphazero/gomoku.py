from gamegym.algorithms.mcts import search, buffer, alphazero, model
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku
from gamegym.algorithms.stats import play_strategies
from gamegym.ui.tree import export_play_tree, export_az_play_tree
from gamegym.algorithms import tournament
from gamegym.strategy import UniformStrategy

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


class MyModel(model.KerasModel):
    pass


def crossentropy_logits(target, output):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)


def build_dense_model(adapter):
    assert len(adapter.data_shapes) == 1
    action_shapes = adapter.action_data_shapes
    assert len(action_shapes) == 1
    action_shape = action_shapes[0]
    inputs = keras.layers.Input(adapter.data_shapes[0])
    x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    x = keras.layers.Dense(16, activation="tanh")(x)
    x = keras.layers.Dense(16, activation="tanh")(x)

    out_values = keras.layers.Dense(2, activation="tanh")(x)

    y = keras.layers.Dense(np.prod(action_shape), activation="softmax")(x)
    out_policy = keras.layers.Reshape(action_shape)(y)

    model = keras.models.Model(
        inputs=inputs,
        outputs=[out_values, out_policy])

    model.compile(
        loss=['mean_squared_error', crossentropy_logits],
        optimizer='adam')

    m = MyModel(MyModel.SYMMETRIC_MODEL, adapter, False, model)
    m.name = "dense"
    return m

def build_conv2_model(adapter):
    game = adapter.game
    assert len(adapter.data_shapes) == 1
    action_shapes = adapter.action_data_shapes
    assert len(action_shapes) == 1
    action_shape = action_shapes[0]
    inputs = keras.layers.Input(adapter.data_shapes[0])
    x = inputs
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    #x = keras.layers.Conv2D(16, (3, 3), padding="same")
    #x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    #x = keras.layers.Dense(16, activation="tanh")(x)
    #x = keras.layers.Dense(16, activation="tanh")(x)

    y = keras.layers.Conv2D(8, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    y = keras.layers.MaxPool2D(pool_size=(game.w, game.h), padding="same", data_format="channels_first")(y)
    out_values = keras.layers.Dense(2, activation="tanh", name="out_values")(keras.layers.Flatten()(y))

    #y = keras.layers.Conv2D(2, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    #out_values = keras.layers.Reshape((2,))(keras.layers.MaxPool2D((1, 1), name="out_values")(y))


    y = keras.layers.Conv2D(1, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
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
    m = MyModel(MyModel.SYMMETRIC_MODEL, adapter, False, model)
    m.name = "conv2"
    return m


def build_conv_model(adapter):
    assert len(adapter.data_shapes) == 1
    action_shapes = adapter.action_data_shapes
    assert len(action_shapes) == 1
    action_shape = action_shapes[0]
    inputs = keras.layers.Input(adapter.data_shapes[0])
    x = inputs
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    #x = keras.layers.Conv2D(16, (3, 3), padding="same")
    #x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    #x = keras.layers.Dense(16, activation="tanh")(x)
    #x = keras.layers.Dense(16, activation="tanh")(x)

    out_values = keras.layers.Dense(2, activation="tanh", name="out_values")(keras.layers.Flatten()(x))
    #y = keras.layers.Conv2D(2, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
    #out_values = keras.layers.Reshape((2,))(keras.layers.MaxPool2D((1, 1), name="out_values")(y))


    y = keras.layers.Conv2D(1, (3, 3), padding="same", activation="tanh", data_format="channels_first")(x)
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
    m = MyModel(MyModel.SYMMETRIC_MODEL, adapter, False, model)
    m.name = "conv"
    return m



SIZE = 5
CHAIN_SIZE = 4
GAME_NAME = "gomoku-{}-{}".format(SIZE, CHAIN_SIZE)

def run_train():
    game = Gomoku(SIZE, SIZE, CHAIN_SIZE)
    adapter = Gomoku.TensorAdapter(game, symmetrize=True)

    model = build_conv2_model(adapter)

    az = alphazero.AlphaZero(
        game, model,
        max_moves=20, num_simulations=64, batch_size=64, replay_buffer_size=2800)
    az.prefill_replay_buffer()

    WRITE_STEP = 300
    for i in range(1501):
        az.do_step()
        if i % WRITE_STEP == 0 and i > 0:
            az.train_model(1)
            model.keras_model.save("models/{}-{}-{}.model".format(GAME_NAME, model.name, i))


def run_play():
    game = Gomoku(SIZE, SIZE, CHAIN_SIZE)
    adapter = Gomoku.TensorAdapter(game, symmetrize=True)

    filename = "{}-results.dat".format(GAME_NAME)
    if os.path.isfile(filename):
        results = tournament.GameResults.load(filename)
    else:
        results = None
    pdb = tournament.PlayerList(game, game_results=results)

    for path in os.listdir("models"):
        if not path.startswith(GAME_NAME):
            break
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

    #pdb.game_results.save("results2.dat")

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

    t = frame["tournament_id"].max()
    xframe = frame[frame["tournament_id"] == t]
    xframe.reset_index(drop=True)
    xframe["net"] = xframe["player"].apply(lambda x: split_name(x)[0])
    xframe["iter"] = xframe["player"].apply(lambda x: split_name(x)[1])
    sns.lineplot(x="iter", y="rating", hue="net", data=xframe)
    plt.show()


def run_sample():
    game = Gomoku(SIZE, SIZE, CHAIN_SIZE)
    adapter = Gomoku.TensorAdapter(game, symmetrize=True)

    path = "gomoku-5-4-conv2-500.model"
    keras_model = keras.models.load_model(os.path.join("models", path), custom_objects={"crossentropy_logits": crossentropy_logits})
    model = MyModel(MyModel.SYMMETRIC_MODEL, adapter, True, keras_model)
    s = model.make_strategy(num_simulations=64)

    sit = play_strategies(game, [s, s], after_move_callback=lambda sit: print(game.show_board(sit, colors=True)))



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
    elif args.mode == "play":
        run_play()
    elif args.mode == "show":
        run_show()
    elif args.mode == "sample":
        run_sample()
    else:
        print("Invalid mode")