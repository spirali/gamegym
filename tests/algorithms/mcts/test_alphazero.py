import numpy as np
import pytest
import keras
import tensorflow as tf

from gamegym.algorithms.mcts import search, buffer, alphazero, model
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku
from gamegym.algorithms.stats import play_strategies
from gamegym.ui.tree import export_play_tree, export_az_play_tree


class MyModel(model.KerasModel):
    pass


def build_conv_model(adapter):
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

    def crossentropy_logits(target, output):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                        logits=output)

    model.compile(
        loss=['mean_squared_error', crossentropy_logits],
        optimizer='adam')

    super().__init__(self.SYMMETRIC_MODEL, adapter, False, model)


def build_dense_model(adapter):
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

    return model



def test_alphazero(tmpdir):
    g = Gomoku(4, 4, 3)
    adapter = Gomoku.TensorAdapter(g, symmetrize=True)
    model = MyModel(MyModel.SYMMETRIC_MODEL, adapter, False, build_conv_model(adapter))

    az = alphazero.AlphaZero(
        g, model,
        max_moves=20, num_simulations=64, batch_size=64, replay_buffer_size=2800)

    az.prefill_replay_buffer()

    #assert 32 <= az.replay_buffer.records_count <= 128

    az.train_model(4, 1)

    estimator = az.last_estimator()
    v, dd = estimator(g.start())
    dd.pprint()
    print(v)
    #return

    #print(g.show_board(sit, colors=True))

    for _ in range(300):
        az.do_step()
    az.train_model(4)

    print("STATS:", az.replay_buffer.added, az.replay_buffer.sampled)

    s = az.make_strategy()
    sit = play_strategies(g, [s, s], after_move_callback=lambda sit: print(g.show_board(sit, colors=True)))


    estimator = az.last_estimator()
    v, dd = estimator(g.start())
    dd.pprint()
    print(v)

    output = str(tmpdir.join("output.html"))
    print("OUTPUT", output)
    #play_info = play_strategies(g, [s, s], return_play_info=True)
    #export_play_tree(play_info, output)
    export_az_play_tree(az, output, num_simulations=256)