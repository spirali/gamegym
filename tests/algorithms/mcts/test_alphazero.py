import numpy as np
import pytest
import keras
import tensorflow as tf

from gamegym.algorithms.mcts import search, buffer, alphazero
from gamegym.utils import Distribution
from gamegym.games import Gomoku, gomoku
from gamegym.algorithms.stats import play_strategies

def build_model(adapter):
    assert len(adapter.data_shapes) == 1
    action_shapes = adapter.action_data_shapes
    assert len(action_shapes) == 1
    action_shape = action_shapes[0]
    inputs = keras.layers.Input(adapter.data_shapes[0])
    x = keras.layers.Flatten()(inputs)
    #x = keras.layers.Dense(32, activation=keras.layers.LeakyReLU)(x)
    x = keras.layers.Dense(12, activation="tanh")(x)
    x = keras.layers.Dense(12, activation="tanh")(x)

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

    return model


def test_alphazero():
    g = Gomoku(4, 4, 3)
    adapter = Gomoku.TensorAdapter(g, symmetrize=True)
    model = build_model(adapter)

    az = alphazero.AlphaZero(
        g, adapter, model,
        max_moves=20, num_simulations=64, batch_size=64, replay_buffer_size=1280)
    az.prefill_replay_buffer()

    assert 32 <= az.replay_buffer.records_count <= 128

    az.train_network(4, 1)

    estimator = az.last_estimator()
    v, dd = estimator(g.start())
    dd.pprint()
    print(v)
    #return

    #print(g.show_board(sit, colors=True))

    for i in range(300):
        az.play_game()
        az.train_network(8, 1)

    s = az.make_strategy()
    sit = play_strategies(g, [s, s], after_move_callback=lambda sit: print(g.show_board(sit, colors=True)))


    estimator = az.last_estimator()
    v, dd = estimator(g.start())
    dd.pprint()
    print(v)