from gamegym.games import atpg
from gamegym.ui.azcli import AlphaZeroCli
from gamegym.ui.tree import export_play_tree, export_az_play_tree


import keras
import tensorflow as tf

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


if __name__ == "__main__":
    cli = AlphaZeroCli(keras_custom_objects={"crossentropy_logits": crossentropy_logits},
                       max_moves=20, num_simulations=64, batch_size=128, replay_buffer_size=3000)
    cli.register_game("atpg-5", atpg.Asymetric3PlayerGomoku(5, 5))
    cli.register_model("model1", make_model)
    cli.main()