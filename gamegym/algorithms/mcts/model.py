
import numpy as np
from gamegym import Strategy
from .strategy import MctsStrategy


class Model:

    SYMMETRIC_MODEL = -1

    def __init__(self, player, adapter, trained):
        assert player != self.SYMMETRIC_MODEL or adapter.symmetrize
        self.adapter = adapter
        self.trained = trained

    def estimate(self, situation):
        raise NotImplementedError

    def fit(self, inputs, target_values, target_policy_logits, epochs):
        # Implementation must switch trained to True if fit finishes correctly
        raise NotImplementedError

    def make_input_data_from_observation(self, observation):
        """ Overload when necessary """
        return observation.data

    def make_train_input(self, situation):
        data = self.make_input_data_from_observation(self.adapter.get_observation(situation))
        assert len(data) == len(self.adapter.data_shapes)
        return data

    def make_train_policy_target(self, distribution):
        return self.adapter.encode_actions(distribution)

    def make_train_value(self, situation, value):
        # Since we simetrize the position, we have to switch values for second player
        if situation.player == 1:
            return value[::-1]
        return value

    def make_strategy(self, num_simulations):
        return MctsStrategy(self.adapter.game, self.adapter, self.estimate, num_simulations)



class KerasModel(Model):

    def __init__(self, player, adapter, trained, keras_model):
        super().__init__(player, adapter, trained)
        self.keras_model = keras_model

    def fit(self, inputs, target_values, target_policy_logits, epochs):
        self.trained = True
        self.keras_model.fit(inputs, [target_values, target_policy_logits], epochs=epochs)


    def estimate(self, situation):
        observation = self.adapter.get_observation(situation)

        # Add extra batch dimension
        data = [np.expand_dims(a, 0) for a in self.make_input_data_from_observation(observation)]

        # Do prediction
        prediction = self.keras_model.predict(data)

        # Extra value and logits from result
        value = prediction[0][0]

        # Since we simetrize the position, we have to switch values for second player
        if situation.player == 1:
            value = value[::-1]

        logits = [p for p in prediction[1:][0]]
        return value, self.adapter.decode_actions(observation, logits)
