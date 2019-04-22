
import numpy as np
from .search import MctSearch
from .buffer import ReplayBuffer, ReplayRecord

from gamegym import Strategy
from gamegym.utils import Distribution, flatten_array_list

def dummy_estimator(situation):
    return np.array((0, 0)), Distribution(situation.state[1], None)


class AlphaZeroStrategy(Strategy):

    def __init__(self, game, adapter, estimator, num_simulations):
        super().__init__(game, adapter)
        self.estimator = estimator
        self.num_simulations = num_simulations

    def get_policy(self, situation):
        s = MctSearch(situation, self.estimator)
        s.search(self.num_simulations)
        action = s.best_action_max_visits()
        return Distribution.single_value(action)


class AlphaZero:

    def __init__(self,
                 game,
                 adapter,
                 model,
                 batch_size=128,
                 replay_buffer_size=2024,
                 max_moves=1000,
                 num_simulations=800,
                 num_sampling_moves=30):
        assert batch_size <= replay_buffer_size
        assert adapter.symmetrize
        self.game = game
        self.adapter = adapter
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.num_sampling_moves = num_sampling_moves
        self.last_model = model
        self.model_generation = 0
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def prefill_replay_buffer(self):
        while self.replay_buffer.records_count < self.batch_size:
            self.play_game()

    def last_estimator(self):
        def model_estimator(situation):
            observation = self.adapter.get_observation(situation)

            # Add extra batch dimension
            data = [np.expand_dims(a, 0) for a in observation.data]

            # Do prediction
            prediction = self.last_model.predict(data)

            # Extra value and logits from result
            value = prediction[0][0]
            logits = [p for p in prediction[1:][0]]
            return value, self.adapter.decode_actions(observation, logits)
        if self.model_generation == 0:
            return dummy_estimator
        else:
            return model_estimator

    def play_game(self):
        situation = self.game.start()
        num_simulations = self.num_simulations
        max_moves = self.max_moves
        estimator = self.last_estimator()
        while not situation.is_terminal():
            s = MctSearch(situation, estimator)
            s.search(num_simulations)
            move = len(situation.history)
            if move > max_moves:
                break
            if move <= self.num_sampling_moves:
                action = s.best_action_softmax()
            else:
                action = s.best_action_max_visits()
            self._record_search(s)
            situation = s.root.children[action].situation
        return situation

    def train_network(self, n_batches=1, epochs=1):
        model = self.last_model  # TODO: clone model
        for _ in range(n_batches):
            batch = self.replay_buffer.get_batch(self.batch_size)
            model.fit(batch.inputs[0], [batch.target_values, batch.target_policy_logits], epochs=epochs)
        self.last_model = model
        self.model_generation += 1

    def make_strategy(self, num_simulations=None):
        if num_simulations is None:
            num_simulations = self.num_simulations
        return AlphaZeroStrategy(self.game, self.adapter, self.last_estimator(), num_simulations)

    def _record_search(self, search):
        children = search.root.children
        values = []
        p = []
        for action in children:
            values.append(action)
            p.append(children[action].visit_count)
        policy_target = self.adapter.encode_actions(Distribution(values, p, norm=True))
        #if not search.root.situation.history:
        #    print("VIS")
        #    for a, c in search.root.children.items():
        #        print(a, c.visit_count)
        #    print("TARGET", policy_target)
        data = self.adapter.get_observation(search.root.situation).data
        assert len(data) == len(self.adapter.data_shapes)
        record = ReplayRecord(data,
                              search.root.value,
                              policy_target)
        self.replay_buffer.add_record(record)