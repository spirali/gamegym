
import numpy as np
from .search import MctSearch
from .buffer import ReplayBuffer, ReplayRecord

from gamegym.utils import Distribution, flatten_array_list


def dummy_estimator(situation):
    return np.zeros(situation.game.players), Distribution(situation.actions, None)


class AlphaZero:

    def __init__(self,
                 game,
                 model,
                 batch_size=128,
                 replay_buffer_size=4096,
                 max_moves=1000,
                 num_simulations=800,
                 num_sampling_moves=30):
        assert batch_size <= replay_buffer_size
        self.game = game
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.num_sampling_moves = num_sampling_moves
        self.model = model
        self.batch_size = batch_size
        self.replay_buffers = [ReplayBuffer(replay_buffer_size) for _ in range(model.number_of_models)]

    def prefill_replay_buffer(self):
        while any(b.records_count < self.batch_size for b in self.replay_buffers):
            self.play_game()

    def last_estimator(self):
        model = self.model
        if model.trained:
            return model.estimate
        else:
            return dummy_estimator

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

    def train_model(self, n_batches=1, epochs=1):
        model = self.model  # TODO: Clone model?
        for _ in range(n_batches):
            for index, buffer in enumerate(self.replay_buffers):
                batch = buffer.get_batch(self.batch_size)
                model.fit(index, batch.inputs, batch.target_values, batch.target_policy_logits, epochs=epochs)
        self.model = model

    def make_strategy(self, num_simulations=None):
        if num_simulations is None:
            num_simulations = self.num_simulations
        return AlphaZeroStrategy(self.game, self.model.adapter, self.last_estimator(), num_simulations)

    def do_step(self, epochs=1, sample_gen_ratio=4):
        if any(not b.added or (b.sampled / b.added) > sample_gen_ratio for b in self.replay_buffers):
            self.play_game()
        else:
            self.train_model(epochs)

    def _record_search(self, search):
        root = search.root
        children = root.children
        values = []
        p = []
        for action in children:
            values.append(action)
            p.append(children[action].visit_count)

        model = self.model
        situation = root.situation
        value = model.make_train_value(situation, root.value)
        policy_target = model.make_train_policy_target(situation, Distribution(values, p, norm=True))
        data = model.make_train_input(situation)

        record = ReplayRecord(data,
                              value,
                              policy_target)
        self.replay_buffers[model.model_index(situation)].add_record(record)