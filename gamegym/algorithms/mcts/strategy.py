from gamegym import Strategy
from .search import MctSearch
from gamegym.utils import Distribution


class MctsStrategy(Strategy):

    def __init__(self, game, adapter, estimator, num_simulations):
        super().__init__(game, adapter)
        self.estimator = estimator
        self.num_simulations = num_simulations

    def get_policy(self, situation):
        s = MctSearch(situation, self.estimator)
        s.search(self.num_simulations)
        action = s.best_action_max_visits()
        return Distribution.single_value(action)
