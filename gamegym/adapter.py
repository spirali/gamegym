import re
from typing import Any, Dict, Iterable, NewType, Tuple

import numpy as np

from .game import Game
from .errors import DecodeObservationInvalidData
from .observation import Observation, ObservationData
from .situation import Action, Situation
from .utils import Distribution, flatten_array_list


class Adapter():
    """
    Adapter extracts specific type of observation from a game situation.

    Adapter both filters the visible observation (leaving only what is visible
    to the requested played) and feormats the observation in a format suitable for
    the strategy. The following types of adapters and strategies exist
    (but you can create your own):

    * Text representation (for console play, or textual neural nets)
    * Hashable data representation (for tabular strategies)
    * Numeric ndarrays (for neural networks)
    * JSON/SVG/... (for displying in a web gui)

    Some games have symmetric representation for both players (e.g. gomoku),
    for those games the default adapter behavious is that to report two
    symmetric situations as distinct. When you create such adapters with
    `symmetrize=True`, they will produce all observations as if the active player
    was player 0. Note that this should be done also for public observation.

    Adapter may or may not define observation from a non-current player's
    point of view, but is not required to do so.
    Note that full information games give the same information from any player's
    point of view, regardless of symmetrization.
    """
    SYMMETRIZABLE = False

    def __init__(self, game: Game, symmetrize=False):
        assert isinstance(game, Game)
        self.game = game
        assert self.SYMMETRIZABLE or not symmetrize
        self.symmetrize = symmetrize

    def get_observation(self, sit: Situation, player: int = None) -> Observation:
        """
        Create an `Observation` object for given situation.

        Internally uses `observe_data`. By default, provides an
        observation from the point of active player.
        Use `player=-1` to request public state.

        Some adapters may not provide observations for e.g. inactive players or,
        in rare cases, even for all situations of the active player.
        """
        if player is None:
            player = sit.player
        data = self.observe_data(sit, player)
        return Observation(sit.game, sit.actions, player, data)

    def observe_data(self, situation: Situation, player: int) -> ObservationData:
        """
        Provide the observation data from the point of view of the
        specified player.

        NOTE: symm and public

        Raise `ObservationNotAvailable` where the observation is not
        specified.
        """
        raise NotImplementedError

    def decode_actions(self, observation: Observation, data: Any) -> Distribution:
        """
        Decode given data from the strategy to an action distribution.

        Useful for e.g. tensor, RPC and text adapters. If a strategy creates
        distributions directly, there is no need to implement this.

        Should raise `DecodeObservationInvalidData` on invalid data (e.g. for CLI input).
        """
        raise NotImplementedError


class BlindAdapter(Adapter):

    def observe_data(self, situation: Situation, player: int):
        return None

class TextAdapter(Adapter):
    # Ignore all letter case
    IGNORE_CASE = False
    # Ignore all whitespace
    IGNORE_WHITESPACE = False
    # Convert any whitespace sequence as a single space
    IGNORE_MULTI_WHITESPACE = True
    # Ignore parens and comma on decode "(,)"
    IGNORE_PARENS = False

    """
    Adds action listing, color, aliases and default action text decoding.

    `self.action_names` is a mapping from (canonical) action names to
    """
    def __init__(self, game, colors=False, symmetrize=False):
        super().__init__(game, symmetrize=symmetrize)
        self.action_aliases = self.get_action_aliases()
        self.alias_to_action = {}
        for a in self.game.actions:
            aliases = self.action_aliases[a]
            if isinstance(aliases, str):
                aliases = (aliases, )
            assert len(aliases) > 0
            for al in aliases:
                assert al not in self.alias_to_action
                self.alias_to_action[al] = a
        self.colors = colors

    def _canonicalize_name(self, s: Any) -> str:
        "canonicalize "
        s = str(s)
        if self.IGNORE_CASE:
            s = s.lower()
        if self.IGNORE_WHITESPACE:
            s = re.sub(r'\s+', ' ', s)
        if self.IGNORE_WHITESPACE:
            s = re.sub(r'\s', '', s)
        if self.IGNORE_PARENS:
            s = re.sub(r'[(),]', '', s)
        return s

    def get_action_aliases(self) -> Dict[Action, Tuple[str]]:
        """
        Return a dict from action to tuple of (canonicalized) action names.

        By default uses `str(action)` for every action.
        """
        return {a: (self._canonicalize_name(a), ) for a in self.game.actions}

    def decode_actions(self, observation, text):
        name = self._canonicalize_name(text.strip())
        try:
            action = self.alias_to_action[name]
        except KeyError:
            raise DecodeObservationInvalidData
        if action not in observation.actions:
            raise DecodeObservationInvalidData
        return Distribution([action], None)

    def colored(self, text, color=None, on_color=None, attrs=None):
        """
        Optionally color the given text using termcolor.
        """
        if self.colors:
            import termcolor  # TODO(gavento): Is this slow or bad practice?
            return termcolor.colored(text, color, on_color, attrs)
        return text

    def actions_to_text(self, actions: Iterable[Action]):
        """
        List available action names (with opt coloring).

        Uses first names from `self.action_aliases`.
        """
        return self.colored(', ', 'white', None, ['dark']).join(self.colored(self.action_aliases[a][0], 'yellow') for a in actions)


class TensorShape:

    __slots__ = ("input_shape", "shaped_actions", "flatten_actions", "actions_index")

    def __init__(self, input_shape, shaped_actions):
        self.input_shape = input_shape

        shaped = shaped_actions
        if isinstance(shaped, np.ndarray):
            flatten = shaped.flatten()
            shaped = (shaped,)
        else:
            flatten = flatten_array_list(shaped)

        self.shaped_actions = shaped
        self.flatten_actions = flatten
        self.actions_index = {a: i for i, a in enumerate(flatten)}


class TensorAdapter(Adapter):
    """
    Used to encode

    Also provides methods to decode action distribution from neural net output,
    and encode target policy into neural net output for training.

    By default the decoding assumes the neural net output is a 1D probability vector
    indexed by actions. Other shapes and action ordering in the output can be
    obtained by overriding `_generate_shaped_actions`, or by reimplementing both
    `decode_actions` and `encode_actions`.
    """
    def __init__(self, game, symmetrize=False):
        super().__init__(game, symmetrize=symmetrize)
        self.shapes = self._generate_shapes()

    def _generate_shapes(self) -> Tuple[TensorShape]:
        raise NotImplementedError

    def shape_index(self, situation: Situation):
        """
            Override if more action shapes are provided
        """
        return 0

    def decode_actions(self, observation: Observation, named_action_arrays: Tuple[str, Tuple[np.ndarray]]) -> Distribution:
        """
        Decode a given tuple of likelihood ndarrays to a (normalized) distribution on valid actions.
        """
        # check shapes
        name, action_arrays = named_action_arrays
        shaped_actions = self.shaped_actions[name]
        assert len(shaped_actions) == len(action_arrays)
        for i in range(len(action_arrays)):
            assert shaped_actions[i].shape == action_arrays[i].shape
        policy = flatten_array_list(action_arrays)
        if np.sum(policy) < 1e-30:
            policy = None  # Uniform dstribution
        return Distribution(self.flatten_actions[name], policy, norm=True)

    def encode_actions(self, situation: Situation, dist: Distribution) -> Tuple[np.ndarray]:
        actions_index = self.actions_index
        flatten = np.zeros(len(actions_index))
        for a, p in dist.items():
            flatten[actions_index[a]] = p

        assert len(self.shaped_actions) == 1  # TODO: other options
        return flatten.reshape(self.shaped_actions[0].shape)


#def test_adapter():
#    g = Gomoku(3,3,3)
#    ad = Gomoku.TextAdapter(g)
#    s1 = ConsolePlayer(ad, prompt="Player 1")
#    s2 = ConsolePlayer(ad, prompt="Player 2")
#    g.play_strategies([s1, s2])
