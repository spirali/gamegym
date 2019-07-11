import argparse
import os
import tqdm
from gamegym.algorithms.mcts import alphazero, model as mcts_model
from gamegym.algorithms.stats import play_strategies
import keras


class AlphaZeroCli:

    def __init__(self, keras_custom_objects, num_simulations=64, max_moves=100, batch_size=128, replay_buffer_size=6000):
        self.games = {}
        self.model_builders = {}
        self.keras_custom_objects = keras_custom_objects
        self.num_simulations = num_simulations
        self.max_moves = max_moves
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size

    def register_game(self, name, game):
        self.games[name] = game

    def register_model(self, name, model_builder):
        self.model_builders[name] = model_builder

    def _model_dir(self, game_name):
        return os.path.join("game-{}".format(game_name), "models")

    def _command_info(self, args):
        print("Games:")
        for game in sorted(self.games):
            print("\t" + game)
        print("Models:")
        for model in sorted(self.model_builders):
            print("\t" + model)

    def _command_train(self, args):
        name = args.name or args.model
        game = self.games[args.game]
        player_models = self.model_builders[args.model](game)
        assert len(player_models) == game.players
        model = mcts_model.KerasModel(False, game.TensorAdapter(game), False, player_models)


        model_dir = self._model_dir(args.game)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        az = alphazero.AlphaZero(
            game,
            model,
            num_simulations=self.num_simulations,
            max_moves=self.max_moves,
            batch_size=self.batch_sizes,
            replay_buffer_size=self.replay_buffer_size)
        az.prefill_replay_buffer()

        for i in tqdm.tqdm(range(1, args.steps + 1)):
            az.do_step()
            if i % args.write_step == 0 or i == args.steps:
                for (p, m) in enumerate(model.keras_models):
                    m.save(self._player_file(args.game, name, p, i))

    def _player_file(self, game_name, name, player_pos, steps):
        return os.path.join(self._model_dir(game_name), "{}-p{}@{}".format(name, player_pos, steps))

    def _load_player(self, game, player_pos, string):
        if "@" in string:
            name, steps = string.split("@")
            try:
                steps = int(steps)
            except ValueError:
                raise Exception("Invalid number of steps in name: {}".format(string))
            path = self._player_file(game, name, player_pos, steps)
            return keras.models.load_model(path, custom_objects=self.keras_custom_objects)
        else:
            raise Exception("Invalid player name: {}", string)

    def _command_sample_play(self, args):
        game = self.games[args.game]
        players = args.p
        num_simulations = args.simulations or self.num_simulations

        if len(players) != 1 and len(players) != game.players:
            raise Exception("Invalid number of players")

        if not players:
            raise Exception("Players not defined")
        adapter = game.TensorAdapter(game)
        models = [mcts_model.KerasModel(False, adapter, True,
                                        [self._load_player(args.game, i, p) for i in range(game.players)])
            for p in players]
        strategies = [m.make_strategy(num_simulations=num_simulations) for m in models]
        if len(strategies) == 1:
            strategies = strategies * game.players
        sit = play_strategies(game, strategies, after_move_callback=lambda sit: print(game.show_board(sit, colors=True)))
        print("Payoff:", sit.payoff)

    def _parse_args(self):
        parser = argparse.ArgumentParser("AlphaZeroCli")
        subparsers = parser.add_subparsers(title="command", dest="command")
        p = subparsers.add_parser('info')

        p = subparsers.add_parser('train')
        p.add_argument("game", choices=sorted(self.games))
        p.add_argument("model", choices=sorted(self.model_builders))
        p.add_argument("steps", type=int)
        p.add_argument("--write-step", type=int, default=100)
        p.add_argument("--name")

        p = subparsers.add_parser('sample-play')
        p.add_argument("game", choices=sorted(self.games))
        p.add_argument("-p", action="append")
        p.add_argument("--simulations", type=int)
        p.add_argument("--repeat", type=int, default=1)
        return parser.parse_args()

    def main(self):
        args = self._parse_args()
        if args.command == "info":
            self._command_info(args)
        elif args.command == "train":
            self._command_train(args)
        elif args.command == "sample-play":
            self._command_sample_play(args)
        else:
            raise Exception("Invalid command")