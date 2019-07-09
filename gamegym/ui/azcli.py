import argparse
import os
import tqdm
from gamegym.algorithms.mcts import alphazero


class AlphaZeroCli:

    def __init__(self):
        self.games = {}
        self.model_builders = {}
        self.alphazero_config = {}

    def set_config(self, **kw):
        self.alphazero_config.update(kw)

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
        model = self.model_builders[args.model](game)

        model_dir = self._model_dir(args.game)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        az = alphazero.AlphaZero(
            game, model, **self.alphazero_config)
        az.prefill_replay_buffer()

        for i in tqdm.tqdm(range(1, args.steps + 1)):
            az.do_step()
            if i % args.write_step == 0 or i == args.steps:
                for (p, m) in enumerate(model.keras_models):
                    filename = os.path.join(model_dir, "{}-step{}-p{}".format(name, i, p))
                    m.save(filename)


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
        return parser.parse_args()

    def main(self):
        args = self._parse_args()
        if args.command == "info":
            self._command_info(args)
        elif args.command == "train":
            self._command_train(args)
        else:
            raise Exception("Invalid command")