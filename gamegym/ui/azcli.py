import argparse
import os
import tqdm
from gamegym.algorithms.mcts import alphazero, model as mcts_model
from gamegym.algorithms.stats import play_strategies
from gamegym.strategy import UniformStrategy
from gamegym.algorithms import tournament

import keras
import seaborn as sns
import matplotlib.pyplot as plt


class AlphaZeroCli:

    def __init__(self, keras_custom_objects, num_simulations=64, max_moves=100, batch_size=128, replay_buffer_size=6000):
        self.games = {}
        self.model_builders = {}
        self.keras_custom_objects = keras_custom_objects
        self.num_simulations = num_simulations
        self.max_moves = max_moves
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.default_game = None

    def register_game(self, name, game):
        self.games[name] = game

        if self.default_game is None:
            self.default_game = name

    def register_model(self, name, model_builder):
        self.model_builders[name] = model_builder

    def _model_dir(self, game_name):
        return os.path.join("game-{}".format(game_name), "models")

    def _tournament_db(self, game_name):
        return os.path.join("game-{}".format(game_name), "tournament.db")

    def _command_info(self, args):
        print("Games:")
        for game in sorted(self.games):
            if game == self.default_game:
                print("\t" + game + " [default]")
            else:
                print("\t" + game)

        print("Models:")
        for model in sorted(self.model_builders):
            print("\t" + model)

    def _command_train(self, args):
        game_name = args.game or self.default_game
        name = args.name or args.model
        game = self.games[game_name]
        player_models = self.model_builders[args.model](game)
        assert len(player_models) == game.players
        model = mcts_model.KerasModel(False, game.TensorAdapter(game), False, player_models)


        model_dir = self._model_dir(game_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        az = alphazero.AlphaZero(
            game,
            model,
            num_simulations=self.num_simulations,
            max_moves=self.max_moves,
            batch_size=self.batch_size,
            replay_buffer_size=self.replay_buffer_size)
        az.prefill_replay_buffer()

        for i in tqdm.tqdm(range(1, args.steps + 1)):
            az.do_step()
            if i % args.write_step == 0 or i == args.steps:
                for (p, m) in enumerate(model.keras_models):
                    m.save(self._player_file(game_name, name, p, i))

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

    def _command_show(self, args):
        game_name = args.game or self.default_game
        game = self.games[game_name]
        tournament_file = self._tournament_db(game_name)
        results = tournament.GameResults.load(tournament_file)
        df = results.player_stats()
        df["avg"] = df.payoff / df.plays
        df["steps"] = df["name"].apply(lambda x: int(x.split("@")[1]) if "@" in x else 0)
        df["basename"] = df["name"].apply(lambda x: x.split("@")[0])
        sns.barplot(data=df, y="avg", x="name")
        plt.show()
        print(df)

    def _command_tournament(self, args):
        game_name = args.game or self.default_game
        game = self.games[game_name]
        adapter = game.TensorAdapter(game)
        players = args.players
        num_simulations = args.simulations or self.num_simulations

        tournament_file = self._tournament_db(game_name)
        strategies = {
            p: mcts_model.KerasModel(False, adapter, True,
                                  [self._load_player(game_name, i, p) for i in range(game.players)])
                                    .make_strategy(num_simulations=num_simulations)
                if p != "uniform" else UniformStrategy()
            for p in set(players)}


        #if os.path.isfile(tournament_file):
        #    results = tournament.GameResults.load(tournament_file)
        #else:
        results = None
        pdb = tournament.PlayerList(game, game_results=results)
        for p in players:
            for i in range(game.players):
                pdb.add_player("{}#{}".format(i, p), strategies[p], i)
        pairing = tournament.AllPlayAllPairing()
        for i in range(args.repeat):
            pdb.play_tournament(i, pairing)
        pdb.game_results.save(tournament_file)

    def _command_sample_play(self, args):
        game_name = args.game or self.default_game
        game = self.games[game_name]
        players = args.players
        num_simulations = args.simulations or self.num_simulations

        if len(players) != 1 and len(players) != game.players:
            raise Exception("Invalid number of players")

        if len(players) == 1:
            players = players * game.players

        if not players:
            raise Exception("Players not defined")
        adapter = game.TensorAdapter(game)
        strategies = {
            p: mcts_model.KerasModel(False, adapter, True,
                                  [self._load_player(game_name, i, p) for i in range(game.players)])
                                    .make_strategy(num_simulations=num_simulations)
                if p != "uniform" else UniformStrategy()
            for p in set(players)}
        sit = play_strategies(game, [strategies[p] for p in players], after_move_callback=lambda sit: print(game.show_board(sit, colors=True)))
        print("Payoff:", sit.payoff)

    def _parse_args(self):
        parser = argparse.ArgumentParser("AlphaZeroCli")
        parser.add_argument("--game", choices=sorted(self.games), default=None)
        subparsers = parser.add_subparsers(title="command", dest="command")
        p = subparsers.add_parser('info')

        p = subparsers.add_parser('train')
        p.add_argument("model", choices=sorted(self.model_builders))
        p.add_argument("steps", type=int)
        p.add_argument("--write-step", type=int, default=100)
        p.add_argument("--name")

        p = subparsers.add_parser('sample-play')
        p.add_argument("players", nargs="+")
        p.add_argument("--simulations", type=int)
        p.add_argument("--repeat", type=int, default=1)

        p = subparsers.add_parser('tournament')
        p.add_argument("players", nargs="+")
        p.add_argument("--simulations", type=int)
        p.add_argument("--repeat", type=int, default=1)

        p = subparsers.add_parser('show')
        return parser.parse_args()

    def main(self):
        args = self._parse_args()
        if args.command == "info":
            self._command_info(args)
        elif args.command == "train":
            self._command_train(args)
        elif args.command == "sample-play":
            self._command_sample_play(args)
        elif args.command == "tournament":
            self._command_tournament(args)
        elif args.command == "show":
            self._command_show(args)
        else:
            raise Exception("Invalid command")