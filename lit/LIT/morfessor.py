import math
import morfessor


class MorfessorTrainer:
    def __init__(self, corpus_path: str):
        self.io = morfessor.MorfessorIO()
        self.train_data = list(self.io.read_corpus_file(corpus_path))
        self.models = {}

    @staticmethod
    def _log_fn(x):
        return int(round(math.log(x + 1, 2)))

    def train(self):
        model_types = morfessor.BaselineModel()
        model_logtokens = morfessor.BaselineModel()
        model_tokens = morfessor.BaselineModel()

        model_types.load_data(self.train_data, count_modifier=lambda x: 1)
        model_logtokens.load_data(self.train_data, count_modifier=self._log_fn)
        model_tokens.load_data(self.train_data)

        self.models = {
            'types': model_types,
            'logtokens': model_logtokens,
            'tokens': model_tokens,
        }

        for model in self.models.values():
            model.train_batch()

    def save(self, output_dir: str = 'data/processed'):
        for name, model in self.models.items():
            path = f'{output_dir}/morfessor_{name}.txt'
            self.io.write_segmentation_file(path, model.get_segmentations())
            print(f'Saved {name} segmentation -> {path}')