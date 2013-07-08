import numpy as np

class WuLarusClassifier:
    def fit(self, features, target):
        return self

    def predict(self, features):
        [call, guard, loop_branch, loop_exit, loop_header, opcode, pointer,
                ret, store, name, not_taken, taken] = xrange(0, 12)
        heuristics = [0.78, 0.62, 0.88, 0.8, 0.75, 0.84, 0.6, 0.72, 0.55]
        num_records = features.shape[0]
        probs = np.zeros(num_records) + 0.5
        for i in xrange(0, len(heuristics)):
            multiplier = np.ones(num_records)
            matches = features[:, i] == 1
            h = heuristics[i]
            p = probs[matches]
            probs[matches] = p * h / (p * h + (1 - p) * (1 - h))
        return np.array(probs >= 0.5, np.float)

