"""

"""
import operator
import numpy as np

from copy import deepcopy
from collections import Counter
from itertools import combinations

from keras.models import Model, clone_model
from keras.layers import Dense


class AllVsAllModel(Model):
    """
    Description:
    This model trains and predicts using an All-vs-All scheme.

    Attributes:
    """
    def fit(self, x, y, epochs, batch_size, epochs_sub=12, batch_size_sub=128, **kwargs):
        super().fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)

        if "validation_data" in kwargs.keys():
            kwargs.pop("validation_data")

        # train second layer models
        self.pair_models = {}
        y_array = np.argmax(y, axis=1).ravel()
        self.labels = sorted(list(set(y_array)))
        pair_list = list(combinations(self.labels, 2))
        for pair in pair_list:
            idx1 = [i for i in range(x.shape[0]) if y_array[i] == pair[0]]
            idx2 = [i for i in range(x.shape[0]) if y_array[i] == pair[1]]

            x_class = x[idx1+idx2, :]
            y_class = np.zeros((x_class.shape[0], 2))
            y_class[:len(idx1), 0] = 1
            y_class[-len(idx2):, 1] = 1

            res = {pair: self.fit_pair(x_class, y_class, batch_size=batch_size_sub, epochs=epochs_sub, **kwargs)}
            self.pair_models.update(res)


    def fit_pair(self, x, y, batch_size, epochs, **kwargs):
        """

        """
        top_layer = (Dense(units=2, activation="softmax"))(self.layers[-2].output)
        pair_model = Model(self.input, top_layer)
        pair_model = clone_model(pair_model)
        pair_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        pair_model.fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)
        return pair_model


    def predict(self, x):
        """

        """
        labels = {}
        [labels.update({l:0.0}) for l in self.labels]
        results = [deepcopy(labels) for _ in range(x.shape[0])]
        for pair, pair_model in self.pair_models.items():
            pair_res = pair_model.predict(x)
            for i in range(x.shape[0]):
                results[i][pair[0]] = results[i][pair[0]] + pair_res[i][0]
                results[i][pair[1]] = results[i][pair[1]] + pair_res[i][1]

        results = [max(res.items(), key=operator.itemgetter(1))[0] for res in results]
        return results


class HierarchicalModel(Model):
    """
    Description:
    This model uses a hierarchical approach to narrow down class choices.

    Attributes:
    """
    def fit(self, x, y, epochs, batch_size, epochs_sub=1, batch_size_sub=128, **kwargs):
        #super().fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)

        if "validation_data" in kwargs.keys():
            kwargs.pop("validation_data")

        # train combination layer models
        self.comb_models = {}
        y_array = np.argmax(y, axis=1).ravel()
        self.labels = sorted(list(set(y_array)))
        for i in range(len(self.labels)-1):
            comb_list = list(combinations(self.labels, len(self.labels)-i))
            for comb in comb_list:
                print(comb)
                idx = [j for j in range(x.shape[0]) if int(np.argmax(y[j])) in comb]
                x_comb = x[idx]
                y_comb = y[idx][:, comb]
                res = {comb: self.fit_comb(x_comb, y_comb, batch_size=batch_size_sub, epochs=epochs_sub, **kwargs)}
                self.comb_models.update(res)

        #  set combination levels
        self.levels = []
        for i in range(len(self.labels) - 1):
            level = []
            for comb in self.comb_models.keys():
                if len(comb) == (len(self.labels) - i):
                    level.append(comb)
            self.levels.append(level)


    def fit_comb(self, x, y, batch_size, epochs, **kwargs):
        """

        """
        top_layer = (Dense(units=y.shape[1], activation="softmax"))(self.layers[-2].output)
        comb_model = Model(self.input, top_layer)
        comb_model = clone_model(comb_model)
        comb_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        comb_model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=True, **kwargs)
        return comb_model


    def predict(self, x):
        """

        """
        comb = deepcopy(self.labels)
        comb_model = self.comb_models[tuple(comb)]
        Y = comb_model.predict(x)
        Y = np.argsort(-Y, axis=1)
        N, d = Y.shape

        # for each subsequent level determine the next level
        for i, levels in enumerate(self.levels):
            if i == 0:
                continue

            # for each model in this level predict
            Y_new = np.zeros((N, d), dtype=float)
            for col_idx in levels:

                # gather row indices corresponding to the col_idx
                col_idx = [int(j) for j in col_idx]
                Y = np.sort(Y[:, :(d-i)], axis=1)
                row_idx = [int(j) for j in range(N) if list(Y[j, :]) == list(col_idx)]

                # predict model on subset of points
                if row_idx:
                    row_ones = np.multiply(np.array(row_idx)[:, None],
                                           np.ones((len(row_idx), len(col_idx)), dtype=int))
                    col_ones = np.multiply(np.array(col_idx)[None, :],
                                           np.ones((len(row_idx), len(col_idx)), dtype=int))

                    comb_model = self.comb_models[tuple(col_idx)]
                    Y_new[row_ones, col_ones] = comb_model.predict(x[row_idx, :])

            # produce new ordering of likely labels
            Y = np.argsort(-Y_new, axis=1)

        # produce final prediction
        y_hat = Y[:, 0]
        return y_hat
