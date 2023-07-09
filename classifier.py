import igraph as ig
import numpy as np
from math import dist


class PGR:
    """Classification based on importance [Carneiro and Zhao, 2017]

    Attributes:
        fitted: a boolean flag to know if the model is trained or not
        Ea: training efficiency list
        Y: model balancing parameter
        I: training importance list
        X: training dataset
        y: targets list of X
        dist: dissimilarity heuristic (distancy function)

    References:
        Carneiro, M. G., & Zhao, L. (2017). Organizational data classification based on the importance concept of complex networks. IEEE transactions on neural networks and learning systems, 29(8), 3361-3373.

    """

    def __init__(self, Y=2.39, dst=dist):
        self.fitted = False
        self.Ea = {}
        self.Y = Y
        self.I = None
        self.X = None
        self.y = None
        self.dist = dst

    def fit(self, g: ig.Graph, X, y):
        """Training step

        args:
            g: dataset network structure
            X: dataset
            y: targets of X

        """
        E = np.zeros(len(X))
        self.I = g.pagerank()
        q_y = {}
        self.X = X
        self.y = y
        for i, neighbors in enumerate(g.get_adjlist()):
            for j in neighbors:
                E[i] += g[i, j]
            if len(neighbors):
                E[i] /= len(neighbors)
            if y[i] in self.Ea:
                self.Ea[y[i]] += E[i]
            else:
                self.Ea[y[i]] = E[i]
            if y[i] in q_y:
                q_y[y[i]] += 1
            else:
                q_y[y[i]] = 1
        for i in self.Ea:
            self.Ea[i] /= q_y[i]
        self.fitted = True

    def predict(self, y):
        """Predict step

        args:
            y: test sample
        returns:
            target: class label for data sample.
        """
        if not self.fitted:
            raise NoTrainException("To classify a new sample it is necessary to train the model")
        I_y = {}
        A = {}
        target = 0
        target_value = 0

        for i, x_i in enumerate(self.X):
            f = self.Ea[self.y[i]] * self.Y - self.dist(x_i, y)
            if f >= 0:
                if self.y[i] in A:
                    A[self.y[i]].append(i)
                else:
                    A[self.y[i]] = [i]

        for i in A:
            for j in A[i]:
                if self.y[j] in I_y:
                    I_y[self.y[j]] += self.I[j]
                else:
                    I_y[self.y[j]] = self.I[j]

        for i, j in zip(I_y.keys(), I_y.values()):
            if j > target_value:
                target_value = j
                target = i

        return target


class NoTrainException(Exception):
    """the model is not trained"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"
