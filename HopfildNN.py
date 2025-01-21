import numpy as np


class HopfildNN:
    def train(self, data):
        d = np.array(data)
        self.__data = np.copy(d)
        W = np.zeros((d.shape[1], d.shape[1]))

        for obr in d:
            obr = np.array([obr])
            W += np.dot(obr.T, obr)
        np.fill_diagonal(W, 0)
        self.__W = W

    def activate(self, x):
        return np.sign(x)

    def predict(self, data):
        i = 0
        res = []

        for test in data:
            test = np.array([test]).T
            test_prev = []
            k = 0


            while not self.__in_res(test, test_prev):
                test_prev.append(np.copy(test))
                test = self.activate(np.dot(self.__W, test))
                k += 1
                if k > 200:
                    break

            distances = np.linalg.norm(self.__data - test.T[0], axis=1)
            threshold = 0.5
            closest_idx = np.argmin(distances) if np.min(distances) < threshold else -1

            res.append(closest_idx)
            i += 1

        return res

    def __in_res(self, test, test_prev):
        for obr in reversed(test_prev):
            if np.array_equal(obr, test):
                return True
        return False


