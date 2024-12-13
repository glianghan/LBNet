import numpy as np
import itertools


class UnmixingUtils:
    def __init__(self, A, S):
        self.A = A
        self.S = S
        pass
    def hyperSAD(self, A_est):
        L = np.size(A_est, 0)
        R = np.size(A_est, 1)
        meanDistance = np.inf
        p = np.array(list(itertools.permutations(np.arange(0, R), R)))
        num = np.size(p, 0)
        sadMat = np.zeros([R, R])
        for i in range(R):
            temp = A_est[:, i]
            temp = np.tile(temp, (R, 1)).T
            sadMat[i, :] = np.arccos(
                sum(temp * self.A, 0) / (np.sqrt(np.sum(temp * temp, 0)+1e-4) * np.sqrt(np.sum(self.A * self.A, 0))+1e-4))
        sadMat = sadMat.T
        temp = np.zeros([1, R])
        sor = p[0, :]
        Distance = np.ones([1, R]) * meanDistance
        for i in range(num):
            for j in range(R):
                temp[0, j] = sadMat[j, p[i, j]]
            if np.mean(temp, 1) < meanDistance:
                sor = p[i, :]
                meanDistance = np.mean(temp, 1)
                Distance = temp.copy()
        if np.isclose(meanDistance, np.inf, atol=1e-8):
            meanDistance=np.array([np.array(-1)])
            Distance = np.ones([1, R]) * meanDistance
        return Distance, meanDistance, sor

    def hyperRMSE(self, S_est, sor):
        N = np.size(self.S, 0)
        S_est = S_est[:, sor]
        rmse = self.S - S_est
        rmse = rmse * rmse
        rmse_all = np.sqrt(np.sum(rmse, 0) / N)
        rmse = np.mean(rmse_all)
        return rmse, rmse_all
