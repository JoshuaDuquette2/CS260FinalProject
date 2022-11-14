import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

mat = scipy.io.loadmat(r'C:\Users\joshu\Client Work\data\qm7.mat')
c_matrix = mat["X"]
atomized_energies = mat["T"]

sns.kdeplot(atomized_energies.squeeze());

from scipy.stats import skew, kurtosis
cols = ['rms', 'peak2peak', 'skewness', 'kurtosis', 'min', 'max', 'mean', "std"]
statistics = pd.DataFrame(columns=cols)

def extract_features(mat):
    def rms(n):
        return np.sqrt(np.sum(np.square(n))/len(n))
    feavec = []
    feavec.append(rms(mat))
    feavec.append(np.max(mat) - np.min(mat))
    feavec.append(skew(mat, axis=0))
    feavec.append(kurtosis(mat, axis=0))
    feavec.append(np.min(mat))
    feavec.append(np.max(mat))
    feavec.append(np.mean(mat))
    feavec.append(np.std(mat))

    feavec = np.array(feavec)

    return feavec

data = []
for molecule in c_matrix:
    feavec = extract_features(molecule)
    data.append(feavec)

statistics[cols] = data
statistics

statistics["target"] = atomized_energies.squeeze()

for col in set(cols) - set(["skewness", "kurtosis"]):
    statistics.plot(x=[col], y=["target"], kind="hexbin", figsize=(12,10), colorbar=False, colormap="magma");
