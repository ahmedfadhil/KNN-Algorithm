import numpy as np
import math as sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[5, 6], [7, 7], [8, 5]]}
new_feature = [5, 7]
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_feature[0], new_feature[1])
# plt.show()


# KNN algorithm
def knn_alg(data, predict, k=3):
    if len(data) > k:
        return warnings.warn('k is set to a value less than the total voting groups!')
    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([euclidean_distance, group])
        votes = [i[1] for i in sorted(distance)[:k]]
        print(Counter(votes).most_common(1))
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result

    result = knn_alg(dataset, new_feature, k=3)
    print(result)


[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color= result)
plt.show()
