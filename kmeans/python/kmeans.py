import argparse
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def argument_parser(test_size=0.2, n_clusters=3):
  """
    Parser les arguments de la ligne de commande
    pour choisir les différents paramètres. 
    :param test_size: part des données tests
    :param threshold: seuil pour CM
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("-ts", "--test_size", type=float, default=test_size)
  parser.add_argument("-c", "--cluster", type=float, default=n_clusters)

  args = parser.parse_args()

  return args.test_size, args.cluster

def main():
  test_size, n_clusters = argument_parser()

  iris = datasets.load_iris()

  X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=test_size)

  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(X_train)
  predicted = kmeans.predict(X_test)

  print("predicted score: {}".format(np.mean(predicted == y_test)))

  return 0

if __name__ == '__main__':
  main()