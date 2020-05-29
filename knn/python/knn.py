import argparse
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def argument_parser(test_size=0.3, n_neighbors=3):
  """
    Parser les arguments de la ligne de commande
    pour choisir les différents paramètres. 
    :param test_size: part des données tests
    :param threshold: seuil pour CM
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("-ts", "--test_size", type=float, default=test_size)
  parser.add_argument("-n", "--neighbors", type=float, default=n_neighbors)

  args = parser.parse_args()

  return args.test_size, args.neighbors

def main():
  test_size, n_neighbors = argument_parser()

  iris = datasets.load_iris()

  X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=test_size)

  knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=2)
  knn.fit(X_train, y_train)
  predicted = knn.predict(X_test)

  print("predicted score: {}".format(np.mean(predicted == y_test)))

  return 0

if __name__ == '__main__':
  main()