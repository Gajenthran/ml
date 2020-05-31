import argparse
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def argument_parser(test_size=0.4, learning_rate=0.01, miter=500, hidden_layer=10):
  """
    Parser les arguments de la ligne de commande
    pour choisir les différents paramètres. 
    :param test_size: part des données tests
    :param n_clusters: nombre de clusters pour KMeans
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("-ts", "--test_size", type=float, default=test_size)
  parser.add_argument("-lr", "--learning_rate", type=float, default=learning_rate)
  parser.add_argument("-i", "--iter", type=int, default=miter)
  parser.add_argument("-hl", "--hidden_layer", type=int, default=hidden_layer)

  args = parser.parse_args()
  return args.test_size, args.learning_rate, args.iter, args.hidden_layer

def main():
  test_size, learning_rate, miter, hidden_layer = argument_parser()

  iris = datasets.load_iris()

  X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=test_size)

  mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layer, 
    solver='sgd', 
    learning_rate_init=learning_rate,
    max_iter=miter)
  mlp.fit(X_train, y_train)
  predicted = mlp.predict(X_test)

  print("predicted score: {}".format(np.mean(predicted == y_test)))

  return 0

if __name__ == '__main__':
  main()