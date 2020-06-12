# Machine Learning

Ce repertoire comporte un ensemble d'algorithmes lié au machine learning. Les algorithmes seront regroupés sous 3 catégories: apprentissage supervisé, apprentissage non supervisé et apprentissage par renforcement. La plupart des codes que vous verrez sur ce repertoire seront écrit en C et/ou en Python.

## Usage

Pour exécuter un code écrit en C, il suffit de lancer la commande suivante:
```bash
make <repository_name.c>
./<repository_name> iris.data
```

Pour exécuter un code écrit en Python, il suffit de lancer la commande suivante:
```bash
python3 <repository_name.py>
```

## Algorithmes

Les algorithmes dans ce répertoire seront écrits from scratch en C et grâce à la librairie ``` sklearn ``` en Python.

- ```c/```
  - ``` <repo.cfg> ``` fichier de configuration
  - ``` <repo.c> ``` fichier regroupant les algos du modèle
  - ``` parser.c ``` parsing du fichier de configuration + datasets
  - ``` documentation/ ``` documentation (```doxyfile```)

- ```python/```
  - ``` <repo.py> ``` fichier principal utilisant ``` sklearn``` pour le modèle

## TODO 

La liste des éléments à rajouter/modifier dans l'avenir:
- Libération de la mémoire
- Création d'un parser d'arguments 
- API pour uniformiser les codes ``` ml ```
- Datasets à regrouper


## License 

[MIT](https://choosealicense.com/licenses/mit/)
