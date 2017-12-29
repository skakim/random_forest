import csv
import pandas as pd
import numpy as np
from statistics import mean, stdev
import math
import random
from random_tree import RandomTree
from argv_parser import parser
from random_forest import RandomForest


def read_dataset(dataset):
    """
    :param filename: string, the name of the dataset
    :return: pandas DataFrame
    """
    data = None
    attrs = {}
    if dataset == 'survival':
        # age
        # year
        # aux_nodes
        # survival (output)
        columns = ['age', 'year', 'aux_nodes', 'y']
        data = pd.read_csv("datasets/haberman/haberman.data", names=columns)
        attrs = {x: 'numerical' for x in columns}
        del attrs['y']

    elif dataset == 'wine':
        # class (output)
        # alcohol
        # malic
        # ash
        # alcal
        # magnes
        # phenols
        # flavan
        # n-flavan
        # prc
        # color
        # hue
        # od
        # proline
        columns = ['y', 'alcohol', 'malic', 'ash', 'alcal', 'magnes', 'phenols',
                   'flavan', 'n-flavan', 'prc', 'color', 'hue', 'od', 'proline']
        data = pd.read_csv("datasets/wine/wine.data", names=columns)
        attrs = {x: 'numerical' for x in columns}
        del attrs['y']

    elif dataset == 'contraceptive':
        # wage
        # weduc
        # heduc
        # child
        # wrelig
        # wwork
        # hoccup
        # sol
        # mexp
        # method (output)
        columns = ['wage', 'weduc', 'heduc', 'child', 'wrelig',
                   'wwork', 'hoccup', 'sol', 'mexp', 'y']
        data = pd.read_csv("datasets/cmc/cmc.data", names=columns)
        attrs = {'wage': 'numerical', 'weduc': 'categorical', 'heduc': 'categorical', 'child': 'numerical',
                 'wrelig': 'binary', 'wwork': 'binary', 'hoccup': 'categorical', 'sol': 'categorical', 'mexp': 'binary'}

    elif dataset == 'cancer':
        # diagnosis: M-B (output)
        columns = ['id', 'y']
        for name in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
                     'concave_points', 'symmetry', 'fractal_dim']:
            columns += [name + "_mean", name + "_stderror", name + "_worse"]
        data = pd.read_csv(
            "datasets/breast-cancer-wisconsin/wdbc.data", names=columns)
        del data['id']
        attrs = {x: 'numerical' for x in columns}
        del attrs['y']
        del attrs['id']

    return data, attrs


def holdout(dataset, percentage_train):
    """
    :param dataset: the full dataset
    :param percentage_train: float, percentage of instances that needs to go to the test partition
    :return: (train_dataset, test_dataset (aka Out-of-Bag))
    """
    y_values = list(set(dataset['y']))
    sub_train_datasets = []
    sub_test_datasets = []
    for y_value in y_values:
        sub_dataset = dataset[dataset.y == y_value]
        sub_train_dataset = sub_dataset.sample(frac=percentage_train)
        sub_test_dataset = sub_dataset.drop(sub_train_dataset.index)
        sub_train_datasets.append(sub_train_dataset)
        sub_test_datasets.append(sub_test_dataset)
        # print(y_value,len(sub_dataset),len(sub_train_dataset),len(sub_test_dataset))
    train_dataset = pd.concat(sub_train_datasets)
    test_dataset = pd.concat(sub_test_datasets)
    # print(len(train_dataset),len(test_dataset),len(train_dataset)/len(dataset))
    return (train_dataset, test_dataset)


def cross_validation(dataset, attributes, percentage_train, folds, ntrees, nattributes=-1):
    """
    :param dataset: the full dataset
    :param attributes: the attributes
    :param percentage_train: float, percentage of instances that needs to go to the train partition
    :param folds: int, number of holdouts to execute
    :param ntrees: int, number of trees for each ensemble
    :param nattributes: int, number of sampled attributes (if -1, use sqrt(total))
    :return: (average_performance, stddev_performance)
    """
    if nattributes == -1:
        nattributes = int(math.sqrt(len(dataset.columns)))

    accuracies = []
    precisions = []
    recalls = []
    for fold in range(1, folds + 1):
        # print("Iteration",fold)
        train_dataset, test_dataset = holdout(dataset, percentage_train)
        # rf = RandomForest(train_dataset, attributes, ntrees, nattributes)
        accuracy, precision, recall = test_RF(None, test_dataset)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
    return mean(accuracies), stdev(accuracies), mean(precisions), stdev(precisions), mean(recalls), stdev(recalls)


def accuracy(cm, n_instances):
    acc = 0
    for i in range(len(cm)):
        acc += cm[i][i]
    return acc / n_instances


def precision(cm):
    acc = 0.0
    cm = np.array(cm)
    if len(cm) == 2:
        true_positive = cm[0, 0]
        false_positive = sum(cm[:, 0]) - cm[0, 0]
        if true_positive + false_positive != 0.0:
            acc = true_positive / (true_positive + false_positive)
    else:
        for i in range(len(cm)):
            true_positive = cm[i, i]
            false_positive = sum(cm[:, i]) - cm[i, i]
            if true_positive + false_positive != 0.0:
                p = true_positive / (true_positive + false_positive)
                # print("precision\n", cm, true_positive,false_positive,p)
                acc += p
        acc = acc / len(cm)

    return acc


def recall(cm):
    acc = 0.0
    cm = np.array(cm)

    if len(cm) == 2:
        true_positive = cm[0, 0]
        false_negative = sum(cm[0, :]) - cm[0, 0]
        if true_positive + false_negative != 0.0:
            acc = true_positive / (true_positive + false_negative)
    else:
        for i in range(len(cm)):
            true_positive = cm[i, i]
            false_negative = sum(cm[i, :]) - cm[i, i]
            if true_positive + false_negative != 0.0:
                r = true_positive / (true_positive + false_negative)
                # print("recall\n", cm, true_positive, false_negative, r)
                acc += r
        acc = acc / len(cm)
    return acc


def test_RF(RF, test_dataset):
    """
    :param RF: the random forest object
    :param test_dataset: the test dataframe
    :return: the performance of the RF
    """
    classes = list(set(test_dataset['y']))
    # print(classes)
    confusion_matrix = [[0.0] * len(classes) for _ in range(len(classes))]
    test_dataset = test_dataset.transpose().to_dict()
    number_of_instances = len(test_dataset)
    for instance in test_dataset.values():
        # print(instance)
        expected = instance['y']
        # y = RF.classify(instance)
        y = 1
        confusion_matrix[classes.index(expected)][classes.index(y)] += 1
    # print(confusion_matrix)
    return (accuracy(confusion_matrix, number_of_instances),
            precision(confusion_matrix),
            recall(confusion_matrix))


def print_cross_validation(return_value):
    mean_accuracies, stdev_accuracies, mean_precisions, stdev_precisions, mean_recalls, stdev_recalls = return_value
    print("Accuracies mean:", mean_accuracies, " stdev:", stdev_accuracies)
    print("Precisions mean:", mean_precisions, " stdev:", stdev_precisions)
    print("Recalls mean:", mean_recalls, " stdev:", stdev_recalls)


if __name__ == "__main__":
    mode_parser = parser()

    if str(mode_parser.mode) == 'verify':
        # benchmark
        m = []
        with open('datasets/benchmark/benchmark.csv') as bfile:
            breader = csv.reader(bfile, delimiter=';')
            for row in breader:
                m.append(row)
        attributes = {x: 'categorical' for x in m[0][:-1]}
        attributes_names = m[0][:-1]
        del m[0]
        dataset = pd.DataFrame(m, columns=attributes_names + ['y'])
        RT = RandomTree(dataset, attributes)
        RT.print_tree()

    elif str(mode_parser.mode) == 'wine':
        dataset, attributes = read_dataset('wine')

    elif str(mode_parser.mode) == 'survival':
        dataset, attributes = read_dataset('survival')

    elif str(mode_parser.mode) == 'cancer':
        dataset, attributes = read_dataset('cancer')

    elif str(mode_parser.mode) == 'contraceptive':
        dataset, attributes = read_dataset('contraceptive')
    else:
        dataset, attributes = read_dataset('contraceptive')

    """
    forest = RandomForest(dataset, attributes, 10, 10)
    for row in dataset:
        print(row)
    """
    print(mode_parser.mode)
    n_trees = [1, 5, 10, 25, 50]
    for n in n_trees:
        print("#trees =", n)
        print_cross_validation(cross_validation(
            dataset, attributes, 0.8, 5, n))

    """
    #RandomTree debug only
    dataset, attributes = read_dataset('survival')
    RT = RandomTree(dataset, attributes)
    RT.print_tree()
    dataset, attributes = read_dataset('wine')
    RT = RandomTree(dataset, attributes)
    RT.print_tree()
    dataset, attributes = read_dataset('contraceptive')
    RT = RandomTree(dataset, attributes)
    RT.print_tree()
    dataset, attributes = read_dataset('cancer')
    RT = RandomTree(dataset, attributes)
    RT.print_tree()
    """
