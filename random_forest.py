import random
from random_tree import RandomTree


def select_attributes(dataset, attributes, nattributes):
    """
    :param dataset: the dataset
    :param percentage_train: float, percentage of instances that needs to go to the test partition
    :return: (selected_dataset, selected_attributes)
    """
    selected_attributes = dict(random.sample(attributes.items(), nattributes))
    selected_dataset = dataset[list(selected_attributes.keys()) + ['y']]
    return (selected_dataset, selected_attributes)


def bootstrap(dataset):
    """
    :param dataset: the full dataset
    :return: a bootstrap (sample with reposition)
    """
    return dataset.sample(n=len(dataset), replace=True)


def gen_random_forest(dataset, attributes, ntrees, nattributes):
    """
    :param dataset: the dataset dataframe
    :param attributes: the attributes dict
    :param ntrees: int, number of trees
    :param nattributes: int, number of attributes of each tree (aka. "m")
    Algorithm in slide 32 aula 16
    Probably will be something like:
        random_forest = []
        for _ in ntrees:
            tree_dataset, tree_attributes = select_attributes(bootstrap(dataset),attributes, nattributes)
            rt = RandomTree(tree_dataset, tree_attributes)
            random_forest.append(rt)
    """
    random_forest = []
    return random_forest


class RandomForest:

    def __init__(self, dataset, attributes, ntrees, nattributes):
        self.random_forest = gen_random_forest(
            dataset, attributes, ntrees, nattributes)
        self.attributes = attributes

    def classify(self, instance, stdout=False):
        """
        :param instance: a instance
        :param stdout: if stdout == True, print voting result (para debug)
        return class
        Look at random_tree classify, probably will be a loop calling rt.classify and return who wins the voting
        """
        return 0
