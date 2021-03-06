import random
from random_tree import RandomTree
from collections import Counter

def select_attributes(dataset, attributes, nattributes):
    """
    :param dataset: the dataset
    :param percentage_train: float, percentage of instances that needs to go to the test partition
    :return: (selected_dataset, selected_attributes)
    """
    selected_attributes = dict(random.sample(attributes.items(), nattributes))
    selected_dataset = dataset[list(selected_attributes.keys()) + ['y']]
    return selected_dataset, selected_attributes


def bootstrap(dataset):
    """
    :param dataset: the full dataset
    :return: a bootstrap (sample with reposition)
    """
    return dataset.sample(n=len(dataset), replace=True)


def gen_random_forest(dataset, attributes, ntrees, nattributes, depth_limit=None):
    """
    :param dataset: the dataset dataframe
    :param attributes: the attributes dict
    :param ntrees: int, number of trees
    :param nattributes: int, number of attributes of each tree (aka. "m")
    Algorithm in slide 32 class 16
    """
    return [RandomTree(*select_attributes(bootstrap(dataset), attributes, nattributes),depth_limit=depth_limit) for _ in range(0, ntrees)]


class RandomForest:

    def __init__(self, dataset, attributes, ntrees, nattributes, depth_limit):
        self.random_forest = gen_random_forest(
            dataset, attributes, ntrees, nattributes, depth_limit=depth_limit)
        self.attributes = attributes
        self.dataset = dataset

    def classify(self, instance, stdout=False):
        """
        :param instance: a instance
        :param stdout: if stdout == True, print voting result (debug)
        return class
        In case of a tie in the voting, use a coinflip (random.choice()) between the 'y' values that have tied
        """
        votes = Counter([tree.classify(instance, stdout) for tree in self.random_forest]).most_common()
        if stdout:
            print(votes)
        votes = [vote for vote in votes if vote[1] == votes[0][1]]
        return random.choice(votes)[0]

