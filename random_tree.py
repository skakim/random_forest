import pandas as pd
from math import log2

# WARNING: NOT TESTED YET

def info(dataset):
    probs = dict(dataset.y.value_counts(normalize=True))
    acc = 0.0
    for atr in probs.keys():
        acc += probs[atr] * log2(probs[atr])
    return -acc

def info_a(dataset, part_datasets):
    acc = 0.0
    for dataset_v in part_datasets:
        acc += ((len(dataset)/len(dataset_v))*info(dataset_v))
    return acc


def choose_attribute(dataset, attributes):  # Information Gain (ID3)
    max_gain = 0.0
    max_part_datasets = []
    max_attr = ''
    info_d = info(dataset)
    for attr in attributes.keys():
        if attributes(attr) == 'numerical':
            dataset = dataset.sort_values(attr)
            values = list(dataset[attr])
            classes = list(dataset['y'])
            possible_split_points = []
            for i in range(len(values)-1):
                if classes[i] != classes[i+1]:
                    possible_split_points.append((values[i]+values[i+1])/2.0)
            
            for sp in possible_split_points:
                part_datasets = []
                part_datasets.append(dataset[dataset[attr] <= sp])
                part_datasets.append(dataset[dataset[attr] > sp])
            gain = info_d - info_a(dataset,part_datasets)
            if gain > max_gain:
                max_gain = gain
                max_part_datasets = part_datasets
                max_attr = attr

                    
        else: #categorical or binary
            part_datasets = []
            for v in set(dataset[attr]):
                part_datasets.append(dataset[dataset.A.isin([v])])
            gain = info_d - info_a(dataset,part_datasets)
            if gain > max_gain:
                max_gain = gain
                max_part_datasets = part_datasets
                max_attr = attr
    
    return max_attr,max_part_datasets


def gen_random_tree(dataset, attributes):
    """
    dataset: pandas dataframe  
    attributes: dict {attribute: type} type = numerical, categorical, binary
    """
    N = Node()
    if len(set(dataset['y'])) == 1:  # all examples have the same attribute
        N.y = dataset['y'][0]
        return N
    elif len(attributes.keys()) == 0:  # attributes list is empty
        N.y = dataset['y'].value_counts().idxmax()
        return N
    else:
        A, part_datasets = choose_attribute(dataset, attributes)
        N.y = A
        next_attributes = attributes.copy()
        del next_attributes[A]
        for dataset_v in part_datasets:
            if len(dataset_v) == 0:
                N.y = dataset['y'].value_counts().idxmax()
                return N
            else:
                child_n = gen_random_tree(dataset_v, next_attributes)
                N.children.append(child_n)
        return N


class RandomTree:

    def __init__(self, dataset, attributes):
        self.random_tree = gen_random_tree(dataset, attributes)


class Node:

    def __init__(self):
        self.y = None
        self.children = []