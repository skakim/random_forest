import csv
import pandas as pd
from math import log2


def info(dataset):
    probs = dict(dataset.y.value_counts(normalize=True))
    acc = 0.0
    for atr in probs.keys():
        acc += probs[atr] * log2(probs[atr])
    return -acc


def info_a(dataset, part_datasets):
    acc = 0.0
    for dataset_v in part_datasets:
        acc += ((len(dataset_v) / len(dataset)) * info(dataset_v))
    return acc


def choose_attribute(dataset, attributes):  # Information Gain (ID3)
    max_gain = 0.0
    max_part_datasets = []
    max_attr = ''
    max_sp = None
    info_d = info(dataset)
    attr_values = []
    for attr in attributes.keys():
        if attributes[attr][0] == 'numerical':
            dataset = dataset.sort_values(attr)
            values = list(map(float, dataset[attr]))
            classes = list(dataset['y'])
            possible_split_points = []
            for i in range(len(values) - 1):
                if classes[i] != classes[i + 1]:
                    possible_split_points.append(
                        (values[i] + values[i + 1]) / 2.0)
            for sp in possible_split_points:
                part_datasets = []
                part_datasets.append(dataset[dataset[attr] <= sp])
                part_datasets.append(dataset[dataset[attr] > sp])
                gain = info_d - info_a(dataset, part_datasets)
                if gain > max_gain:
                    max_gain = gain
                    max_part_datasets = part_datasets
                    max_attr = attr
                    max_sp = sp
                    attr_values = ["Yes", "No"]
        else:  # categorical or binary
            part_datasets = []
            temp = []
            for v in attributes[attr][1]:
                part_datasets.append(dataset[dataset[attr].isin([v])])
                temp.append(v)
            gain = info_d - info_a(dataset, part_datasets)
            if gain > max_gain:
                max_gain = gain
                max_part_datasets = part_datasets
                max_attr = attr
                max_sp = None
                attr_values = temp
    return max_attr, max_sp, max_part_datasets, attr_values


def gen_random_tree(dataset, attributes, depth_limit, depth=1):
    """
    dataset: pandas dataframe  
    attributes: dict {attribute: [type,possible_values]} type = numerical, categorical, binary
    """
    N = Node()
    N.info = info(dataset)
    N.instances = len(dataset)
    if len(set(dataset['y'])) == 1:  # all examples have the same class
        N.y = list(dataset['y'])[0]
        N.attr = 'y'
        return N
    elif len(attributes.keys()) == 0:  # attributes list is empty
        N.y = dataset['y'].value_counts().idxmax()
        N.attr = 'y'
        return N
    elif (depth_limit != None) and depth == depth_limit:  # depth_limit reached
        N.y = dataset['y'].value_counts().idxmax()
        N.attr = 'y'
        return N
    else:
        A, max_sp, part_datasets, attr_values = choose_attribute(dataset, attributes)
        if A == '':
            N.y = dataset['y'].value_counts().idxmax()
            N.attr = 'y'
            return N 
        N.sp = max_sp
        N.attr = A
        next_attributes = attributes.copy()
        if depth_limit == None or attributes[A][0] != 'numerical':
            del next_attributes[A]
        i = 0
        for dataset_v in part_datasets:
            if len(dataset_v) == 0:
                child_n = Node()
                child_n.attr_value = attr_values[i]
                child_n.y = dataset['y'].value_counts().idxmax()
                child_n.attr = 'y'
                N.children.append(child_n)
            else:
                child_n = gen_random_tree(dataset_v, next_attributes, depth_limit, depth=depth+1)
                if max_sp != None:
                    N.sp_side = ("<=" if i == 0 else ">")
                    child_n.attr_value = ("Yes" if i == 0 else "No")
                else:
                    child_n.attr_value = dataset_v[A].value_counts().idxmax()
                N.children.append(child_n)
            i += 1
        return N


class RandomTree:

    def __init__(self, dataset, attributes, depth_limit=None):
        self.random_tree = gen_random_tree(dataset, attributes, depth_limit=depth_limit)
        self.attributes = attributes

    def print_tree(self):  # TODO: make print_tree prettier
        queue = [[self.random_tree, 0]]
        while queue:
            q = queue.pop()
            N = q[0]
            tabs = q[1]
            print("\t\t" * tabs, N)
            for C in N.children:
                queue.append([C, tabs + 1])

    def classify(self, instance, stdout=False):
        N = self.random_tree
        while(not(N.y)):
            prev_N = N
            attr = N.attr
            value = instance[attr]
            typ = self.attributes[attr][0]
            if typ == 'numerical':
                if eval(str(value) + N.sp_side + str(N.sp)):
                    if(stdout):
                        print(attr, str(value) + N.sp_side + str(N.sp), end=' -> ')
                    N = N.children[0]
                else:
                    if(stdout):
                        print(attr, "!(" + str(value) + N.sp_side +
                              str(N.sp) + ")", end=' -> ')
                    N = N.children[1]
            else:
                for C in N.children:
                    if value == C.attr_value:
                        if(stdout):
                            print(str(attr) + " == " + str(C.attr_value), end=' -> ')
                        N = C
                        break
                if prev_N == N:
                    if stdout:
                        print("A tree couldn\'t classify this instance")
                    return None
        if(stdout):
            print("y =",N.y)
        return N.y


class Node:

    def __init__(self):
        self.attr_value = "Root"
        self.attr = None
        self.sp = None
        self.sp_side = None
        self.y = None
        self.children = []
        self.info = 0.0
        self.instances = 0

    def __str__(self):
        return "%s %s %s %s" % (self.attr_value, ("y=" + str(self.y) if self.y else (self.attr + self.sp_side + str(self.sp) if self.sp_side else self.attr)), self.info, self.instances)


if __name__ == "__main__":
    # benchmark
    m = []
    with open('datasets/benchmark/benchmark.csv') as bfile:
        breader = csv.reader(bfile, delimiter=';')
        for row in breader:
            m.append(row)
    attributes = {x: ['categorical'] for x in m[0][:-1]}
    attributes_names = m[0][:-1]
    del m[0]
    dataset = pd.DataFrame(m, columns=attributes_names + ['y'])
    for x in attributes.keys():
        attributes[x].append(dataset[x].unique())
    RT = RandomTree(dataset, attributes)
    RT.print_tree()

    print(RT.classify({'Tempo': 'Ensolarado', 'Temperatura': 'Quente',
                       'Umidade': 'Alta', 'Ventoso': 'Falso'}, stdout=True))
    print(RT.classify({'Tempo': 'Nublado', 'Temperatura': 'Quente',
                       'Umidade': 'Alta', 'Ventoso': 'Falso'}, stdout=True))

    # benchmark numerical
    m = []
    with open('datasets/benchmark/benchmark_numerical.csv') as bfile:
        breader = csv.reader(bfile, delimiter=';')
        for row in breader:
            m.append(row)
    attributes = {x: ['categorical'] for x in m[0][:-1]}
    attributes['Graus'] = ['numerical']
    attributes_names = m[0][:-1]
    del m[0]
    dataset = pd.DataFrame(m, columns=attributes_names + ['y'])
    dataset['Graus'] = dataset['Graus'].astype('float64')
    for x in attributes.keys():
        if x != 'Graus':
            attributes[x].append(dataset[x].unique())
    RT = RandomTree(dataset, attributes)
    RT.print_tree()

    print(RT.classify({'Tempo': 'Ensolarado',
                       'Graus': 29.5, 'Umidade': 'Alta'}, stdout=True))
    print(RT.classify({'Tempo': 'Chuvoso', 'Graus': 12.5,
                       'Umidade': 'Alta'}, stdout=True))
