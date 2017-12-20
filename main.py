import csv
import pandas as pd
from random_tree import RandomTree


# will use to put everything between 0 and 1
def normalize(value, oldmin, oldmax, newmin, newmax):
    newvalue = (((float(value) - oldmin) * (newmax - newmin)) /
                (oldmax - oldmin)) + newmin
    return newvalue


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
        for name in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dim']:
            columns += [name + "_mean", name + "_stderror", name + "_worse"]
        data = pd.read_csv(
            "datasets/breast-cancer-wisconsin/wdbc.data", names=columns)
        del data['id']
        attrs = {x: 'numerical' for x in columns}
        del attrs['y']
        del attrs['id']
        
    return data, attrs


if __name__ == "__main__":
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
