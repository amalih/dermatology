import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus as pdot
import graphviz
import collections

from subprocess import call
import matplotlib.pyplot as plt

def main():
    # Import data and create decision tree
    #dermatology=datasets.load_dermatology()
    dermData = pd.read_csv('dermatologyWOnan.data')
    dermData.columns = ['erythema', 'scaling', 'definite borders', 'itching', 'koebner phenomenon', 'polygonal papules',
                        'follicular papules', 'oral mucosal involvement', 'knee and elbow involvement', 'scalp involvement',
                        'family history', 'melanin incontinence', 'eosinophils in the infiltrate', 'PNL infiltrate',
                        'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
                        'clubbing of the rete ridges', 'elongation of the rete ridges', 'thinning of the suprapapillary epidermis',
                        'spongiform pustule', 'munro microabcess', 'focal hypergranulosis', 'disappearance of the granular layer',
                        'vacuolisation and damage of basal layer', 'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug',
                        'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate', 'band-like infiltrate', 'age', 'label']
    dermData.to_csv('dermData.csv')

    target = dermData['label']  #provided your csv has header row, and the label column is named "Label"

    #select all but the last column as data
    data = dermData.iloc[:,:-1]
    data_feature_names = ['erythema', 'scaling', 'definite borders', 'itching', 'koebner phenomenon', 'polygonal papules',
                        'follicular papules', 'oral mucosal involvement', 'knee and elbow involvement', 'scalp involvement',
                        'family history', 'melanin incontinence', 'eosinophils in the infiltrate', 'PNL infiltrate',
                        'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
                        'clubbing of the rete ridges', 'elongation of the rete ridges', 'thinning of the suprapapillary epidermis',
                        'spongiform pustule', 'munro microabcess', 'focal hypergranulosis', 'disappearance of the granular layer',
                        'vacuolisation and damage of basal layer', 'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug',
                        'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate', 'band-like infiltrate', 'age']

    #df=pd.DataFrame(dermatology.data, columns=dermatology.names)

    dtree=DecisionTreeClassifier()
    dtree.fit(data,target)

    # Plot decision tree
    #dot_data = StringIO()
    dot_data = export_graphviz(dtree, out_file=None,feature_names=data_feature_names,
                    filled=True, rounded=True, precision = 2,
                    special_characters=True)

    #export_graphviz(dtree, out_file='tree_test.dot', feature_names = iris.feature_names,
    #            class_names = iris.target_names,
    #            rounded = True,  precision = 2, filled = True)

    graph = pdot.graph_from_dot_data(dot_data)
    #colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    #graphviz.render('dot', 'png', 'test-output/holy-grenade.gv')
    graph.write_png('tree_test.png')
    #Image(graph[0].create_png())
    #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    #plt.figure(figsize = (14, 18))
    #lt.imshow(plt.imread('tree_test.png'))
    #plt.axis('off');
    #plt.show();


    #Image(filename = 'tree_test.png')


if __name__== "__main__":
  main()
