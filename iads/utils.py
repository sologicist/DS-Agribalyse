# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

# genere_dataset_uniform:
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    return (np.random.uniform(binf, bsup, (2*n, p)), np.array([-1 for i in range(n)] + [+1 for i in range(n)]))
    
def analyse_perfs(perf):
    return np.mean(perf), np.var(perf)
    
# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """

    return (np.vstack((np.random.multivariate_normal(negative_center, negative_sigma, nb_points),
                       np.random.multivariate_normal(positive_center, positive_sigma, nb_points))), 
            np.array([-1 for i in range(nb_points)]+ [+1 for i in range(nb_points)]))
    
# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
   #TODO: A Compléter  
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:,0], data_negatifs[:,1], marker='o', color='red')
    plt.scatter(data_positifs[:,0], data_positifs[:,1], marker='x', color='blue')
# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
    
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    data1 = [0, 0] + var * np.random.standard_normal(size=(n, 2))
    data2 = [1, 1] + var * np.random.standard_normal(size=(n, 2))
    data3 = [0, 1] + var * np.random.standard_normal(size=(n, 2))
    data4 = np.random.normal([1, 0], var, size=(n,2))

    return (np.vstack((data1, data2, data3, data4)),
           np.hstack((np.ones(2*n, dtype=int), -1*np.ones(2*n, dtype=int))))

def crossval(X, Y, n, i):
    start, end = i*int(len(Y)/n), (i+1)*int(len(Y)/n)
    Xtrain = np.delete(X, np.s_[start:end], axis=0)
    Ytrain = np.delete(Y, np.s_[start:end], axis=0)
    Xtest = X[start:end]
    Ytest = Y[start:end]
    return Xtrain, Ytrain, Xtest, Ytest

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n, i):
    Xtrains, Ytrains, Xtests, Ytests = [], [], [], []
    for y in np.unique(Y):
        Xtrainy, Ytrainy, Xtesty, Ytesty = crossval(X[Y==y], Y[Y==y], n, i)
        Xtrains.append(Xtrainy)
        Ytrains.append(Ytrainy)
        Xtests.append(Xtesty)
        Ytests.append(Ytesty)
    return (np.concatenate(Xtrains), np.concatenate(Ytrains),
            np.concatenate(Xtests), np.concatenate(Ytests))
