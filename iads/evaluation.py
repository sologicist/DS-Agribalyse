# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 

def crossval(X, Y, n_iterations, iteration):
    #############
    np.random.seed(42)
    index = np.random.permutation(len(X)) # mélange des index
    size=len(X)//n_iterations
    Xm = X[index]
    Ym = Y[index]
    
    n_iterx = [Xm[0+size*i : size*(i+1)] for i in range(n_iterations)]
    n_itery = [Ym[0+size*i : size*(i+1)] for i in range(n_iterations)]
    
    appx = n_iterx.copy()
    appx.pop(iteration)
    appx = np.concatenate(appx)

    appy = n_itery.copy()
    appy.pop(iteration)
    appy = np.concatenate(appy)
    
    #############    
    return appx, appy, n_iterx[iteration], n_itery[iteration]

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    Xtrains, Ytrains, Xtests, Ytests = [], [], [], []
    for y in np.unique(Y):
        Xtrainy, Ytrainy, Xtesty, Ytesty = crossval(X[Y==y], Y[Y==y], n_iterations, iteration)
        Xtrains.append(Xtrainy)
        Ytrains.append(Ytrainy)
        Xtests.append(Xtesty)
        Ytests.append(Ytesty)
    return (np.concatenate(Xtrains), np.concatenate(Ytrains),
            np.concatenate(Xtests), np.concatenate(Ytests))


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.sqrt(np.var(L)) 

