# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist

# ------------------------ 

def normalisation(df):
    normalized_df = (df-df.min())/(df.max()-df.min())
    return pd.DataFrame(normalized_df, columns=df.columns)
    
import math

def dist_euclidienne(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))
    
def centroide(vects):
    return np.mean(vects, axis=0)

def dist_centroides(vects1, vects2):
    return dist_euclidienne(centroide(vects1), centroide(vects2))
    
    
def initialise_CHA(df):
    return {i: [i] for i in range(len(df))}
    
def fusionne(df, partition, verbose=False):
    dist_min = 10e9999
    k1_pp = -1
    k2_pp = -1
    for k1, v1 in partition.items():
        for k2, v2 in partition.items():
            if k1 == k2:
                continue
            dist = dist_centroides(df.iloc[v1], df.iloc[v2])
            if dist < dist_min:
                dist_min = dist
                k1_pp = k1
                k2_pp = k2
    P1 = partition.copy()
    if(k1_pp!= -1):
        del P1[k1_pp]
        del P1[k2_pp]
        P1[max(partition.keys())+1] = [*partition[k1_pp], *partition[k2_pp]]
        if verbose:
            print(f'Distance mininimale trouvée entre  [{k1_pp}, {k2_pp}]  =  {dist_min}')
    
    return P1, k1_pp, k2_pp, dist_min
    
    
    
def CHA_centroid(df):
    partition = initialise_CHA(df)
    res = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne(df, partition)
        tmp = [k1, k2, dist, len(partition[max(partition.keys())])]
        res.append(tmp)
    return res[:len(df)-1]
    
    


def CHA_centroid(df, verbose=False, dendrogramme=False):
    partition = initialise_CHA(df)
    res = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne(df, partition, verbose=verbose)
        result = [k1, k2, dist, len(partition[max(partition.keys())])]
        res.append(result)
    res = res[:len(df)-1]
    
    if dendrogramme:
        
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(res, leaf_font_size=24.)  # taille des caractères de l'axe des

        # Affichage du résultat obtenu:
        plt.show()
    
    return res
    
    
def dist_linkage(linkage, vects1, vects2):
    #on recupere la distance euclidienne la plus grande entre 2 points parmi tous les points du DF
    
    res = cdist(vects1, vects2, 'euclidean')
    if linkage == 'complete':
        return np.max(res)
    if linkage == 'simple':
        return np.min(res)
    if linkage == 'average':
        return np.mean(res)
    

def fusionne_linkage(df, linkage, partition, verbose=False):
    dist_min = 10e9999
    k1_pp = -1
    k2_pp = -1
    for k1, v1 in partition.items():
        for k2, v2 in partition.items():
            if k1 == k2:
                continue
            dist = dist_linkage(linkage, df.iloc[v1], df.iloc[v2])
            if dist < dist_min:
                dist_min = dist
                k1_pp = k1
                k2_pp = k2
    P1 = partition.copy()
    if(k1_pp!= -1):
        del P1[k1_pp]
        del P1[k2_pp]
        P1[max(partition.keys())+1] = [*partition[k1_pp], *partition[k2_pp]]
        if verbose:
            print(f'Distance mininimale trouvée entre  [{k1_pp}, {k2_pp}]  =  {dist_min}')
    
    return P1, k1_pp, k2_pp, dist_min


def CHA_linkage(df, linkage, verbose=False, dendrogramme=False):
    partition = initialise_CHA(df)
    res = []
    for _ in range(len(df)):
        partition, k1, k2, dist = fusionne_linkage(df, linkage, partition, verbose=verbose)
        result = [k1, k2, dist, len(partition[max(partition.keys())])]
        res.append(result)
    res = res[:len(df)-1]
    
    if dendrogramme:
        
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(res, leaf_font_size=24.)  # taille des caractères de l'axe des

        # Affichage du résultat obtenu:
        plt.show()
    
    return res
    
def clustering_hierarchique_complete(df, verbose=False, dendrogramme=False):
    return CHA_linkage(df, 'complete', verbose, dendrogramme)
    
def clustering_hierarchique_simple(df, verbose=False, dendrogramme=False):
    return CHA_linkage(df, 'simple', verbose, dendrogramme)
    
def clustering_hierarchique_average(df, verbose=False, dendrogramme=False):
    return CHA_linkage(df, 'average', verbose, dendrogramme)
    
    
def CHA(DF,linkage='centroid', verbose=False, dendrogramme=False):

    if linkage == 'centroid':
        return CHA_centroid(DF, verbose,dendrogramme)
    if linkage == 'complete':
        return clustering_hierarchique_complete(DF, verbose, dendrogramme)
    if linkage == 'simple':
        return clustering_hierarchique_simple(DF, verbose, dendrogramme)
    if linkage == 'average':
        return clustering_hierarchique_average(DF, verbose, dendrogramme)
        
        
def inertie_cluster(Ens):
    return np.sum(dist_euclidienne(Ens, centroide(Ens))**2)
    
import random
def init_kmeans(K,Ens):
    return np.array(random.sample(list(np.array(Ens)), K))
    
from scipy.spatial.distance import cdist
def plus_proche(Exe,Centres):
    return np.argmin(cdist(np.array(Exe).reshape(1, -1), Centres), axis=1)[0]
    
    
def affecte_cluster(Base,Centres):
    indpp = np.argmin(cdist(np.array(Base), Centres), axis=1)
    res = {}
    for c in range(len(Centres)):
        res[c] = []
        for i in range(len(indpp)):
            if c == indpp[i]:
                res[c].append(i) 
    return res

def nouveaux_centroides(Base,U):
    B = np.array(Base)
    arr = []
    for i in U.values():
        arr.append([np.mean(B[i], axis = 0)][0])
    return np.array(arr)  
    
    
def inertie_globale(Base, U):
     return np.sum([inertie_cluster(np.array(Base)[ui]) for ui in U.values()])
     
     
     
def kmoyennes(K, Base, epsilon, iter_max):
    ig = 0
    U = {}
    C = init_kmeans(K, Base)
    for i in range(iter_max):
        U = affecte_cluster(Base, C)
        ig_ = inertie_globale(Base, U)
        print(f'iteration {i} Inertie : {ig_:.4f} Difference: {np.abs(ig_-ig):.4f}')
        if np.abs(ig_-ig) < epsilon:
            break
        ig = ig_
        C = nouveaux_centroides(Base, U)
    return C, U
    
    
def affiche_resultat(Base,Centres,Affect):
    B = np.array(Base)
    plt.scatter(B[Affect[0]][:,0], B[Affect[0]][:,1], c='green')
    plt.scatter(B[Affect[1]][:,0], B[Affect[1]][:,1], c='blue')
    plt.scatter(B[Affect[2]][:,0], B[Affect[2]][:,1], c='y')
    plt.scatter(Centres[:,0], Centres[:,1], c='r', marker='x')
    
def affiche_resultat_multi(Base, Centres, Affect):
	B = np.array(Base)
	plt.title("Clusters")
	plt.scatter(B[Affect[0]][:,0], B[Affect[0]][:,1], c='green')
	plt.scatter(B[Affect[1]][:,0], B[Affect[1]][:,1], c='blue')
	plt.scatter(B[Affect[2]][:,0], B[Affect[2]][:,1], c='yellow')
	plt.scatter(B[Affect[3]][:,0], B[Affect[3]][:,1], c='red')
	plt.scatter(B[Affect[4]][:,0], B[Affect[4]][:,1], c='black')
	plt.scatter(B[Affect[5]][:,0], B[Affect[5]][:,1], c='pink')
	plt.scatter(B[Affect[6]][:,0], B[Affect[6]][:,1], c='purple')
	plt.scatter(B[Affect[7]][:,0], B[Affect[7]][:,1], c='grey')
	plt.scatter(B[Affect[8]][:,0], B[Affect[8]][:,1], c='olive')
	plt.scatter(B[Affect[9]][:,0], B[Affect[9]][:,1], c='brown')
	plt.scatter(B[Affect[10]][:,0], B[Affect[10]][:,1], c='y')
	plt.scatter(Centres[:,0], Centres[:,1], c='black', marker='x')
    
def affiche_resultat_multi_3d(Base, Centres, Affect):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	plt.title("Clusters en 3D")
	B = np.array(Base)
	plt.scatter(B[Affect[0]][:,0], B[Affect[0]][:,1], B[Affect[0]][:,2], c='green')
	plt.scatter(B[Affect[1]][:,0], B[Affect[1]][:,1], B[Affect[1]][:,2], c='blue')
	plt.scatter(B[Affect[2]][:,0], B[Affect[2]][:,1], B[Affect[2]][:,2], c='yellow')
	plt.scatter(B[Affect[3]][:,0], B[Affect[3]][:,1], B[Affect[3]][:,2], c='red')
	plt.scatter(B[Affect[4]][:,0], B[Affect[4]][:,1], B[Affect[4]][:,2], c='black')
	plt.scatter(B[Affect[5]][:,0], B[Affect[5]][:,1], B[Affect[5]][:,2], c='pink')
	plt.scatter(B[Affect[6]][:,0], B[Affect[6]][:,1], B[Affect[6]][:,2], c='purple')
	plt.scatter(B[Affect[7]][:,0], B[Affect[7]][:,1], B[Affect[7]][:,2], c='grey')
	plt.scatter(B[Affect[8]][:,0], B[Affect[8]][:,1], B[Affect[8]][:,2], c='olive')
	plt.scatter(B[Affect[9]][:,0], B[Affect[9]][:,1], B[Affect[9]][:,2], c='brown')
	plt.scatter(B[Affect[10]][:,0], B[Affect[10]][:,1], B[Affect[10]][:,2], c='y')
	plt.scatter(Centres[:,0], Centres[:,1], Centres[:,2], c='black', marker='x')
    
def distance_max_cluster(cluster):
    return np.max(cdist(cluster, cluster))
    
    
def co_dist(X, U):
    d = 0
    X = np.array(X)
    for idxs in U.values():
        d += distance_max_cluster(X[idxs])
    return d
    
    
def index_dunn(X, U):
    return co_dist(X, U) / inertie_globale(X, U)    
    


