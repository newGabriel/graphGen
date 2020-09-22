# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import igraph as ig
import cv2 as cv


"""sKnnGraph(X, n)

	Retorna um grafo(igraph) gerado pelo Knn simétrico.

	O grafo gerado contem len(X) vértices e as arestas são a maximização
	das $n menores distâncias entre cada um dos vértices.
	O grafo é um objeto da classe igraph
	
	@param X: Matriz numpy onde cada linha contem as características de
			  um objeto.
	
	@param n: Número de arestas (mínimo) para ser considerado para maximização.
	
	@return: grafo igraph.
"""
def sKnnGraph(X, n):
  m = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    la = []
    ld = []
    for j in range(len(X)):
      if i == j:
        continue
      elif len(la)<n:
        la.append(j)
        ld.append(cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA))
      else:
        if cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA)<max(ld):
          del la[ld.index(max(ld))]
          ld.remove(max(ld))
          la.append(j)
          ld.append(cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA))
    for j in la:
      m[i, j] = 1
  m = np.fmax(m, m.T)
  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
  return g


"""mKnnGraph(X, n)

	Retorna um grafo(igraph) gerado pelo Knn mutuo.
	
	O grafo gerado contem len(X) vértices e as arestas são a minimização
	das $n menores distâncias entre cada um dos vértices.
	O grafo é um objeto da classe igraph

	@param X: Matriz numpy onde cada linha contem as características de
			  um objeto.
	
	@param n: Número de arestas (máximo) para ser considerado para minimização. 
	
	@return: grafo igraph.
	
"""
def mKnnGraph(X, n):
  m = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    la = []
    ld = []
    for j in range(len(X)):
      if i == j:
        continue
      elif len(la)<n:
        la.append(j)
        ld.append(cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA))
      else:
        if cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA)<max(ld):
          del la[ld.index(max(ld))]
          ld.remove(max(ld))
          la.append(j)
          ld.append(cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA))
    for j in la:
      m[i, j] = 1
  m = np.fmin(m, m.T)
  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
  return g


"""eNGraph(X, e)

	Retorna um grafo(igraph) gerado por vizinhança e.
	
	O grafo gerado contem len(X) vértices e as arestas conectam todos
	vértices a uma distancia menor ou igual a $e.
	O grafo é um objeto da classe igraph
	
	@param X: Matriz numpy onde cada linha contem as características de
			  um objeto.
	
	@param e: Distancia do raio de conecção. 
	
	@return: grafo igraph.
	
"""
def eNGraph(X, e):
  m = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    for j in range(len(X)):
      if i == j:
        continue
      elif cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA)<=e:
        m[i,j]=1
  m = np.fmax(m, m.T)
  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
  return g


"""eSKnnGraph(X, n, e)
	
	Retorna um grafo (igraph) gerado pela maximização da vizinhança e
	pelo SKnn
	
	@param X: Matriz numpy onde cada linha contem as características de
			  um objeto.
	
	@param n: Número de arestas para ser considerado no SKnn.
	
	@param e: Distancia do raio de conecção.
	
	@return: grafo igraph.

"""
def eSKnnGraph(X, n, e):
  knn = sKnnGraph(X, n)
  eN = eNGraph(X, e)
  m = np.maximum(list(knn.get_adjacency()),list(eN.get_adjacency()))
  return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


"""eMKnnGraph(X, n, e)
	
	Retorna um grafo (igraph) gerado pela maximização da vizinhança e
	pelo MKnn
	
	@param X: Matriz numpy onde cada linha contem as características de
			  um objeto.
	
	@param n: Número de arestas para ser considerado no MKnn.
	
	@param e: Distancia do raio de conecção.
	
	@return: grafo igraph.

"""
def eMKnnGraph(X, n, e):
  knn = mKnnGraph(X, n)
  eN = eNGraph(X, e)
  m = np.maximum(list(knn.get_adjacency()),list(eN.get_adjacency()))
  return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def pureza(c,y):
  t = 0
  for i in c:
    p = np.zeros(y.max())
    for j in i:
      p[y[j]-1] += 1
    t += p.max()
  pr = t/float(len(y))
  return pr


def colocacao(c, y):
  t = 0
  y1 = np.array(range(len(y)))
  y1 = y1.reshape((41,int(len(y)/41)))
  for i in y1:
    p = np.zeros(len(c))
    for j in i:
      for k in range(len(c)):
        if j in c[k]:
          p[k] += 1
    t += p.max()
  pr = t/float(len(y))
  return pr
