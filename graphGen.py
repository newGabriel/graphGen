# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import igraph as ig
import cv2 as cv


def knnGraph(X, n, t='S', pon=0):
  """
  
	Retorna um grafo(igraph) gerado pelo Knn.

	O grafo gerado contem len(X) vértices e as arestas são a maximização das n 
	menores distâncias entre cada um dos vértices. O grafo é um objeto da classe igraph

	@param X Matriz numpy onde cada linha contem as características de um objeto.

	@param n Número de arestas para ser considerado para maximização/minimização.

	@param t Modo da construção do grafo, 'M' para knn mutuo, 'S' para knn Simetrico(padrão).

	@param pon Modo de contrução do grafo, 0 para não ponderado(padrão), 1 para ponderado.

	@return Grafo igraph.
  """

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

  if t=='S':
    m = np.fmax(m, m.T)
  elif t=='M':
    m = np.fmin(m, m.T)

  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
  if pon==1:
    g.es["weight"] = 1.0
    for i in g.get_edgelist():
      g[i[0],i[1]] = cv.compareHist(X[i[0]],X[i[1]],cv.HISTCMP_BHATTACHARYYA)

  return g


def kmnGraph(X, n, pon=0):
  """

	Retorna um grafo(igraph) gerado pelo Kmn.

	O grafo gerado contem len(X) vértices e as arestas são a maximização
	das n maiores distâncias entre cada um dos vértices.
	O grafo é um objeto da classe igraph

	@param X Matriz numpy onde cada linha contem as características de um objeto.

	@param n Número de arestas para ser considerado para maximização.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

	@return Grafo igraph.
  """
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
        if cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA)>min(ld):
          del la[ld.index(min(ld))]
          ld.remove(min(ld))
          la.append(j)
          ld.append(cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA))
    for j in la:
      m[i, j] = 1
      
  m = np.fmax(m, m.T)

  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
  if pon==1:
    g.es["weight"] = 1.0
    for i in g.get_edgelist():
      g[i[0],i[1]] = cv.compareHist(X[i[0]],X[i[1]],cv.HISTCMP_BHATTACHARYYA)

  return g


def sKnnGraph(X, n,pon=0):
  """

	Retorna um grafo(igraph) gerado pelo Knn simétrico.

	O grafo gerado contem len(X) vértices e as arestas são a maximização
	das $n menores distâncias entre cada um dos vértices.
	O grafo é um objeto da classe igraph
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas (mínimo) para ser considerado para maximização.
	
	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.
	
	@return Grafo igraph.
  """
  g = knnGraph(X,n,'S',pon)
  return g


def mKnnGraph(X, n, pon=0):
  """

	Retorna um grafo(igraph) gerado pelo Knn mutuo.
	
	O grafo gerado contem len(X) vértices e as arestas são a minimização
	das $n menores distâncias entre cada um dos vértices.
	O grafo é um objeto da classe igraph

	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas (máximo) para ser considerado para minimização. 

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

	@return Grafo igraph.
	
  """
  g = knnGraph(X,n,'M',pon)
  return g


def eNGraph(X, e,pon=0):
  """

	Retorna um grafo(igraph) gerado por vizinhança e.
	
	O grafo gerado contem len(X) vértices e as arestas conectam todos
	vértices a uma distancia menor ou igual a $e.
	O grafo é um objeto da classe igraph
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param e Distancia do raio de conecção. 
	
	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.
	
	@return Grafo igraph.
	
  """
  
  m = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    for j in range(len(X)):
      if i == j:
        continue
      elif cv.compareHist(X[i],X[j],cv.HISTCMP_BHATTACHARYYA)<=e:
        m[i,j]=1
  m = np.fmax(m, m.T)
  g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)  
  if pon==1:
    g.es["weight"] = 1.0
    for i in g.get_edgelist():
      g[i[0],i[1]] = cv.compareHist(X[i[0]],X[i[1]],cv.HISTCMP_BHATTACHARYYA)
  return g


def eSKnnGraph(X, n, e, pon=0):
  """
	
	Retorna um grafo (igraph) gerado pela maximização da vizinhança e
	pelo SKnn
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas para ser considerado no SKnn.
	
	@param e Distancia do raio de conecção.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.
	
	@return Grafo igraph.

  """
  
  knn = sKnnGraph(X, n, pon)
  eN = eNGraph(X, e, pon)
  m = np.maximum(list(knn.get_adjacency()),list(eN.get_adjacency()))
  return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eMKnnGraph(X, n, e, pon=0):
  """
	
	Retorna um grafo (igraph) gerado pela maximização da vizinhança e
	pelo MKnn
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas para ser considerado no MKnn.
	
	@param e Distancia do raio de conecção.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.
	
	@return Grafo igraph.

  """

  knn = mKnnGraph(X, n,pon)
  eN = eNGraph(X, e,pon)
  m = np.maximum(list(knn.get_adjacency()),list(eN.get_adjacency()))
  return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eSKnnMST(X, n, e, pon=0):
  '''
	
	Retorna um grafo (igraph) gerado pela união da arvore geradora minima
	com a maximização da vizinhaça e pelo SKnn
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas para ser considerado no MKnn.
	
	@param e Distancia do raio de conecção.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

	@return Grafo igraph.
	
  '''

  MST = knnGraph(X,len(X),pon=1).spanning_tree()
  eknn = eSKnnGraph(X,n,e,pon)
  return eknn.union(MST)


def eMKnnMST(X, n, e, pon=0):
  '''
	
	Retorna um grafo (igraph) gerado pela união da arvore geradora minima
	com a maximização da vizinhaça e pelo MKnn
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param n Número de arestas para ser considerado no MKnn.
	
	@param e Distancia do raio de conecção.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

	@return Grafo igraph.
	
  '''

  MST = knnGraph(X,len(X),pon=1).spanning_tree()
  eknn = eMKnnGraph(X,n,e)
  return eknn.union(MST)


def kmnnGraph(X,k,k1,t='S',pon=0):
  '''
	
	Retorna um grafo (igraph) gerado pela união de knn + k'

	O grafo gerado contem len(X) vértices e as arestas são a maximização
	das $k menores distâncias e das $k1 maiores distancias entre cada um dos vértices.
	O grafo é um objeto da classe igraph
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param k Número de arestas para ser considerado no Knn.
	
	@param k1 Número de vertices mais distantes para serem ligados a cada vertice.
	
	@param t Modo da construção do grafo, 'M' para knn mutuo, 'S' para knn Simetrico(padrão).

	@param pon Modo de contrução do grafo, 0 para não ponderado(padrão), 1 para ponderado.

	@return Grafo igraph.
'''

  g = knnGraph(X,k,t,pon)
  return g.union(kmnGraph(X,k1,pon))


def eKmnnGraph(X,k,k1,e,pon=0):
  '''

	Retorna um grafo (igraph) gerado pela união da arvore geradora minima
	com a maximização da vizinhaça e pelo Knn+k'
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param k Número de arestas para ser considerado no Knn.
	
	@param k1 Número de vertices mais distantes para serem ligados a cada vertice.
	
	@param e Distancia do raio de conecção.
	
	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

	@return Grafo igraph.
  '''

  knn = kmnnGraph(X, k, k1)
  eN = eNGraph(X, e)
  m = np.maximum(list(knn.get_adjacency()),list(eN.get_adjacency()))
  return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eKmnnMST(X,k,k1,e,pon=0):
  '''
	
	Retorna um grafo (igraph) gerado pela união da arvore geradora minima
	com a maximização da vizinhaça e pelo Knn+k'
	
	@param X Matriz numpy onde cada linha contem as características de um objeto.
	
	@param k Número de arestas para ser considerado no Knn.
	
	@param k1 Número de vertices mais distantes para serem ligados a cada vertice.
	
	@param e Distancia do raio de conecção.

	@param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.
	
	@return Grafo igraph.
  '''

  MST = knnGraph(X,len(X),pon=1).spanning_tree()
  eknn = eKmnnGraph(X,k,k1,e)
  return eknn.union(MST)


def pureza(c,y):
  """
	
	Retorna um número real com valor da pureza de uma clusterização.
	
	@param c Clusterização ig.Graph.communit_...
	
	@param y Matriz onde cada linha é o label do objeto (iniciando em 1).
	
	@return Valor float.

  """

  t = 0
  for i in c:
    p = np.zeros(y.max())
    for j in i:
      p[y[j]-1] += 1
    t += p.max()
  pr = t/float(len(y))
  return pr


def colocacao(c, y):
  """
	
	Retorna um número real com valor da colocação de uma clusterização.
	
	
	@param c Clusterização ig.Graph.communit_...
	@param y Matriz onde cada linha é o label do objeto (iniciando em 1).
	
	@return Valor float.

  """

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
