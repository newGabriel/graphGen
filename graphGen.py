# -*- coding: utf-8 -*-


import numpy as np
import igraph as ig
from numpy.linalg import norm


def similarity(x1, x2):
    return norm(x1 - x2)


def knnGraph(X, n: int, t='S', pon=0, dif=similarity, target: bool = False, y=None):
    """

    Retorna um grafo(igraph) gerado pelo Knn.

    O grafo gerado contem len(X) vértices e as arestas são a maximização das n
    menores distâncias entre cada um dos vértices. O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado para maximização/minimização.

    @param t Modo da construção do grafo, 'M' para knn mutuo, 'S' para knn Simetrico(padrão).

    @param pon Modo de contrução do grafo, 0 para não ponderado(padrão), 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    if target and y is None:
        raise Exception('Para criação de grafos com classes é necessario receber uma lista de classes')
    m = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        neighbor_list = []
        distance_list = []
        for j in range(len(X)):
            if not target or y[i] == y[j]:
                if i == j:
                    continue
                elif len(neighbor_list) < n:
                    neighbor_list.append(j)
                    distance_list.append(dif(X[i], X[j]))
                else:
                    if dif(X[i], X[j]) < max(distance_list):
                        del neighbor_list[distance_list.index(max(distance_list))]
                        distance_list.remove(max(distance_list))
                        neighbor_list.append(j)
                        distance_list.append(dif(X[i], X[j]))
        for j in neighbor_list:
            m[i, j] = 1

    if t == 'S':
        m = np.fmax(m, m.T)
    elif t == 'M':
        m = np.fmin(m, m.T)

    g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
    if pon == 1:
        g.es["weight"] = 1.0
        for i in g.get_edgelist():
            g[i[0], i[1]] = dif(X[i[0]], X[i[1]])

    if target:
        g.vs["class"] = y

    return g


def kmnGraph(X, n: int, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo(igraph) gerado pelo Kmn.

    O grafo gerado contem len(X) vértices e as arestas são a maximização
    das n maiores distâncias entre cada um dos vértices.
    O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado para maximização.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.
    """
    if target and y is None:
        raise Exception('Para criação de grafos com classes é necessario receber uma lista de classes')
    m = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        la = []
        ld = []
        for j in range(len(X)):
            if not target or y[i] == y[j]:
                if i == j:
                    continue
                elif len(la) < n:
                    la.append(j)
                    ld.append(dif(X[i], X[j]))
                else:
                    if dif(X[i], X[j]) > min(ld):
                        del la[ld.index(min(ld))]
                        ld.remove(min(ld))
                        la.append(j)
                        ld.append(dif(X[i], X[j]))
        for j in la:
            m[i, j] = 1

    m = np.fmax(m, m.T)

    g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
    if pon == 1:
        g.es["weight"] = 1.0
        for i in g.get_edgelist():
            g[i[0], i[1]] = dif(X[i[0]], X[i[1]])

    if target:
        g.vs["class"] = y

    return g


def eNGraph(X, e: float, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo(igraph) gerado por vizinhança e.

    O grafo gerado contem len(X) vértices e as arestas conectam todos
    vértices a uma distancia menor ou igual a $e.
    O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    if target and y is None:
        raise Exception('Para criação de grafos com classes é necessario receber uma lista de classes')
    m = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if target == 0 or y[i] == y[j]:
                if i == j:
                    continue
                elif dif(X[i], X[j]) <= e:
                    m[i, j] = 1
    m = np.fmax(m, m.T)
    g = ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)
    if pon == 1:
        g.es["weight"] = 1.0
        for i in g.get_edgelist():
            g[i[0], i[1]] = dif(X[i[0]], X[i[1]])

    if target:
        g.vs["class"] = y

    return g


def sKnnGraph(X, n, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo(igraph) gerado pelo Knn simétrico.

    O grafo gerado contem len(X) vértices e as arestas são a maximização
    das $n menores distâncias entre cada um dos vértices.
    O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas (mínimo) para ser considerado para maximização.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.
    """
    g = knnGraph(X, n, 'S', pon, dif, target, y)
    return g


def mKnnGraph(X, n, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo(igraph) gerado pelo Knn mutuo.

    O grafo gerado contem len(X) vértices e as arestas são a minimização
    das $n menores distâncias entre cada um dos vértices.
    O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas (máximo) para ser considerado para minimização.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """
    g = knnGraph(X, n, 'M', pon, dif, target, y)
    return g


def eSKnnGraph(X, n, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela maximização da vizinhança e pelo SKnn

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado no SKnn.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    knn = sKnnGraph(X, n, pon, dif, target, y)
    eN = eNGraph(X, e, pon, dif, target, y)
    m = np.maximum(list(knn.get_adjacency()), list(eN.get_adjacency()))
    return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eMKnnGraph(X, n, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela maximização da vizinhança e pelo MKnn

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado no MKnn.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    knn = mKnnGraph(X, n, pon, dif, target, y)
    eN = eNGraph(X, e, pon, dif, target, y)
    m = np.maximum(list(knn.get_adjacency()), list(eN.get_adjacency()))
    return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eSKnnMST(X, n, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela união da arvore geradora minima com a maximização da vizinhaça e pelo SKnn

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado no MKnn.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    MST = knnGraph(X, len(X), pon=1, dif=dif, target=target, y=y).spanning_tree()
    eknn = eSKnnGraph(X, n, e, pon, dif, target, y)
    return eknn.union(MST)


def eMKnnMST(X, n, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela união da arvore geradora minima com a maximização da vizinhaça e pelo MKnn

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param n Número de arestas para ser considerado no MKnn.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.

    """

    MST = knnGraph(X, len(X), pon=1, dif=dif, target=target, y=y).spanning_tree()
    eknn = eMKnnGraph(X, n, e, pon, dif, target, y)
    return eknn.union(MST)


def kmnnGraph(X, k, k1, t='S', pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela união de knn + k'

    O grafo gerado contem len(X) vértices e as arestas são a maximização
    das $k menores distâncias e das $k1 maiores distancias entre cada um dos vértices.
    O grafo é um objeto da classe igraph

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param k Número de arestas para ser considerado no Knn.

    @param k1 Número de vertices mais distantes para serem ligados a cada vertice.

    @param t Modo da construção do grafo, 'M' para knn mutuo, 'S' para knn Simetrico(padrão).

    @param pon Modo de contrução do grafo, 0 para não ponderado(padrão), 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.
    """

    g = knnGraph(X, k, t, pon, dif, target, y)
    return g.union(kmnGraph(X, k1, pon, dif, target, y))


def eKmnnGraph(X, k, k1, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela união da arvore geradora minima com a maximização da vizinhaça e pelo Knn+k'

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param k Número de arestas para ser considerado no Knn.

    @param k1 Número de vertices mais distantes para serem ligados a cada vertice.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.
    """

    knn = kmnnGraph(X, k, k1, pon=pon, dif=dif, target=target, y=y)
    eN = eNGraph(X, e, dif=dif, target=target, y=y)
    m = np.maximum(list(knn.get_adjacency()), list(eN.get_adjacency()))
    return ig.Graph.Adjacency(m.tolist(), mode=ig.ADJ_MAX)


def eKmnnMST(X, k, k1, e, pon=0, dif=similarity, target=False, y=None):
    """

    Retorna um grafo (igraph) gerado pela união da arvore geradora minima com a maximização da vizinhaça e pelo Knn+k'

    @param X Matriz numpy onde cada linha contem as características de um objeto.

    @param k Número de arestas para ser considerado no Knn.

    @param k1 Número de vertices mais distantes para serem ligados a cada vertice.

    @param e Distancia do raio de conecção.

    @param pon Modo de contrução do grafo, 0 para não ponderado, 1 para ponderado.

    @param dif função usada para comparar a similaridade dos objetos, distancia euclidiana por padrão

    @param target modo de construção do grafo, 0 para nós sem classe, 1 para nós com classe

    @param y lista contendo a classe de cada um dos objetos em X

    @return Grafo igraph.
    """

    MST = knnGraph(X, len(X), pon=1, dif=dif, target=target, y=y).spanning_tree()
    eknn = eKmnnGraph(X, k, k1, e, dif=dif, target=target, y=y)
    return eknn.union(MST)


def graphFromSeries(serie, n=None):
    """

    Retorna um grafo (igraph) gerado pelo algoritimo de grafo de visibilidade para series numericas

    @param serie array contendo os números da serie que será transformada em grafo

    @param n numero maximo de vizinhos a ser analizado durante a construção do grafo

    @return: grafo (igraph)
    """

    g = ig.Graph()
    g.add_vertices(len(serie))
    g.vs['mag'] = serie

    enum_serie = list(enumerate(serie))

    if n is None:
        n = len(serie)

    for i in range(len(serie)):
        for j in range(i, min(i+n, len(serie))):

            ta = i
            ya = serie[i]
            tb = j
            yb = serie[j]

            connected = True

            for tc, yc in enum_serie[ta:tb]:
                if tc != ta and tc != tb and yc > yb + (ya - yb) * ((tb - tc) / (tb - ta)):
                    connected = False
                    break

            if connected:
                g.add_edge(ta, tb)

    return g


def pureza(c, y):
    """

      Retorna um número real com valor da pureza de uma clusterização.

      @param c Clusterização ig.Graph.communit_...

      @param y Matriz onde cada linha é o label do objeto (iniciando em 1).

      @return Valor float.
    """

    t = 0
    for i in c:
        p = {}
        for j in i:
            if y[j] in p:
                p[y[j]] += 1
            else:
                p[y[j]] = 1
        t += max(p.values())
    pr = t / float(len(y))
    return pr


def colocacao(c, y):
    """

      Retorna um número real com valor da colocação de uma clusterização.


      @param c Clusterização ig.Graph.communit_...
      @param y Matriz onde cada linha é o label do objeto (iniciando em 1).

      @return Valor float.
    """
    m = {}
    for p, i in enumerate(y):
        if i in m:
            m[i].append(p)
        else:
            m[i] = [p]
    c1 = np.zeros((len(y))).astype('int')
    for p, i in enumerate(c):
        for j in i:
            c1[j] = p
    return pureza(m.values(), c1)


def media_harmonica(c, y):
    """

    Retorna um número real com valor da media harmonica entre pureza e colocação de uma clusterização.

    @param c Clusterização ig.Graph.communit_...
    @param y Matriz onde cada linha é o label do objeto (iniciando em 1).

    @return Valor float.
    """
    pf = pureza(c, y)
    cf = colocacao(c, y)
    return (2 * cf * pf) / (cf + pf)
