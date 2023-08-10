import torch
import numpy as np
from torch_geometric.data import Data


def tGraphFromSeries(serie: np.ndarray, n=None, y=None):
    """

    Retorna um tensor representando o grafo gerado pelo algoritimo de grafo de visibilidade para series numericas

    @param serie: lista com a serie numerica a ser transformada em grafo

    @param n: numero maximo de vizinhos a ser analizado durante a construção do grafo

    @param y: target

    @return: tensor grafo
    """

    enum_serie = list(enumerate(serie))

    x = np.array(serie).reshape(len(serie), 1)
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []

    if n is None:
        n = len(serie)

    for i in range(len(serie)):
        for j in range(i+1, min(i + n, len(serie))):

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
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    if y is not None:
        target = torch.tensor(y, dtype=torch.long)
    else:
        target = None

    g = Data(x=x, edge_index=edge_index, Y=target)

    return g
