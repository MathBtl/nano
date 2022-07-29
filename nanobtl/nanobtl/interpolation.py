"""
Interpolation algorithm based on:
Falkenberg, Jesper Toft, and Mads Brandbyge. Simple and Efficient Way of Speeding up Transmission Calculations with k -Point Sampling.
Beilstein Journal of Nanotechnology, vol. 6, July 2015, pp. 1603-08. DOI.org (Crossref), https://doi.org/10.3762/bjnano.6.164.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
import heapq

def _find_shortest_paths(graph, start_point):
    """Dijkstra's algorithm"""
    # initialize graphs to track if a point is visited,
    # current calculated distance from start to point,
    # and previous point taken to get to current point
    visited = [[False for col in row] for row in graph]
    distance = [[float('inf') for col in row] for row in graph]
    distance[start_point[0]][start_point[1]] = 0
    prev_point = [[None for col in row] for row in graph]
    n, m = len(graph), len(graph[0])
    number_of_points, visited_count = n * m, 0
    directions = [(0, 1), (1, 0), (1, 1)]
    min_heap = []

    # min_heap item format:
    # (pt's dist from start on this path, pt's row, pt's col)
    heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

    while visited_count < number_of_points:
        current_point = heapq.heappop(min_heap)
        distance_from_start, row, col = current_point
        for direction in directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                dist_to_new_point = distance_from_start + graph[new_row][new_col]
                if dist_to_new_point < distance[new_row][new_col]:
                    distance[new_row][new_col] = dist_to_new_point
                    prev_point[new_row][new_col] = (row, col)
                    heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
        visited[row][col] = True
        visited_count += 1

    return distance, prev_point


def _find_shortest_path(prev_point_graph, end_point):
    """Postprocessing of Dijkstra's algorithm"""
    shortest_path = []
    current_point = end_point
    while current_point is not None:
        shortest_path.append(current_point)
        current_point = prev_point_graph[current_point[0]][current_point[1]]
    shortest_path.reverse()
    return shortest_path

def interpolate(E,T,Na=100,l=None,wx=None,wy=None):
    """
    E is a 1d array
    T is a (ne,nk) array
    Na is the number of interpolated curves between 2 successive k-points
    l is the minimum euclidean length between 2 successive data
    wx,wy is the weight applied for distance calculation to E,T
    return T where Na*(nk-1) interpolated k points have been added : shape (ne,(Na+1)*nk-Na)
    """
    ne, nk = T.shape

    if l == None:
        l = np.median((np.diff(np.tile(E,[nk,1]).T,axis=0)**2+np.diff(T,axis=0)**2)**0.5)
    if wx == None:
        wx = 1/(E.max()-E.min())**2
    if wy == None:
        wy = 1/(T.max()-T.min())**2

    #TRANSFORM OF DATA
    E_tr = []
    T_tr = []
    for j in range(nk):
        Ek_tr = []
        Tk_tr = []
        
        Tk = T[:,j]
        
        for i, (ei,ti) in enumerate(zip(E[:-1],Tk[:-1])):
            Ek_tr.append(E[i])
            Tk_tr.append(Tk[i])

            ej,tj = E[i+1], Tk[i+1]
            L = ((ej-ei)**2+(tj-ti)**2)**0.5

            if L > l:
                E_add = np.linspace(ei,ej,int(L/l)+1,endpoint=False)[1:]
                line_int = interp1d([ei,ej],[ti,tj])
                Ek_tr += list(E_add)
                Tk_tr += list(line_int(E_add))

        Ek_tr.append(E[-1])
        Tk_tr.append(Tk[-1])
        
        E_tr.append(Ek_tr)
        T_tr.append(Tk_tr)

    # CORRESPONDANCE BETWEEN CURVES
    alphas = np.linspace(0,1,Na+1,endpoint=False)[1:]

    Tnew = np.zeros((ne,(Na+1)*nk-Na))
    for j in range(nk):
        Tnew[:,j*(Na+1)] = T[:,j]

    for j in range(nk-1):
        Ek, Tk = np.array(E_tr[j]), np.array(T_tr[j])
        Ek1, Tk1 = np.array(E_tr[j+1]), np.array(T_tr[j+1])
        D = distance_matrix(np.stack([Ek*wx**0.5,Tk*wy**0.5]).T, np.stack([Ek1*wx**0.5,Tk1*wy**0.5]).T)
        N1,N2 = D.shape[0]-1,D.shape[1]-1
        distance, prev_point = _find_shortest_paths(D, (0, 0))
        C = np.array(_find_shortest_path(prev_point, (N1,N2)))
            
        #INTERPOLATION OF CURVES
        for ia,a in enumerate(alphas):
            Ea = a*Ek[C[:,0]] + (1-a)*Ek1[C[:,1]]
            Ta = a*Tk[C[:,0]] + (1-a)*Tk1[C[:,1]]
            Tnew[:,j*(Na+1)+ia+1] = interp1d(Ea,Ta)(E)

    return Tnew