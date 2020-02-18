import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi


def plot_voronoi(points, data_points, ax, line_style='--', line_colour='black', margin=0.5):
    vor = Voronoi(points)    
    centre = points.mean(axis=0)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - centre, n)) * n * 100
            ax.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], linestyle=line_style, color=line_colour)
            
    L = min(data_points[:, 0].min(), data_points[:, 1].min()) - margin
    R = max(data_points[:, 0].max(), data_points[:, 1].max()) + margin

    ax.set_xlim(L, R)        
    ax.set_ylim(L, R)        
    ax.set_aspect('equal')
