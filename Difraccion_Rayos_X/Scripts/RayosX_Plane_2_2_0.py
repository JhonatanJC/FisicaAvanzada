# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_cube(cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    point  = np.array([0.5, 0, 1])
    normal = np.array([2, 2, 1])
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    #d = -point.dot(normal)
    fig = fig.gca(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    d = -point.dot(normal)
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy -d) * 1. /normal[2]
    for i,x in enumerate(z):
        for j,y in enumerate(x):
            if y >=1:
                z[i][j] = 1
            if y<0:
                z[i][j] = 0

    xx = ( - normal[1] * yy + 1) * 1. /normal[0]
    # plot the surface

    for i,x in enumerate(xx):
        for j,y in enumerate(x):
            if y >=1:
                xx[i][j] = 1
                yy[i][j] = 0.5
            if y<0:
                xx[i][j] = 0
                yy[i][j] = 0.5

    fig.plot_surface(xx, yy, z, alpha=0.6)

    fig.set_xlim(0,1)
    fig.set_ylim(0,1)
    fig.set_zlim(0,1) 
    
    ax.add_collection3d(faces)

    #and i would like to plot this point : 

    Points_inter = np.array([np.array([0.5,0.5,0,0,0.5]),np.array([0,0,0.5,0.5,0]),np.array([0,1,1,0,0])])
        
    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    ax.scatter(Points_inter[0],Points_inter[1],Points_inter[2], c="red", s=20)
    for x, y, z in zip(Points_inter[0,:-1], Points_inter[1,:-1], Points_inter[2,:-1]):
        label = '(%.1f, %.1f, %.1f)' % (x, y, z)
        fig.text(x, y, z, label, fontsize=15, va='bottom', color='black')


    ax.plot(Points_inter[0],Points_inter[1],Points_inter[2], color='red')
    
    ax.set_xlabel('$X$', fontsize=20, rotation=0)
    ax.set_ylabel('$Y$', fontsize=20, rotation=0)
    ax.set_zlabel('$Z$', fontsize=20, rotation=0)
    ax.set_title("Planos cristalográficos para el indice de Miller [2, 2, 0]", fontsize=28)
    ax.set_aspect('auto')

cube_definition = [
    (0,0,0), (0,1,0), (1,0,0), (0,0,1)
]
plot_cube(cube_definition)




