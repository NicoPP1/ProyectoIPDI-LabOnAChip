# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:46:28 2021

@author: nicop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import imageio
import glob

#Librerias y código para ordenar correctamente el batch (antes estaba primero el 10 que el 2)
import os
import re

pat=re.compile("(\d+)\D*$")

def key_func(x):
    mat=pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits.



#Batch
path = "3_ensayo/*.*"
#Contador según el numero de imagen a procesar
img_number = 1

# Implementation of Shoelace formula para obtener el area de un conjunto de puntos en el plano
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#Las unidades se encuentran en micrómetros
MradTestTube = 350
MareaTestTube = np.pi*(MradTestTube)**2

#obtengo dimensiones de imagen
url = "3_ensayo/BARRIDO1_1.jpg"
img = imageio.imread(url)
img = rgb2gray(img)

shape = img.shape
heightY = shape[0]
widthY = shape[1]

#Centros y radios del contorno de inicialización (unidades en pixeles)
cx = widthY//2 + 10
cy = heightY//2 - 20
radInit = 225
#Radio de la probeta en pixeles (distinto al de inicialización para obtener mejores resultados)
radTest = 240

#alpha depende de la cantidad de puntos, creo la circunferencia inicial
s = np.linspace(0, 2*np.pi, 800)
r = cy + radInit*np.sin(s)
c = cx + radInit*np.cos(s)
init = np.array([r, c]).T

rprob = cy + radTest*np.sin(s)
cprob = cx + radTest*np.cos(s)
prob = np.array([rprob, cprob]).T

print(f"Area de la probeta en micrómetros cuadrados: {MareaTestTube}")

#Area en pixeles de la probeta
areaTestTube = np.pi*(radTest)**2
#Listas que se exportaran a excel
imglist = []
arealist = []
radEquivlist = []

#Recorro los archivos de forma ordenada y realizo la detección del contorno
for file in sorted(glob.glob(path),key = key_func):

    if img_number < 10:
        alpha = 0.65
        beta = 0
        gamma = 0.005
    else:
        alpha = 0.3
        beta = 0
        gamma = 0.005



    img = imageio.imread(file)
    img = rgb2gray(img)
    gimage = gaussian(img,0.7)
    
    snake = active_contour(gimage,init, alpha=alpha, beta=beta, gamma = gamma,convergence=0.000001,max_iterations=5000)

    areaSnake = PolyArea(snake[:,0],snake[:,1])
    radEquiv = np.sqrt(areaSnake/np.pi)
    
    # print(radEquiv)
    er = cy + radEquiv*np.sin(s)
    ec = cx + radEquiv*np.cos(s)
    equivCircle = np.array([er, ec]).T
    
    MareaSnake = areaSnake*MareaTestTube/areaTestTube
    MradEquiv = np.sqrt(MareaSnake/np.pi)
    print(f"Area del contorno en micrómetros cuadrados: {MareaSnake} para el archivo {file}")
    
    # Serie de ifs para que, en caso de que el contorno anterior sea mayor o menor en un 
    # 50%, se realice una correción de alpha para obtener una detección correcta a lo largo
    # del batch
    if (img_number > 1) and (MareaSnake < 0.5*arealist[-1]) and (MareaSnake > 3000):

        alpha -= 0.2
        print(f"Area nueva menor al 50% de la anterior, alpha: {alpha} para {file}")
        snake = active_contour(gimage,init, alpha=alpha, beta=beta, gamma = gamma,convergence=0.000001,max_iterations=5000)

        areaSnake = PolyArea(snake[:,0],snake[:,1])
        radEquiv = np.sqrt(areaSnake/np.pi)
        
        # print(radEquiv)
        er = cy + radEquiv*np.sin(s)
        ec = cx + radEquiv*np.cos(s)
        equivCircle = np.array([er, ec]).T
        
        MareaSnake = areaSnake*MareaTestTube/areaTestTube
        MradEquiv = np.sqrt(MareaSnake/np.pi)
        print(f"Area del contorno en micrómetros cuadrados: {MareaSnake} para el archivo {file}")
    elif MareaSnake <= 3000:
        alpha = alpha/2
        print(f"Area anterior menor a 3000, alpha: {alpha} para {file}")
        snake = active_contour(gimage,init, alpha=alpha, beta=beta, gamma = gamma,convergence=0.000001,max_iterations=5000)

        areaSnake = PolyArea(snake[:,0],snake[:,1])
        radEquiv = np.sqrt(areaSnake/np.pi)
        
        # print(radEquiv)
        er = cy + radEquiv*np.sin(s)
        ec = cx + radEquiv*np.cos(s)
        equivCircle = np.array([er, ec]).T
        
        MareaSnake = areaSnake*MareaTestTube/areaTestTube
        MradEquiv = np.sqrt(MareaSnake/np.pi)
        print(f"Area del contorno en micrómetros cuadrados: {MareaSnake} para el archivo {file}")
        
    elif (img_number > 1) and (MareaSnake > 1.5*arealist[-1]):
        alpha += 0.2
        print(f"Area nueva mayor al 150% de la anterior, alpha: {alpha} para {file}")
        snake = active_contour(gimage,init, alpha=alpha, beta=beta, gamma = gamma,convergence=0.000001,max_iterations=5000)

        areaSnake = PolyArea(snake[:,0],snake[:,1])
        radEquiv = np.sqrt(areaSnake/np.pi)
        
        # print(radEquiv)
        er = cy + radEquiv*np.sin(s)
        ec = cx + radEquiv*np.cos(s)
        equivCircle = np.array([er, ec]).T
        
        MareaSnake = areaSnake*MareaTestTube/areaTestTube
        MradEquiv = np.sqrt(MareaSnake/np.pi)
        print(f"Area del contorno en micrómetros cuadrados: {MareaSnake} para el archivo {file}")
    
    
    plt.ioff()
    
    #Figuras
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(cx,cy, lw = 1,c = '#ffffff')
    ax.imshow(gimage, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.plot(equivCircle[:, 1], equivCircle[:, 0], '--g', lw=3)
    ax.plot(prob[:, 1], prob[:, 0], 'go--', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    
    #Guardo las figuras
    fig.savefig('resTest10/deteccion'+str(img_number)+'.jpg',bbox_inches="tight")
    
    # Agrego nombre, area y radio equivalente a las listas para su posterior exportación
    # a la hoja de calculos
    imglist.append(file)
    arealist.append(MareaSnake)
    radEquivlist.append(MradEquiv)
    img_number += 1
    
# Exporto la hoja de calculos    
data = pd.DataFrame({'Nombre': imglist,'Area':arealist,'radio eq':radEquivlist})
file_name = 'resTest10/data.xlsx'
data.to_excel(file_name)
print('El archivo fue escrito a Excel correctamente.')

