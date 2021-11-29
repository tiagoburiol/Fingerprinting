#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:09:48 2021

@author: tiagoburiol
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.linalg import solve


"""
A Python implementation of Clarke and Minella (2016) method of measures the 
increase in uncertainty when sampling sediment fingerprint
"""
class Fingerprint:

    def __init__(self, filename):
        self.df = pd.read_excel(filename, sheet_name=None)
        return None
    
    def infos(self):
        for key in self.df:
            print ("Sheet name:", key)
            print ("Columns:", self.df[key].columns.ravel())
            print ("Number of samples:", self.df[key].shape[0])
            print("--")
    
    def means(self):
        for key in list(self.df):
            print (key,':')
            print ( self.df[key].mean(axis=0))
            print ('--')
            
    def std(self):
        for key in list(self.df):
            print (key,':')
            print ( self.df[key].std(axis=0))
            print ('--')
            
    def hist(self):
        for key in list(self.df):
            self.df[key].hist(bins=10)
            
    def _buildSourceContribMatrix(self, a):
        g = len(self.df.keys())-1 # número de fontes
        A = np.ones((g+1,g+1))
        A[:g,:g] = np.dot(a,a.T)
        A[g, g]=0
        return A     
    
    def plot2D(self, P, marker=',', mean = False, convex_hull = False):
        if mean==True: 
            Pm = np.mean(P, axis=1)
            plt.plot(Pm[0],Pm[1], "r.")
            
        if convex_hull ==True:
            from scipy.spatial import ConvexHull
            points = P.T   
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
                plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=1)
                plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

            print ("Área:", hull.volume) # para convex hull 2d é a área
            
        plt.plot(P[0],P[1], ",")
        plt.grid()
        plt.show()

            
    def run(self, nY, nD, nE, nL):
        #from numpy.linalg import inv
        D = np.array(self.df['a_fonte(D)'].values)
        E = np.array(self.df['a_fonte(E)'].values)
        L = np.array(self.df['a_fonte(L)'].values)
        Y = np.array(self.df['Sedimentos(Y)'].values)
        
        P1 = np.array([])
        P2 = np.array([])
        P3 = np.array([])
        nsamples = nY*nD*nE*nL
        #print ("Número de amostras:", nsamples)
        #print ("Rodando... aguarde")
        Ys = Y[np.random.choice(len(Y), nY, replace=False)] # toma uma amostra aleatória de Y
        Ds = D[np.random.choice(len(D), nD, replace=False)] # toma uma amostra aleatória de D
        Es = E[np.random.choice(len(E), nE, replace=False)] # toma uma amostra aleatória de E 
        Ls = L[np.random.choice(len(L), nL, replace=False)] # toma uma amostra aleatória de L
        
        for i in range (nsamples):
            randY = Ys[randrange(len(Ys))]    # toma uma amostra aleatória de randY
            randD = Ds[randrange(len(Ds))]    # toma uma amostra aleatória de randD 
            randE = Es[randrange(len(Es))]    # toma uma amostra aleatória de randE 
            randL = Ls[randrange(len(Ls))]    # toma uma amostra aleatória de randL
    
            # Cria uma matriz com as contribuições (ai) normalizadas de D, E e L nas linhas 
            # Cada vetor linha está sendo dividido por randY para normalizar
            a = np.array([randD/randY, randE/randY, randL/randY])
    
            # Chama a função para montar a matriz A 
            A = self._buildSourceContribMatrix(a)
    
            # Cria o vetor Z
            Z = np.append(np.dot(a, randY/randY), 1)
    
    
            # Resolve o sistema para AP = Z para a solução P
            #P = np.dot(inv(A),Z)
            P = solve(A, Z)
            
        
            # Inclui apenas valores em que P1,P2>0 e P1+P2<1
            if P[0]>0 and P[1]>0 and 1-P[1]-P[0]>0:
                P1 = np.append(P1, P[0])
                P2 = np.append(P2, P[1])
                P3 = np.append(P3, 1-P[1]-P[0])

        #print("Quantidade de soluções viáveis:", len(P1))
        return np.array([P1,P2])
            
    def confidence_region(self, P, p = 95):
        Pm = np.mean(P, axis=1)
        #print ("P médio: Pm=",Pm)
        
        #desvios
        vect = np.array([P[0]-Pm[0], P[1]-Pm[1]]) 
        
        S = np.cov(P)
        #print(vect.shape)
        dist = []
        for v in vect.T:
            d = np.dot(np.dot(v.T, np.linalg.inv(S)), v)
            dist.append(d)
        
        # distâncias
        dist = np.array(dist)
        #dist = np.linalg.norm(vect.T, axis=1)
        #print ("Distâncias:", dist)
        
        sorted_idx = np.argsort(dist)
        Psorted = P.T[sorted_idx].T
        #print(Psorted.shape)
        
        # em ordem crescente
        end_idx = int((p/100)*len(Psorted.T))
        #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
        return (Psorted[:,:end_idx])
    
    
    def multi_runs(self, n, nY,nD,nE,nL):
        from scipy.spatial import ConvexHull
        import numpy as np
        cv = lambda x: np.std(x) / np.mean(x) *100

        areas = []
        #areas_medias = []
        for i in range(n):
            #print ("Rodando simulação número:", i)
            P = self.run(nY,nD,nE,nL)
            P = self.confidence_region(P, p = 95)
    
            hull = ConvexHull(P.T)
            areas.append(hull.volume)
                
        mean = np.round(np.mean(areas),3)
        SD = np.round(np.std(areas),4)
        P1 = np.round(np.mean(P[0]),3)
        P2 = np.round(np.mean(P[1]),3)
        P3 = np.round(1-P1-P2,3)
        #print ('nY','nD','nE','nL','Mean','SD','Possible','Feasible','P1','P2','P3')
        print (nY,nD,nE,nL, mean, SD, nY*nD*nE*nL, len(P.T), P1, P2, P3)
        CV = cv(areas)     
        return (CV)
            
            
            