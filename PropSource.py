#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:09:48 2021

@author: tiagoburiol
@coworker: buligonl
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.linalg import solve


"""
A Python implementation of the “fingerprinting approach” based on the Clarke and Minella (2016) method.  
"""
class PropSource:
    
    '''Inicialize'''
    def __init__(self, filename):
        self.df = pd.read_excel(filename, sheet_name=None)
        self.CB = np.array(self.df['g_Source(CB)'].values)
        self.UR = np.array(self.df['g_Source(UR)'].values)
        self.CF = np.array(self.df['g_Source(CF)'].values)
        self.Y = np.array(self.df['Sediment(Y)'].values)
        self.Si = np.linalg.inv(np.cov(self.Y.T))
        return None
        
    
    def infos(self):
        for key in self.df:
            print ("Sheet name:", key)
            print ("Columns:", self.df[key].columns.tolist())
            print ("Number of samples:", self.df[key].shape[0])
            print("--")
            
    def nsample(self,ns):
        ns=[]
        for key in self.df:
            nsa =self.df[key].shape[0]
            ns.append(nsa)
        return np.array(ns)            
    
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
        
        
    '''
    ----------------------------------------------------------------
    SOLVERS OPTIONS: To solve the overdetermined linear system of equations by least squares method
    1) OLS - ordinary least squares = solve_ols_4x4 (solve_ols_2x2 or solve_minimize2 or solve_ols_4x4_mod)
    2) GLS - generalized least squares = solve_gls_4x4
                    
    ----------------------------------------------------------------
    '''

    def solve_ols_4x4(self,y,d,e,l, normalize=True):
        if normalize == True:
             d=d/y; e=e/y; l=l/y; y=y/y
        X = np.array([d/y,e/y,l/y])
        A = np.vstack([X.T, np.ones(3)])
    
        AtA = np.dot(A.T,A)
        AtA = np.vstack([AtA, np.ones(3)])
        AtA = np.vstack([AtA.T, np.ones(4)]).T
        AtA[-1,-1] = 0
    
        y = np.append(y,[1])
        y = y[:, np.newaxis]
        
        Aty = np.dot(A.T,y)
        Aty = np.append(Aty,[1])

        P = np.dot(np.linalg.inv(AtA),Aty)
        return (P)
        
    def solve_gls_4x4(self,y,d,e,l,S_inv):
        #Xy = np.array([d/y,e/y,l/y]).T
        X = np.array([d,e,l]).T
        #S_inv = np.linalg.inv(np.cov(X.T))
  
        B = S_inv.dot(X)
        C = S_inv.dot(y)
        AtA = np.dot(X.T,B)
        yty = np.dot(X.T,C)
        AtA = np.vstack([AtA, np.ones(3)])
        AtA = np.vstack([AtA.T, np.ones(4)]).T
        AtA[-1,-1] = 0
        Aty = np.append(yty,[1])
        Ps = np.dot(np.linalg.inv(AtA),Aty)
        return (Ps, X, y, AtA, Aty)
    '''
    ----------------------------------------------------------------
    SOLVERS OLS Extras: 
    '''
    
    def solve_ols_2x2(self,y,d,e,l, normalize = True):
        if normalize == True:
            d=d/y; e=e/y; l=l/y; y=y/y
        A =  np.array([(d-l).T , (e-l).T]).T
        y=y-l
        #P = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y-l)
        P = np.linalg.lstsq(A, y, rcond=None)[0]
        P = np.append(P,1-P[0]-P[1])
        return P 
        
    def solve_minimize2(self,y,d,e,l):
        from scipy.optimize import minimize, Bounds, LinearConstraint
        A =  np.array([d.T ,e.T, l.T]).T
        P0 = np.array([0.3, 0.3, 0.3])
        #X0 = -np.log((1./P0)-1.)
        def f(P):   
            return sum(((y-np.dot(A,P))/y)**2)
        lc = LinearConstraint([1,1,1], [1], [1])
        bnds = Bounds([0,0,0],[1,1,1])
        S = minimize(f, P0, 
                     method='trust-constr', 
                     #jac=rosen_der, 
                     #hess='cs',
                     constraints=lc,
                     options={'verbose': 0},
                     bounds=bnds)
        #P = 1.0/(1.0+np.exp(-S.x))
        P = S.x
        return(P)
    
    def solve_ols_4x4_mod(self,y,d,e,l, normalize=True):
        if normalize == True:
             d=d/y; e=e/y; l=l/y; y=y/y
        a = np.array([d/y,e/y,l/y])
        A = self._buildSourceContribMatrix(a)
        Z = np.append(np.dot(a, y/y), 1)
        P = np.dot(np.linalg.inv(A),Z)
        #P = solve(A, Z)
        return (P)
        
    '''
    ----------------------------------------------------------------------
    DISTANCE OPTIONS: Use scipy.spatial.distance implemantation for 
    compute distance between two numeric vectors. 
    - braycurtis: Compute the Bray-Curtis distance between two 1-D arrays.
    - euclidian: Computes the Euclidean distance between two 1-D arrays.
    - mahalanobis: Compute the Mahalanobis distance between two 1-D arrays. 
    -----------------------------------------------------------------------
    '''
    def mahalanobis_dist(self, P_arr, Pm):
        from scipy.spatial import distance
        S = np.cov(P_arr)
        dist = []
        for P in P_arr.T:
            d = distance.mahalanobis(P, Pm, np.linalg.inv(S))       
            dist.append(d)
        return np.array(dist)
        
    def braycurtis_dist(self,P_arr):
        from scipy.spatial import distance
        Pm = np.mean(P_arr, axis=1)
        #S = np.cov(P_arr)
        dist = []
        for P in P_arr.T:
            #S = np.cov(P,Pm)
            d = distance.braycurtis(P, Pm)
            dist.append(d)
        return np.array(dist)
        
    def euclidean_dist(self,P_arr):
        from scipy.spatial import distance
        Pm = np.mean(P_arr, axis=1)
        
        dist = []
        for P in P_arr.T:
            d = distance.euclidean(P, Pm)
            dist.append(d)
        return np.array(dist)
        
    '''
    Calculate de confidence region
    '''    
    def confidence_region(self, P, p = 95, spacedist= 'mahalanobis'):
        if P.shape[0]>2:
            P=P[0:2,:]
        Pm = np.mean(P, axis=1)
        if spacedist=='mahalanobis':
            dist = self.mahalanobis_dist(P, Pm)
        if spacedist=='euclidean':
            dist = self.euclidean_dist(P)
        
        #dist = self.mahalanobis_dist(P, Pm)
        sorted_idx = np.argsort(dist)
        Psorted = P.T[sorted_idx].T
        end_idx = int((p/100)*len(Psorted.T))
        return (Psorted[:,:end_idx])
        
        
        
        
    
    def draw_hull(self, P, ss, n, title = "Convex Hull", savefig = True,
                        xlabel = "P1", ylabel="P2"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        points = P.T   
        hull = ConvexHull(points)

        fig = plt.figure(figsize=(6, 4))
    
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        #plt.plot(Pm[0],Pm[1], "ro")
        plt.plot(P[0],P[1], "k," )
    
        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'k-')
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'k')
        plt.title(title)
        plt.xlabel('P1')
        plt.ylabel('P2')
        plt.xlim([-0.1, 0.9])
        plt.ylim([-0.1, 0.9])
        if savefig == True:
            plt.savefig(title+'.png')
        plt.show()
        return hull
    
    
    def plot2D(self, P, ss, n, marker=',', mean = False, convex_hull = False, 
                     title = "Scatter plot: P1 and P2", savefig= True,
                     xlabel = "P1", ylabel= "P2"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        
        #fig = plt.figure(figsize=(6, 6))
        if mean==True: 
            Pm = np.mean(P, axis=1)
            plt.plot(Pm[0],Pm[1], "ko")
            
        if convex_hull ==True:
            from scipy.spatial import ConvexHull
            points = P.T   
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
                #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=1)
                #plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
                #plt.xlim([-0.1, 0.9])
                #plt.ylim([-0.1, 0.9])
                #plt.axis('equal')

            print ("Area:", hull.volume) # para convex hull 2d é a área
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(1.0) #sets the height to width ratio to 1.5. 
        plt.xlim(-0.1, 1)
        plt.ylim(-0.1, 1)
        plt.plot(P[0],P[1], "k,")
        plt.grid()                
        #plt.title(title+ '('+str(P.shape[1])+')')
        plt.title(title+ '('+str(ss)+')'+ '('+str(n)+')'+ '('+str(P.shape[1])+')')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.axis('equal')

        

        if savefig == True:
            #plt.savefig(title+ '.png')
            plt.savefig(title+' Samples='+str(ss)+'_and_runs='+str(n)+'.png')
        plt.show()




    '''
  Subset random: It randomly chooses, without repetition, the samples from each subset.
    '''
    def randon_choice(self, nY, nCB, nUR, nCF):
        Ys = self.Y[np.random.choice(len(self.Y), nY, replace=False)] 
        CBs = self.CB[np.random.choice(len(self.CB), nCB, replace=False)] 
        URs = self.UR[np.random.choice(len(self.UR), nUR, replace=False)] 
        CFs = self.CF[np.random.choice(len(self.CF), nCF, replace=False)] 
        return Ys,CBs,URs,CFs


    '''
    Run
    '''
    def run(self, Ys,Ds,Es,Ls, solve=0):
        P1 = []; P2 = []; P3 = []
        nY = len(Ys); nD = len(Ds); nE = len(Es); nL = len(Ls)
        S_inv = self.Si
        for i in range (nY):
            y = Ys[i]
            for j in range (nD): 
                d = Ds[j]
                for k in range(nE):
                    e = Es[k]
                    for w in range(nL):
                        l = Ls[w]
                        if solve==0:
                            P = self.solve_ols_2x2(y,d,e,l, normalize=True)
                        elif solve==1:
                            #P = self.solve_minimize2(y,d,e,l)
                            P = self.solve_ols_4x4(y,d,e,l, normalize=True)
                        elif solve==2:
                            P = self.solve_gls_4x4(y,d,e,l,S_inv)[0]
                        elif solve==3:
                            P = self.solve_minimize2(y,d,e,l)
                        # Inclui apenas valores em que P1,P2>0 e P1+P2<1
                        if P[0]>0 and P[1]>0 and P[2]>0:
                            if P[0]<1 and P[1]<1 and P[2]<1:
                                P1.append(P[0])
                                P2.append(P[1])
                                P3.append(P[2])
                                        #print (P[0], P[1], P[2])
        #print("Quantidade de soluções viáveis:", len(P1))
        return np.array([P1, P2, P3])
            
 
           
    def report(self, infos, coefs_var, areas_mean):
        print("nSamp \tMean \tStd \tTotal \tFeas \tMeanP1 \tMeanP2 \tMeanP3")
        print('_____________________________________________________________')
        for row in infos:
            print("%i \t%.3f \t%.3f \t%i \t%i \t%.3f \t%.3f \t%.3f" \
              % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
        print('_____________________________________________________________')
        print ("Mean of the areas:", np.round(areas_mean,3))
        print ("Coefficient of variation:", np.round(coefs_var,3))
    

        
    def multi_runs(self, n, nY,nCB,nUR,nCF, plots2D=[]):
        from scipy.spatial import ConvexHull
        cv = lambda x: np.std(x) / np.mean(x) *100
        
	##repeat the simulation n times varying nY, nCB, nUR and nCF
        CVs_0= []; CVs_1= []; CVs_2= []; CVs_m= []
        areas_mean_0 = []; areas_mean_1 = []; areas_mean_2 = []; 
        infos_0 = []; infos_1 = []; infos_2 = []; 
        areas_mean_m = []
        
        if isinstance(nY, list):
            samples_sizes = nY.copy()
            #self.Si = np.linalg.inv(np.cov(self.Y.T))
            idx = 0
        if isinstance(nCB, list):
            samples_sizes = nCB.copy()
            #self.Si = np.linalg.inv(np.cov(self.D.T))
            idx = 1
        if isinstance(nUR, list):
            samples_sizes = nUR.copy()
            #self.Si = np.linalg.inv(np.cov(self.E.T))
            idx = 2
        if isinstance(nCF, list):
            samples_sizes = nCF.copy()
            #self.Si = np.linalg.inv(np.cov(self.L.T))
            idx = 3

        print('Samples sizes:',samples_sizes)
        for ss in samples_sizes:
    	    areas_0 = []; areas_1 = []; areas_2 = []
    	    
    	    # Choosing the source or sediment to reduce the number of samples
    	    if idx == 0: 
    	        nY = ss
    	    if idx == 1: 
    	        nCB = ss    
    	    if idx == 2: 
    	        nUR = ss 
    	    if idx == 3: 
    	        nCF = ss 
    
    	    print('Sample size:', ss)
    	    for i in range(n):
                Ys,CBs,URs,CFs = self.randon_choice(nY, nCB, nUR, nCF) 
                nSamples = nY*nCB*nUR*nCF
                #Ymean = Ys.mean(axis=0)
        
                # Solve = 0 
                P_0 = self.run(Ys,CBs,URs,CFs, solve=0)
                P_0 = self.confidence_region(P_0, p = 95)
                points_0 = P_0.T   
                hull_0 = ConvexHull(points_0)
                areas_0.append(hull_0.volume)
              

                # Solve = 1 
                P_1 = self.run(Ys,CBs,URs,CFs, solve=1)
                P_1 = self.confidence_region(P_1, p = 95)
                points_1 = P_1.T   
                hull_1 = ConvexHull(points_1)
                areas_1.append(hull_1.volume)
        
                # Solve = 2 
                P_2 = self.run(Ys,CBs,URs,CFs, solve=2)
                P_2 = self.confidence_region(P_2, p = 95)
                points_2 = P_2.T   
                hull_2 = ConvexHull(points_2)
                areas_2.append(hull_2.volume)
                
            #choose which reduction to print figure ###   
    	    if ss in plots2D:
    	        self.plot2D(P_0, ss, n, mean=True, convex_hull = True, title = "Confidence region 95%_OLS_Clarke Model")
    	        self.plot2D(P_2, ss, n, mean=True, convex_hull = True, title = "Confidence region 95%_GLS_Clarke Model")
         ###### #####
        
            
    	    infos_0.append([ss, np.mean(areas_0),np.std(areas_0), nSamples, \
    	                    len(points_0), np.mean(P_0[0]), np.mean(P_0[1]), \
    	                    1-np.mean(P_0[0])- np.mean(P_0[1])])
            
    	    infos_1.append([ss, np.mean(areas_1),np.std(areas_1), nSamples, \
    	                    len(points_1), np.mean(P_1[0]), np.mean(P_1[1]), \
    	                    1-np.mean(P_1[0])- np.mean(P_1[1])])
            
    	    infos_2.append([ss, np.mean(areas_2),np.std(areas_2), nSamples, \
    	                    len(points_2), np.mean(P_2[0]), np.mean(P_2[1]), \
    	                    1-np.mean(P_2[0])- np.mean(P_2[1])])
        
    	    CVs_0.append(cv(areas_0))
    	    areas_mean_0.append(np.mean(areas_0))

    	    CVs_1.append(cv(areas_1))
    	    areas_mean_1.append(np.mean(areas_1))
    
    	    CVs_2.append(cv(areas_2))
    	    areas_mean_2.append(np.mean(areas_2))
        print('--------------------------------------------------------------')
        print('Solve_0')
        self.report(infos_0, CVs_0, areas_mean_0)
        print('--------------------------------------------------------------')
        print('Solve_1')
        self.report(infos_1, CVs_1, areas_mean_1)
        print('--------------------------------------------------------------')
        print('Solve_2')
        self.report(infos_2, CVs_2, areas_mean_2)
        print('--------------------------------------------------------------')
        print("Next step: Figure")
        return (CVs_0,CVs_1, CVs_2)
        
       
            
            
            
