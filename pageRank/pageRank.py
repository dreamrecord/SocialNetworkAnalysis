#!/usr/bin/python

import numpy as np

class pageRank:
    
    def __init__(self, adjMatrix, dampingFactor):
        
        # adj matrix
        self.A = adjMatrix[:]
        self.dim = self.A.shape[0]

        for row in self.A:
            row /= row.sum()

        self.A = (1-dampingFactor) * np.ones(self.A.shape) / self.dim + dampingFactor * self.A

        # page rank 
        self.P = np.ones(self.dim) / self.dim

    def runKIter(self, k):
        
        for i in xrange(k):
            self.P = np.dot(self.A.transpose(), self.P)

    def printPageRank(self):
        
        print self.P

    def printAdjMatrix(self):
        
        print self.A
