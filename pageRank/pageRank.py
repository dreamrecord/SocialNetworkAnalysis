#!/usr/bin/python

import numpy as np
import copy

CONVERGE_THRES = 10**-3

class pageRank:
    
    def __init__(self, adjMatrix, dampingFactor, convergeThres = CONVERGE_THRES):
        
        # adj matrix
        self.A = copy.deepcopy(adjMatrix)
        self.dim = self.A.shape[0]
        self.convergeThres = convergeThres

        for row in self.A:
            row /= row.sum()

        self.A = (1-dampingFactor) * np.ones(self.A.shape) / self.dim + dampingFactor * self.A

        # page rank 
        self.P = np.ones(self.dim) / self.dim

    def runKIter(self, k):
        
        for i in xrange(k):
            self.P = np.dot(self.A.transpose(), self.P)

    def run(self):
        
        numOfIters = 0
        diff = self.P.sum()

        while diff > self.convergeThres:

            newP = np.dot(self.A.transpose(), self.P)
            diff = np.linalg.norm(newP-self.P, 1)
            self.P = copy.deepcopy(newP)
            
            numOfIters += 1

        return numOfIters 

    def printPageRank(self):
        
        print self.P

    def printAdjMatrix(self):
        
        print self.A

if __name__ == '__main__':
    
    # parameter
    adjMatrix = np.array([[0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0]], dtype=float)
    dampingFactor = 0.8

    pageRankAlgo = pageRank(adjMatrix, dampingFactor)

    print 'Adjacency Matrix'
    pageRankAlgo.printAdjMatrix()

    print 'Original Page Rank Value:'
    pageRankAlgo.printPageRank()

    numOfIters = pageRankAlgo.run()

    print 'Page Rank Value after', numOfIters, 'iters:'
    pageRankAlgo.printPageRank()
