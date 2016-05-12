#!/usr/bin/python

import numpy as np

CONVERGE_THRES = 10**-3

class HITS:
    
    def __init__(self, adjMatrix, convergeThres = CONVERGE_THRES):
        
        self.authMatrix = np.dot(adjMatrix.transpose(), adjMatrix)
        self.hubMatrix= np.dot(adjMatrix, adjMatrix.transpose())
        self.dim = adjMatrix.shape[0]
        self.convergeThres = convergeThres

        self.auth = np.ones(self.dim)
        self.hub = np.ones(self.dim)

    def runKIter(self, k):
        
        for i in xrange(k):
            
            self.auth = np.dot(self.authMatrix, self.auth)
            self.hub = np.dot(self.hubMatrix, self.hub)

            # normalize
            self.auth /= self.auth.sum()
            self.hub /= self.hub.sum()

    def run(self):
        
        numOfIters = 0
        authDiff = self.auth.sum()
        hubDiff = self.hub.sum()

        while authDiff > self.convergeThres or hubDiff > self.convergeThres:

            newAuth = np.dot(self.authMatrix, self.auth)
            newHub = np.dot(self.hubMatrix, self.hub)

            # normalize
            newAuth /= newAuth.sum()
            newHub /= newHub.sum()

            authDiff = np.linalg.norm(newAuth-self.auth, 1)
            hubDiff = np.linalg.norm(newHub-self.hub, 1)
            
            self.auth = newAuth
            self.hub = newHub

            numOfIters += 1

        return numOfIters 

    def printAuthHubTupleList(self):
        
        print zip(self.auth, self.hub)

if __name__ == '__main__':
    
    # parameter
    adjMatrix = np.array([[0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0]], dtype=float)

    HITSAlgo = HITS(adjMatrix)

    print 'Original (Auth, Hub) tuple list:'
    HITSAlgo.printAuthHubTupleList()

    numOfIters = HITSAlgo.run()

    print '(Auth, Hub) tuple list After', numOfIters, 'iters:'
    HITSAlgo.printAuthHubTupleList()
