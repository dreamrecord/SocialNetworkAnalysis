#!/usr/bin/python

import numpy as np

class HITS:
    
    def __init__(self, adjMatrix):
        
        self.authMatrix = np.dot(adjMatrix.transpose(), adjMatrix)
        self.hubMatrix= np.dot(adjMatrix, adjMatrix.transpose())
        self.dim = adjMatrix.shape[0]

        self.auth = np.ones(self.dim)
        self.hub = np.ones(self.dim)

    def runKIter(self, k):
        
        for i in xrange(k):
            
            self.auth = np.dot(self.authMatrix, self.auth)
            self.hub = np.dot(self.hubMatrix, self.hub)

            # normalize
            self.auth /= self.auth.sum()
            self.hub /= self.hub.sum()

    def printAuthHubTupleList(self):
        
        print zip(self.auth, self.hub)
