class PropagationParameters:
   # Graph
   # L = no of iterations
   # n = no of nodes
    def __init__(self,n,e,a,b,X,W,Y,L=5,t=5,alpha = 0.85):
        self.n = n
        self.L = L
        self.e = e
        self.a = a
        self.b = b
        self.X = X
        self.W = W
        self.Y = Y
        self.t = t
        self.alpha = alpha
