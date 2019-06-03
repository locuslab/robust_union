import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def collapseN(x, n): 
    return x.view(*x.size()[:-n], -1)
def unsqueezeN(x,n): 
    for _ in range(n): 
        x = x.unsqueeze(-1)
    return x

lookup_x, lookup_y = torch.load('./lookup_table.pth')

def calculate_linear_init(n,k=1): 
    for i in range(1,len(lookup_x)): 
        if lookup_x[i] > n: 
            break
    aij_sq = lookup_y[i-1].float()
    factor = (1 - 2/n + 3/(n*(n+2)) + (n-1)*aij_sq + 1)/2.
    return math.sqrt((1./n)/(factor**k))

def normN(x, n): 
    return unsqueezeN(collapseN(x, n).norm(dim=-1),n)

class Halfspace(nn.Module):
    def __init__(self, *size, k=0, bias=False):
        super(Halfspace, self).__init__()
        self.size = size
        
        normal = torch.randn(*size)
        unit_vector = normal/normN(normal, len(size)-k)

        self.normal_vector = nn.Parameter(unit_vector)
        self.k = k
        if bias: 
            if k > 0: 
                self.threshold = nn.Parameter(torch.zeros(*size[:k]))
            else: 
                self.threshold = nn.Parameter(torch.zeros(1))
        else: 
            self.threshold = 0
            
    def forward(self, x):
        ndim = len(self.size)
        ax = self.normal_vector*x
        ax = collapseN(ax, ndim - self.k).sum(-1)
        I = ax <= self.threshold
        Inz = I.view(-1).nonzero()[:,0]
         
        mag = unsqueezeN(collapseN(self.normal_vector**2, ndim-self.k).sum(-1),ndim-self.k)
        px = x  + unsqueezeN(self.threshold - ax, ndim - self.k)/mag*self.normal_vector

        x0 = x.clone().view(-1, *x.size()[-(ndim-self.k):]).contiguous()
        px0 = px.view(-1,*x.size()[-(ndim-self.k):]).contiguous()
        x0.index_copy_(0, Inz, px0[Inz])

        return x0.view(*x.size())

class FilterHalfspace0(Halfspace):
    def __init__(self, *size, k=0, bias=False):
        #super(FilterHalfspace, self).__init__()
        self.size = size
        assert len(size) == 1
        super(FilterHalfspace0, self).__init__(*size, 1, 1, bias=bias) 
        
    def forward(self, x):
        # redefine halfspace to be over the filter dimension instead
        ndim = len(self.size)
        ax = self.normal_vector*x
        #ax = collapseN(ax, ndim - self.k).sum(-1)
        ax = ax.sum(-3).unsqueeze(-3)
        I = ax <= self.threshold
         
        mag = (self.normal_vector**2).sum()
        px = x  + (self.threshold - ax)/mag*self.normal_vector

        return torch.where(I, px, x)


class FilterHalfspace(nn.Module):
    def __init__(self, size, k=0, bias=False, init=None, kernel_size=1, padding=0):
        super(FilterHalfspace, self).__init__()
        self.size = size
        
        if init is not None: 
            unit_vector = torch.zeros(size, kernel_size, kernel_size)
            unit_vector[init] = 1
        else: 
            normal = torch.randn(size, kernel_size, kernel_size)
            unit_vector = normal/(normal.norm())

        self.normal_vector = nn.Parameter(unit_vector)
        self.k = k
        self.padding = padding

        if bias:  
            self.threshold = nn.Parameter(torch.zeros(1))
        else: 
            self.threshold = 0
        
    def forward(self, x):
        # redefine halfspace to be over the filter dimension instead
        a = self.normal_vector.unsqueeze(0)
        #print(x.size(), a.size())
        ax = F.conv2d(x, a, padding=self.padding)
        #mag = torch.dot(self.normal_vector, self.normal_vector)
        # print(a.size(), ax.size(), x.size())
        return x + F.relu(self.threshold-ax)*a#/mag

        #ax = self.normal_vector*x
        #ax = collapseN(ax, ndim - self.k).sum(-1)
        # ax = ax.sum(-3).unsqueeze(-3)
        # I = ax <= self.threshold
         
        # mag = (self.normal_vector**2).sum()
        # px = x  + (self.threshold - ax)/mag*self.normal_vector

        # return torch.where(I, px, x)
        

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'normal_vector'):
            w = module.normal_vector.data
            # c = w.view(w.size(0), -1).norm(dim=-1)
            # while c.dim() < w.dim(): 
            #     c = c.unsqueeze(-1)
            c = w.norm()
            # print(c.size(), w.size())
            # assert False
            module.normal_vector.data = w/c
            # ndim = w.dim() - module.k
            # module.normal_vector.data = w/unsqueezeN(collapseN(w,ndim).norm(dim=-1), ndim)

if __name__ == '__main__': 
    hs = FilterHalfspace(5)
    hs0 = FilterHalfspace0(5)

    #print(hs.normal_vector, hs0.normal_vector)
    hs.normal_vector.data = hs0.normal_vector.data.view(-1,1,1)
    x = torch.randn(8,5,32,32)
    import time
    start_time = time.time()
    for _ in range(5000): 
        hs0(x)
    old = time.time()-start_time
    print(old)

    start_time = time.time()
    for _ in range(5000): 
        hs(x)
    new = time.time()-start_time
    print(new)
    print((old-new)/old)

    hses = [FilterHalfspace(5, init=i) for i in range(5)]
    y = x
    for hs in hses: 
        y = hs(y)
    print((y-F.relu(x)).abs().max())