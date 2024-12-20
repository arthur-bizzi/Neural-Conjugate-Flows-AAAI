import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, sqrt

try:
    import siren
    import rff
except:
    pass

#-----------------------------------------------------------------------------------------

def xavier_normal_init(m):
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        torch.nn.init.xavier_normal_(m.weight, gain=gain)
        torch.nn.init.zeros_(m.bias)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.zeros_(m.bias)

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('leaky_relu')
        torch.nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.zeros_(m.bias)

def ncf_matrix_init(eq,x0, b = False, pad_mode = 'no', pad_times = 1):
    x0 = x0.view(-1)
    f = eq.f(x0).view(-1,1)
    J = torch.autograd.functional.jacobian(eq.f,x0)
    p = x0.view(-1,1)
    pT = p.T
   
    Jp = J @ p
    fp_minus_Jp = (f - Jp)
    p_norm = pT @ p
    M = J + torch.matmul((fp_minus_Jp),pT)/p_norm
    if pad_mode == 'fourier':
        M = (M-M.T)/2
        M = torch.block_diag(*[M*(i+1) for i in range(pad_times)])
    elif pad_mode == 'twin':
        M = torch.block_diag(*[M for i in range(pad_times)])

    if b:
        b = f - M @ x0
        return M, b
    else:
        return M

hard_tanh = nn.Hardtanh()
def LeakyTanh():
    def f(x):
        return (0.01*x+hard_tanh(x))/1.01
    return f

#-----------------------------------------------------------------------------------------------

class OrthogonalLayer(nn.Module):
    def __init__(self, dims_in,dims_out=None, bias = False, init_weight_scale = 1/50, **kwargs):
        super().__init__()
        if dims_out is None:
            dims_out = dims_in
        weight = torch.randn(dims_in,dims_out) * init_weight_scale
        weight = (weight - weight.T)
        weight = torch.matrix_exp(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dims_out))
        else:
            self.register_buffer('bias',torch.zeros(dims_out))
            
        nn.utils.parametrizations.orthogonal(self, **kwargs)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def inverse(self, x):
        return F.linear(x - self.bias, self.weight.T)
    

class RescaleLayer(nn.Module):
    def __init__(self, dims_in,dims_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dims_in)/10)

    def forward(self, x):
        tanh_w = torch.tanh(self.weight)
        return torch.exp(tanh_w)*x

    def inverse(self, x):
        tanh_w = torch.tanh(self.weight)
        return torch.exp(-tanh_w)*x


class SVDLayer(nn.Module):
    def __init__(self, in_dims, out_dims, * , min_singular_value = 0.5, max_singular_value = 1.5, bias_mode = None, orthogonal_map = 'cayley') -> None:
        super().__init__()
        if bias_mode == 'first':
            self.U = OrthogonalLayer(in_dims,out_dims, bias = True, orthogonal_map = orthogonal_map)
            self.V = OrthogonalLayer(in_dims,out_dims, orthogonal_map = orthogonal_map)
        elif bias_mode == 'last':
            self.U = OrthogonalLayer(in_dims,out_dims, orthogonal_map = orthogonal_map)
            self.V = OrthogonalLayer(in_dims,out_dims, bias = True, orthogonal_map = orthogonal_map)
        else:
            self.U = OrthogonalLayer(in_dims,out_dims, orthogonal_map = orthogonal_map)
            self.V = OrthogonalLayer(in_dims,out_dims, orthogonal_map = orthogonal_map)

        self.Sigma = RescaleLayer(out_dims,out_dims)
    
    def forward(self,x):
        x = self.U(x)
        x = self.Sigma(x)
        x = self.V(x)
        return x
    
    def inverse(self,x):
        x = self.V.inverse(x)
        x = self.Sigma.inverse(x)
        x = self.U.inverse(x)
        return x

def norm(x):
    return torch.sqrt((x*x).sum(dim=-1,keepdim=True))

class SphericalReflection(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.center = nn.Parameter(-10*torch.ones(dims))
        self.reflection_plane = nn.Parameter(-10*torch.ones(dims))
        self.B = nn.Parameter(torch.eye(dims))
    
    def householder(self, x, v):
        v = v / norm(v)
        v = v.view(1,1,-1)
        xv = (x*v).sum(dim=-1, keepdim=True)
        return x - 2*xv*v

    def forward(self, x):
        center = self.center
        reflection_plane = self.reflection_plane
        x = x - center
        x = x*norm(center)**2/(norm(x)**2) + center
        x = self.householder(x, reflection_plane)
        return x

    def inverse(self, x):
        center = self.center
        reflection_plane = self.reflection_plane
        x = self.householder(x, reflection_plane)
        x = x - center
        x = x*norm(center)**2/(norm(x)**2) + center
        return x

def CustomMLP(*dims, activation = nn.Tanh(), fourier_feature = False, last_layer_activation = False, clipping = False):
    layers = []
    if fourier_feature:
        layers.append(rff.layers.GaussianEncoding(sigma=2.0, input_size=dims[0], encoded_size=dims[1]//2))
    else:
        layers.append(nn.Linear(dims[0], dims[1]))
        layers.append(activation)
    for i in range(1,(len(dims)-1)):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(activation)
    net = nn.Sequential(*layers)
    if last_layer_activation:
        net.append(activation)
    if clipping:
        net.append(LeakyTanh())
    if activation == nn.Tanh():
        net.apply(xavier_init)
    elif activation == nn.LeakyReLU():
        net.apply(kaiming_init)
    return net
    

def CustomSiren(*dims, w0=1.0, w0_initial=20.0, initializer='siren', c = 6, clipping = False):
    dims = list(dims)
    layers = dims[1:-1]
    net = siren.SIREN(layers, dims[0], dims[-1], w0=1.0, w0_initial=20.0, initializer='siren', c = 6)
    if clipping:
        net = nn.Sequential(net, nn.Hardtanh())
    return net
#---------------------------------------------------------------------------------------------------


def identity(x):
    return x

def skew_symmetric(x):
    return (x - x.transpose(1,2))/2


class LinearFlow(nn.Module):
    def __init__(self,dims, M0 = None, lie_algebra = None, omega_zero = 1e-2, **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims))
        self.matrix_size = matrix_size
        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            self.register_buffer('M0', M0)
        M = torch.randn(matrix_size,matrix_size)*omega_zero
        self.M = torch.nn.Parameter(M)

        if lie_algebra is None:
            self.lie_algebra = identity
        elif lie_algebra == 'skew_symmetric':
            self.lie_algebra = skew_symmetric
    
    @property        
    def A(self):
        A = self.M + self.M0
        A = A.unsqueeze(0)
        # print(A.shape)
        A = self.lie_algebra(A)
        return A

    def forward(self, x, t, t0 = 0):
        t = t.view(-1,1,1)
        A_ex = torch.matrix_exp(t * self.A)

        x = x.view(1,-1,self.matrix_size)
        x = x.transpose(1,2)
        x = torch.matmul(A_ex,x)
        x = x.transpose(1,2)
        return x
    

class AffineFlow(nn.Module):
    def __init__(self,dims, M0 = None, b = None, lie_algebra = None, omega_zero = 1e-2, **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims))
        self.matrix_size = matrix_size
        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            self.register_buffer('M0', M0)
        M = torch.randn(matrix_size,matrix_size)*omega_zero
        self.M = torch.nn.Parameter(M)

        if lie_algebra is None:
            self.lie_algebra = identity
        elif lie_algebra == 'skew_symmetric':
            self.lie_algebra = skew_symmetric
        
        if b is None:
            self.b = torch.nn.Parameter(torch.zeros(matrix_size))
        else:
            self.register_buffer('b', b)
    
    @property        
    def A(self):
        A = self.M + self.M0
        A = self.lie_algebra(A)

        A = torch.cat([A,self.b.view(-1,1)],dim=1)
        A = torch.cat([A,torch.zeros(1,self.matrix_size+1)],dim=0)

        A = A.unsqueeze(0)
        # print(A.shape)
        return A

    def forward(self, x, t, t0 = 0):
        t = t.view(-1,1,1)
        A_ex = torch.matrix_exp(t * self.A)

        x = x.view(1,-1,self.matrix_size)
        x = F.pad(x,(0,1))
        
        x = x.transpose(1,2)
        x = torch.matmul(A_ex,x)
        x = x.transpose(1,2)
        x = x[...,:-1]
        return x


class HamiltonianLinearFlow(nn.Module): # NOT WORKING
    def __init__(self,dims, M0 = None, time_parameterization = False, **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims))
        assert matrix_size %2 == 0, 'Matrix size must be even'
        self.matrix_size = matrix_size

        zeros = torch.zeros(matrix_size//2,matrix_size//2)
        eye = torch.eye(matrix_size//2)
        J1 = torch.cat([zeros,eye],dim=1)
        J2 = torch.cat([-eye,zeros],dim=1)
        J = torch.cat([J1,J2],dim=0)
        self.register_buffer('J',J)

        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            M0 = self.extract_hamiltonian_matrix(M0)
            self.register_buffer('M0', M0)

        tau_net = CustomMLP(1, matrix_size, 1)
        self.tau_net = tau_net.apply(xavier_normal_init)

        self.time_parameterization = time_parameterization

        M = torch.randn(matrix_size,matrix_size)/10
        self.M = torch.nn.Parameter(M)


    def extract_hamiltonian_matrix(self, M):
        J = self.J
        M = -J @ M
        M = (M + M.T)/2
        return M
    
    @property        
    def A(self):
        A = self.M + self.M0
        A = (A + A.T)/2
        return A
    
    def JA(self, A):
        A = self.J @ A
        A = A.unsqueeze(0)
        A = A.unsqueeze(0)
        return A
    
    def hamiltonian(self, A, x):
        A = self.A
        Ax = F.linear(x, A)
        xAx = torch.sum(x*Ax,dim=-1,keepdim=True)
        return xAx
     
    def forward(self, x, t, t0 = 0):
        t = t.view(-1,1,1,1)
        x = x.view(1,-1,self.matrix_size)

        A = self.A
        time_parameterization = self.time_parameterization
        H = self.hamiltonian(A,x)
        H = H.view(1,-1,1,1)
        if self.time_parameterization:
            tau = self.tau_net(t)
        tau = (1 + 0.01*time_parameterization(H))*t
        # print("tau.shape: ",tau.shape)
        JA = self.JA(A)
        # print(JA.shape)
        # print(tauJA.shape)
        A_ex = torch.matrix_exp(tau * JA)
        # print(A_ex.shape)

        x = x.unsqueeze(-1)
        x = torch.matmul(A_ex,x)
        # print(x.shape)
        x = x.squeeze(-1)
        return x
    

class FourierFlow(nn.Module):
    def __init__(self,dims, M0 = None, lie_algebra = None, omega_zero = 2.0, **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims))
        self.matrix_size = matrix_size
        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            self.register_buffer('M0', M0)
        M = torch.tensor([[0,1],[-1,0]])
        M = [M*(i+1) for i in range(matrix_size//2)]
        M = torch.block_diag(*M)
        self.register_buffer('M', M)
        self.omega = nn.Parameter(omega_zero*torch.ones(1))

        if lie_algebra is None:
            self.lie_algebra = identity
        elif lie_algebra == 'skew_symmetric':
            self.lie_algebra = skew_symmetric

    def forward(self, x, t, t0 = 0):
        t = t.view(-1,1,1)
        # print(t.shape)
        A = self.M*self.omega + self.M0
        A = A.unsqueeze(0)
        # print(A.shape)
        A = A*t
        A_ex = torch.matrix_exp(A)

        x = x.unsqueeze(0)
        x = x.transpose(1,2)
        x = torch.matmul(A_ex,x)
        x = x.transpose(1,2)
        return x
    

class TauLinearFlow(nn.Module):
    def __init__(self,dims, M0 = None, lie_algebra = None, **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims))
        self.matrix_size = matrix_size
        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            self.register_buffer('M0', M0)
        M = torch.randn(matrix_size,matrix_size)
        self.M = torch.nn.Parameter(M)

        if lie_algebra is None:
            self.lie_algebra = identity
        elif lie_algebra == 'skew_symmetric':
            self.lie_algebra = skew_symmetric

    def forward(self, x, t, t0 = 0):
        t = t.view(-1,1,1)
        # print(t.shape)
        A = self.M + self.M0
        A = A.unsqueeze(0)
        # print(A.shape)
        A = self.lie_algebra(A)
        A = A*t
        A_ex = torch.matrix_exp(A)

        x = x.unsqueeze(0)
        x = x.transpose(1,2)
        x = torch.matmul(A_ex,x)
        x = x.transpose(1,2)
        return x
    

class TimeVaryingLinearFlow(nn.Module):
    def __init__(self,*dims, M0 = None, lie_algebra = None, type = 'mlp', **kwargs):
        super().__init__()
        matrix_size = int(sqrt(dims[-1]))
        self.matrix_size = matrix_size
        if type == 'mlp':
            self.model = CustomMLP(*dims,**kwargs)
        elif type == 'siren':
            self.model = CustomSiren(*dims,**kwargs)
        self.model.apply(xavier_normal_init)

        if M0 == None:
            self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
        else:
            self.register_buffer('M0', M0)

        if lie_algebra is None:
            self.lie_algebra = identity
        elif lie_algebra == 'skew_symmetric':
            self.lie_algebra = skew_symmetric
        else:
            #Only these two lie algebras are supported
            raise ValueError('''lie_algebra must be either None or "skew_symmetric" ''')

    def forward(self, x, t, t0 : float = 0):
        t = t.view(-1,1)
        t = F.pad(t,(0,0,1,0),'constant',t0)

        # print(t.shape, t.device)

        delta_t = t[1:]-t[:-1]
        t_avg = (t[1:]+t[:-1])/2

        M0 = self.M0.unsqueeze(0)
        # print(M0.shape, M0.device)

        A = self.model(t_avg)
        A = A.view(-1,self.matrix_size,self.matrix_size)
        A = (A+M0)
        A = A*(delta_t.unsqueeze(-1))
        # zeros = torch.zeros_like(A[0]).unsqueeze(0)
        # A = torch.cat([zeros,A],dim=0)
        A = self.lie_algebra(A)
        A_sum = torch.cumsum(A,dim=0)
        # print(A.shape, A.device)

        A_sum_ex = torch.matrix_exp(A_sum)

        x = x.unsqueeze(0)
        x = x.transpose(1,2)
        x = torch.matmul(A_sum_ex,x)
        x = x.transpose(1,2)
        return x
    

# class QuasiLinearFlow(nn.Module):
#     def __init__(self,dims, M0 = None, lie_algebra = None, **kwargs):
#         super().__init__()
#         matrix_size = int(sqrt(dims))
#         self.matrix_size = matrix_size
#         epsilon = torch.tensor([1e-2])
#         # self.epsilon = torch.nn.Parameter(epsilon)
#         self.register_buffer('epsilon',epsilon)

#         if M0 == None:
#             self.register_buffer('M0', torch.zeros(matrix_size,matrix_size))
#         else:
#             self.register_buffer('M0', M0)
#         M = torch.randn(matrix_size,matrix_size)
#         self.M = torch.nn.Parameter(M)

#         if lie_algebra is None:
#             self.lie_algebra = identity
#         elif lie_algebra == 'skew_symmetric':
#             self.lie_algebra = skew_symmetric

#     def forward(self, x, t, t0 = 0):
#         t = t.view(-1,1,1)
#         # print(t.shape)
#         A = self.M + self.M0
#         A = A.unsqueeze(0)
#         # print(A.shape)
#         A = self.lie_algebra(A)
#         At = A*t
#         A_ex = torch.matrix_exp(At)

#         x = x.unsqueeze(0)
#         x = x.transpose(1,2)
#         x = torch.matmul(A_ex,x)
#         # x = x.transpose(1,2)

#         x_cubed = torch.matmul(A,x)
#         x_cubed = x_cubed*x_cubed*x_cubed

#         A_ex_minus = torch.matrix_exp(-At)
#         x_int = torch.matmul(A_ex_minus,x_cubed)
#         int = torch.cumsum(x_int,dim=0)
#         # int = int.transpose(1,2)
#         return x_int
    






