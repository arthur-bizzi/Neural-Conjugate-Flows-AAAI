import torch
import numpy as np
import torch.nn.functional as F

class ODE:
    # The RHS for an ODE.
    def f(self, u):
        return self.speed*torch.cat(self._f(u), dim=-1)
    
    def f_solver(self, t, u):
        # return self.speed*np.concatenate(self._f(u), axis=-1)
        return self.speed*torch.cat(self._f(u), dim=-1)

    def f_np_solver(self, t, u):
        return self.speed*np.concatenate(self._f(u), axis=-1)
    
    def jacobian(self, p):
        J = torch.autograd.functional.jacobian(self.f, p)

    
class HarmonicOscillator(ODE):
    # The classic harmonic oscillator.
    def __init__(self, speed = 1.0, weight=1.0, weight_mean=0.0, weight_primal=0.0):
        self.name = 'Harmonic Oscillator'
        self.order = 2
        self.speed = speed
        self.weight = weight
        self.weight_mean = weight_mean
        self.weight_primal = weight_primal

        self.equilibrium = torch.tensor([0., 0.])
        # def jacobian(p):
        #     return self.speed*torch.tensor([[0, 1], [-1, 0]])
        # self.jacobian = jacobian

    def _f(self, u):
        x = u[...,0:1]
        y = u[...,1:2]
        dx_dt = y
        dy_dt = -x
        return (dx_dt, dy_dt)


class LotkaVolterra(ODE):
    # The Lotka-Volterra equations.
    def __init__(self, speed = 1., weight = 1., weight_mean = 0., weight_primal=0.):
        self.name = 'Lotka-Volterra'
        self.order = 2
        self.speed = speed
        self.weight = weight
        self.weight_mean = weight_mean
        self.weight_primal = weight_primal

        self.equilibrium = torch.tensor([0., 0.])
        # def jacobian(p):
        #     x = p[...,0]+1.0
        #     y = p[...,1]+1.0
        #     #return self.speed*torch.tensor([[0, -1], [1, 0]])
        #     return self.speed*torch.tensor([[1-y, -x], [y, x-1]])
        # self.jacobian = jacobian

    def _f(self, u):
        x = u[...,0:1]+1.0
        y = u[...,1:2]+1.0
        dx_dt = x - x * y
        dy_dt = x * y - y
        return (dx_dt,dy_dt)


class FitzHughNagumo(ODE):
    # The Van der Pol oscillator.
    def __init__(self, speed = 1., weight = 1., weight_mean = 0., weight_primal=0., mu=1.0):
        self.name = 'FitzHugh-Nagumo'
        self.order = 2
        self.speed = speed
        self.weight = weight
        self.weight_mean = weight_mean
        self.weight_primal = weight_primal
        self.mu = mu

        self.equilibrium = torch.tensor([0., 0.])
        # def jacobian(p):
        #     x = p[...,0]
        #     y = p[...,1]
        #     return self.speed*torch.tensor([[(1-x*x), -1], [1/mu, 0]])
        # self.jacobian = jacobian

    def _f(self, u):
        x = u[...,0:1]
        y = u[...,1:2]
        mu = self.mu
        dx_dt = (x - x*x*x/3 - y)
        dy_dt = x/mu
        return (dx_dt,dy_dt)
    
    
class HodgkinHuxley(ODE):
    # The Hodgkin-Huxley model.
    def __init__(self, speed = 1., weight = 1.,
                 # Average potassium channel conductance per unit area (mS/cm^2)
                 gK = 36.0,
                # Average sodoum channel conductance per unit area (mS/cm^2)
                gNa = 120.0,
                # Average leak channel conductance per unit area (mS/cm^2)
                gL = 0.3,
                # Membrane capacitance per unit area (uF/cm^2)
                Cm = 1.0,
                # Potassium potential (mV)
                VK = -12.0,
                # Sodium potential (mV)
                VNa = 115.0,
                # Leak potential (mV)
                Vl = 10.613,
                # Current
                I = 100.0):
        self.name = 'Hodgkin-Huxley'
        self.order = 4
        self.speed = speed

        self.gK = gK
        self.gNa = gNa
        self.gL = gL
        self.Cm = Cm
        self.VK = VK
        self.VNa = VNa
        self.Vl = Vl
        self.I = I

        self.equilibrium = torch.tensor([0., 0., 0., 0.])

    # Potassium ion-channel rate functions

    def alpha_n(self,Vm):
        return (0.01 * (10.0 - Vm)) / (torch.exp(1.0 - (0.1 * Vm)) - 1.0)

    def beta_n(self,Vm):
        return 0.125 * torch.exp(-Vm / 80.0)

    # Sodium ion-channel rate functions

    def alpha_m(self,Vm):
        return (0.1 * (25.0 - Vm)) / (torch.exp(2.5 - (0.1 * Vm)) - 1.0)

    def beta_m(self,Vm):
        return 4.0 * torch.exp(-Vm / 18.0)

    def alpha_h(self,Vm):
        return 0.07 * torch.exp(-Vm / 20.0)

    def beta_h(self,Vm):
        return 1.0 / (torch.exp(3.0 - (0.1 * Vm)) + 1.0)
  
    # n, m, and h steady-state values

    def n_inf(self,Vm=0.0):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))

    def m_inf(self,Vm=0.0):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))

    def h_inf(self,Vm=0.0):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))


    def _f(self, u):
        Vm = u[...,0:1]
        n = u[...,1:2]
        m = u[...,2:3]
        h = u[...,3:4]
  
        # dVm/dt
        GK = (self.gK / self.Cm) * torch.pow(n, 4.0)
        GNa = (self.gNa / self.Cm) * torch.pow(m, 3.0) * h
        GL = self.gL / self.Cm
        
        dVm = (self.I / self.Cm) - (GK * (Vm - self.VK)) - (GNa * (Vm - self.VNa)) - (GL * (Vm - self.Vl))
        
        # dn/dt
        dn = (self.alpha_n(Vm) * (1.0 - n)) - (self.beta_n(Vm) * n)
        
        # dm/dt
        dm = (self.alpha_m(Vm) * (1.0 - m)) - (self.beta_m(Vm) * m)
        
        # dh/dt
        dh = (self.alpha_h(Vm) * (1.0 - h)) - (self.beta_h(Vm) * h)
    
        return (dVm,dn,dm,dh)


class RescaledHodgkinHuxley(ODE):
    # The Hodgkin-Huxley model.
    def __init__(self, speed = 1., weight = 1.,
                 # Average potassium channel conductance per unit area (mS/cm^2)
                 gK = 36.0,
                # Average sodoum channel conductance per unit area (mS/cm^2)
                gNa = 120.0,
                # Average leak channel conductance per unit area (mS/cm^2)
                gL = 0.3,
                # Membrane capacitance per unit area (uF/cm^2)
                Cm = 1.0,
                # Potassium potential (mV)
                VK = -12.0,
                # Sodium potential (mV)
                VNa = 115.0,
                # Leak potential (mV)
                Vl = 10.613,
                # Current
                I = 100.0):
        self.name = 'Hodgkin-Huxley'
        self.order = 4
        self.speed = speed

        self.gK = gK
        self.gNa = gNa
        self.gL = gL
        self.Cm = Cm
        self.VK = VK
        self.VNa = VNa
        self.Vl = Vl
        self.I = I

        self.equilibrium = torch.tensor([0., 0., 0., 0.])

    # Potassium ion-channel rate functions

    def alpha_n(self,Vm):
        return (0.01 * (10.0 - Vm)) / (torch.exp(1.0 - (0.1 * Vm)) - 1.0)

    def beta_n(self,Vm):
        return 0.125 * torch.exp(-Vm / 80.0)

    # Sodium ion-channel rate functions

    def alpha_m(self,Vm):
        return (0.1 * (25.0 - Vm)) / (torch.exp(2.5 - (0.1 * Vm)) - 1.0)

    def beta_m(self,Vm):
        return 4.0 * torch.exp(-Vm / 18.0)

    def alpha_h(self,Vm):
        return 0.07 * torch.exp(-Vm / 20.0)

    def beta_h(self,Vm):
        return 1.0 / (torch.exp(3.0 - (0.1 * Vm)) + 1.0)
  
    # n, m, and h steady-state values

    def n_inf(self,Vm=0.0):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))

    def m_inf(self,Vm=0.0):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))

    def h_inf(self,Vm=0.0):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))


    def _f(self, u):
        Vm = u[...,0:1]
        n = u[...,1:2]
        m = u[...,2:3]
        h = u[...,3:4]

        Vm = Vm*10
        n = n/10
        m = m/10
        h = h/10
  
        # dVm/dt
        GK = (self.gK / self.Cm) * torch.pow(n, 4.0)
        GNa = (self.gNa / self.Cm) * torch.pow(m, 3.0) * h
        GL = self.gL / self.Cm
        
        dVm = ((self.I / self.Cm) - (GK * (Vm - self.VK)) - (GNa * (Vm - self.VNa)) - (GL * (Vm - self.Vl)))/10
        
        # dn/dt
        dn = ((self.alpha_n(Vm) * (1.0 - n)) - (self.beta_n(Vm) * n))*10
        
        # dm/dt
        dm = ((self.alpha_m(Vm) * (1.0 - m)) - (self.beta_m(Vm) * m))*10
        
        # dh/dt
        dh = ((self.alpha_h(Vm) * (1.0 - h)) - (self.beta_h(Vm) * h))*10
    
        return (dVm,dn,dm,dh)
    

class HindmarshRose(ODE):
    # The Hindmarsh-Rose model.
    def __init__(self, speed = 1., weight = 1., weight_mean = 0., weight_primal=0., a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.001, s = 1.0, xR = -8.5, I = 1.0):
        self.name = 'Hindmarsh-Rose'
        self.order = 3
        self.speed = speed
        self.weight = weight
        self.weight_mean = weight_mean
        self.weight_primal = weight_primal

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.s = s
        self.xR = xR
        self.I = I

        self.equilibrium = torch.tensor([0., 0., 0.])

    def _f(self, u):
        x = u[...,0:1]
        y = u[...,1:2]
        z = u[...,2:3]
        dx_dt = y - self.a*x*x*x + self.b*x*x - z + self.I
        dy_dt = self.c - self.d*x*x - y
        dz_dt = self.r*(self.s*(x - self.xR) - z)
        return (dx_dt,dy_dt,dz_dt)