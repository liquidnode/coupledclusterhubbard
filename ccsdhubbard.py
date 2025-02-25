import wicked as w
import time
import numpy as np
import torch
import sys

#contains code from https://github.com/fevangelista/wicked/tree/main
def generate_equation(mbeq, nocc, nvir):
    res_sym = f"r{'o' * nocc}{'v' * nvir}"
    code = [
        f"def evaluate_residual_{nocc}_{nvir}(f,v,T):",
        "    # contributions to the residual",
    ]

    if nocc + nvir == 0:
        code.append("    r = 0.0")
    else:
        dims = ",".join(["nocc"] * nocc + ["nvir"] * nvir)
        code.append(f"    {res_sym} = 0.0")

    for eq in mbeq["o" * nocc + "|" + "v" * nvir]:
        contraction = eq.compile("einsum")
        code.append(f"    {contraction}")

    code.append(f"    return {res_sym}")
    funct = "\n".join(code)
    return funct.replace("np.","torch.").replace(",optimize=\"optimal\"","")

w.reset_space()
w.add_space('o','fermion','occupied',['i','j','k','l','m','n'])
w.add_space('v','fermion','unoccupied',['a','b','c','d','e','f'])

F = w.utils.gen_op('f',1,'ov','ov')
V = w.op("v", ["o+ v+ v o"])
H = F + V

def make_T(n):
    components = [f"{'v+' * k} {'o' * k}" for k in range(1,n + 1)]
    return w.op("T",components)

def cc_equations(n):
    start = time.perf_counter()
    wt = w.WickTheorem()

    T = make_T(n)
    Hbar = w.bch_series(H,T,4)
    expr = wt.contract(w.rational(1), Hbar, 0, 2 * n)
    mbeq = expr.to_manybody_equation("r")
    end = time.perf_counter()    
    t = end - start
    
    equations = {}
    for r in range(0,n + 1):
        s = f"{'o' * r}|{'v' * r}" 
        equations[r] = (mbeq[s])      
        
    return equations, t, mbeq

equations, t, mbeq = cc_equations(2)

#setup CCSD equations
print(generate_equation(mbeq, 0, 0))
exec(generate_equation(mbeq, 0,0))
print(generate_equation(mbeq, 1, 1))
exec(generate_equation(mbeq, 1, 1))
print(generate_equation(mbeq, 2, 2))
exec(generate_equation(mbeq, 2, 2))

#setup hamiltonian
Nx = int(sys.argv[1])
Ny = int(sys.argv[1])
D = 2
nocc = (Nx * Ny) * (D - 1)
nvir = (Nx * Ny)
f = {"vv": torch.zeros((nvir, nvir), requires_grad=False), "ov": torch.zeros((nocc, nvir), requires_grad=False), "vo": torch.zeros((nvir, nocc),requires_grad=False), "oo": torch.zeros((nocc, nocc), requires_grad=False)}
v = {"ovov": torch.zeros((nocc, nvir, nocc, nvir), requires_grad=False)}

t = -1.0
U = float(sys.argv[2])
mu = 0.0
deltas = [
    np.array([1,0]),
    np.array([0,1])
]

def getov(x,y,p):
    if p >= 2:
        return "o"
    return "o" if (x + y + p) % 2 == 0 else "v"

#setup the Hilbert space
sitetoindex = {}
occi = 0
virti = 0
for x in range(Nx):
    for y in range(Ny):
        for p in range(D):
            if getov(x,y,p) == "o":
                sitetoindex[(x,y,p)] = occi
                occi += 1
            else:
                sitetoindex[(x,y,p)] = virti
                virti += 1

def antisymmetrize_residual_2_2(Roovv):
    # antisymmetrize the residual
    Roovv_anti = 0.0
    Roovv_anti += torch.einsum("ijab->ijab", Roovv)
    Roovv_anti -= torch.einsum("ijab->jiab", Roovv)
    Roovv_anti -= torch.einsum("ijab->ijba", Roovv)
    Roovv_anti += torch.einsum("ijab->jiba", Roovv)
    return Roovv_anti / 4.0

#all tensors must be antisymmetric with respect to all cdag indices
#and all c indices (f_ij must not be antisymmetrized, since it involves only 1 index per cdag and c)

#create one particle and two particle terms of the Hamiltonian
for x in range(Nx):
    for y in range(Ny):
        s1i = sitetoindex[(x,y,0)]
        s2i = sitetoindex[(x,y,1)]
        s1o = getov(x,y,0)
        s2o = getov(x,y,1)
        if s1o == "v":
            v["ovov"][s2i,s1i,s2i,s1i] = U
        else:
            v["ovov"][s1i,s2i,s1i,s2i] = U
        for p in range(2):
            #make hopping
            site1ov = getov(x,y,p)
            site1i = sitetoindex[(x,y,p)]
            for d in deltas:
                if Nx == 1 or Ny == 1:
                    nx = x + d[0]
                    if nx >= Nx:
                        continue
                else:
                    nx = (x + d[0]) % Nx
                if Ny == 1 or Nx == 1:
                    ny = y + d[1]
                    if ny >= Ny:
                        continue
                else:
                    ny = (y + d[1]) % Ny
                site2ov = getov(nx,ny,p)
                site2i = sitetoindex[(nx,ny,p)]
                f[site1ov+site2ov][site1i,site2i] += t
                f[site2ov+site1ov][site2i,site1i] += t
            f[site1ov+site1ov] += mu
            # need this because cdagup cup cdagdown cdown = -:cdagup cdagdown cup cdown: + :cdagup cup:
            # if down is fully occupied in groundstate
            #https://theorie.ikp.physik.tu-darmstadt.de/tnp/pub/2013_gebrerufael_master.pdf
            if site1ov == "v" and p < 2:
                f[site1ov+site1ov][site1i,site1i] += U


##setup the optimization parameters
T = {"ov": torch.zeros((nocc, nvir), requires_grad=True), "oovv": torch.zeros((nocc, nocc, nvir, nvir), requires_grad=True)}
Lambda = {"ov": torch.zeros((nocc, nvir), requires_grad=True), "oovv": torch.zeros((nocc, nocc, nvir, nvir), requires_grad=True)}
E_0 = 0.0
with torch.no_grad():
    T["ov"].uniform_(-0.001,0.001)
    T["oovv"].uniform_(-0.001,0.001)

s = 20.0
seeklim = 8.0 * (Nx * Ny) / (36.0) # if the residual is smaller than a threshold, minimize the residual |R|^2 directly

def getER():
    R = {}
    Tr = {}
    Tr["ov"] = T["ov"]
    Tr["oovv"] = antisymmetrize_residual_2_2(T["oovv"])
    Ecorr_w = evaluate_residual_0_0(f, v, Tr)
    Etot_w = Ecorr_w + E_0
    R["ov"] = evaluate_residual_1_1(f, v, Tr)
    R["oovv"] = evaluate_residual_2_2(f, v, Tr)
    R["oovv"] = antisymmetrize_residual_2_2(R["oovv"])
    return Etot_w, R

def getloss():
    Etot_w, R = getER()

    if rrloss > seeklim:
        loss = Etot_w.clone()
    else:
        loss = 0.0
    residum = 0.0
    rloss = 0.0
    for r in R:
        ei = list(range(len(R[r].shape)))
        if rrloss > seeklim:
            loss += torch.einsum(Lambda[r], ei, R[r], ei)
        rloss += torch.einsum(R[r], ei, R[r], ei)
    return loss + rloss, Etot_w, R, rloss

def closure():
    optim.zero_grad()
    loss, Etot_w, R, rloss = getloss()
    loss.backward()
    print('R', rloss.item(), 'energy', (Etot_w.item() / (Nx*Ny)))
    return loss

#optimize
lloss = 10000.
rrloss = 1000.0
jj = 0
while True:
    optim = torch.optim.LBFGS([T["ov"], T["oovv"]])
    while True:
        loss = optim.step(closure)
        if (lloss - loss).abs() < 1e-2 and rrloss > seeklim:
            break
        if (lloss - loss).abs() < 1e-6 and rrloss <= seeklim:
            exit()
        lloss = loss.detach().clone()
    print('Lamstep')
    with torch.no_grad():
        loss, Etot_w, R, rloss = getloss()

        for r in Lambda:
            Lambda[r].data.copy_(Lambda[r] + (1.0 / s) * (1.0 * R[r]))
        rrloss = rloss.item()
    jj += 1
        

