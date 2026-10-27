"""Comprobación independiente mínima del proyecto (sin Gymnasium).
Ejecutar desde la raíz del ZIP: python verificacion_independiente.py
No sustituye los 32 ensayos históricos ni reejecuta los 400 entrenamientos.
"""
from __future__ import annotations
import ast
import json
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "codigo_original"))
from sarsa import epsilon_greedy, episodio_sarsa
from q_learning import episodio_q_learning

def malla_desde_codigo_original():
    arbol = ast.parse((HERE / "codigo_original/env.py").read_text(encoding="utf-8"))
    for n in arbol.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="GRID_WORLD" for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError("No se encontró GRID_WORLD en env.py")

def construir_P():
    grid=malla_desde_codigo_original()
    estados=[(r,c) for r, fila in enumerate(grid) for c, k in enumerate(fila) if k=="F"]
    assert len(estados)==20 and len(grid)==7 and all(len(row)==5 for row in grid)
    index={rc:i for i, rc in enumerate(estados)}
    P={}
    for s,(r,c) in enumerate(estados):
        P[s]=[]
        for a in (0,1):
            if c==4:
                dest=(min(r+1,6),c)
            else:
                dest=(max(0,min(6,r+(-1 if a==0 else 1))),c+1)
            typ=grid[dest[0]][dest[1]]
            rr=5 if typ=="G" else -5 if typ=="X" else -1
            P[s].append((index.get(dest,20),rr,typ!="F"))
    return P

def bellman(P,epsilon):
    q=np.zeros((21,2))
    for _ in range(60):
        nq=np.zeros_like(q)
        for s in range(20):
            for a,(dest,r,term) in enumerate(P[s]):
                nq[s,a]=r if term else r+0.9*((1-epsilon)*q[dest].max()+epsilon*q[dest].mean())
        if np.max(abs(nq-q))<1e-13:
            return nq
        q=nq
    raise AssertionError("Bellman no converge")

def tabla_del_informe():
    text=(HERE/"tables/tab_qstar.tex").read_text(encoding="utf-8")
    vals=np.zeros((20,2));indices=set()
    for line in text.splitlines():
        m=re.search(r'\$s_\{(\d+)\}\s*\$',line)
        if not m: continue
        s=int(m.group(1)); parts=line.split('&')
        assert len(parts)>=3
        for a,x in enumerate(parts[1:3]):
            x=x.replace("$-$","-").replace("$","").strip()
            vals[s,a]=float(x)
        indices.add(s)
    assert indices==set(range(20)),f"Faltan estados {set(range(20))-indices}"
    return vals

class Scripted:
    def __init__(self): self.u=iter([.9,.1]);self.n=iter([1])
    def random(self): return next(self.u)
    def integers(self,*args): return next(self.n)

class Toy:
    class Unwrapped: r_goal=5.
    unwrapped=Unwrapped()
    def __init__(self):self.s=0
    def reset(self,**kw):self.s=0;return 0,{}
    def step(self,action):
        if self.s==0:
            self.s=1;return 1,-1.,False,False,{}
        self.s=2;return 2,5.,True,False,{}

def verificar_actualizaciones():
    for fn, esperado in ((episodio_sarsa,-5.),(episodio_q_learning,4.)):
        q=np.zeros((3,2));q[1]=[10.,-10.]
        result=fn(Toy(),q,.5,.9,.5,Scripted())
        assert np.isclose(q[0,0],esperado)
        assert np.isclose(q[1,1],-2.5) and q[1,0]==10 and q[0,1]==0
        assert np.all(q[2]==0) and result==(4.,2,True)
    picks=np.array([epsilon_greedy(np.array([[1.,2.]]),0,.2,np.random.default_rng(i)) for i in range(2000)])
    assert .85 < (picks==1).mean() < .95

if __name__=="__main__":
    P=construir_P()
    qstar=bellman(P,0.)
    qeps=bellman(P,.2)
    vals=tabla_del_informe()
    assert np.allclose(vals,qstar[:20],atol=5e-5)
    assert np.allclose(qstar[6],[2.15,5.])
    assert np.allclose(qstar[13],[2.15,-5.5])
    assert np.allclose(qeps[13],[1.2995,-5.5855])
    d=json.loads((HERE/"evidencias/numbers.json").read_text(encoding="utf-8"))
    assert abs(d["s6_s13"]["sarsa"]["tail_mean_s13"][0]-qeps[13,0]) < .01
    assert abs(d["s6_s13"]["q_learning"]["tail_mean_s13"][0]-qstar[13,0]) < 1e-3
    verificar_actualizaciones()
    print("PASS: 40 valores q*, q_epsilon(s13), evidencia basal resumida y targets SARSA/Q-learning.")
    print("Nota: no reejecuta el entrenamiento original ni las 32 pruebas históricas.")
