# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\text{$\Delta $3}
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to set to zero
cpG
cpB
cpW
cpWB
cpd
cpD
cWWW
cpqMi
cpQM
c3pq
c3pQ3
cpl1
cpl2
cpl3
c3pl1
c3pl2
c3pl3
cpui
cpdi
cpt
c81qq
c11qq
c83qq
c13qq
c8qt
c1qt
c8ut
c1ut
c8qu
c1qu
c8dt
c1dt
c8qd
c1qd
cQQ1
cQQ8
cQt1
cQt8
ctt1
cll
cbp
ctp
ctG
ccp
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpe
cpmu
cpta
ctap
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctap=(-89*Sqrt(2)*cpta)/12311.
cpmu=cpta
cpe=cpta
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
ctap >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cpe <= 0
cpmu <= 0
cpta <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
-2*cpta
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = lamDelta3f1
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(lamDelta3f1)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = np.emath.sqrt(2)*np.abs(np.emath.sqrt(cpta)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
lamDelta3f1 = (0-1J)*np.emath.sqrt(2)*np.emath.sqrt(cpta)*m
