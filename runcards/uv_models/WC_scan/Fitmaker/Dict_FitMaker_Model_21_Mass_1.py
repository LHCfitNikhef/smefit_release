# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
B
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to set to zero
cpG
cpB
cpW
cpWB
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
cpe
cpmu
cpta
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
ctap
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpd
cpD
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
cpD=-4*cpd
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cpd >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cpD <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
2*cpd
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = gBH
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(gBH)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = np.emath.sqrt(2)*np.abs(np.emath.sqrt(cpd)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
gBH = -(np.emath.sqrt(2)*np.emath.sqrt(cpd)*m)
