# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\Sigma
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
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpl1
cpl2
cpl3
c3pl1
c3pl2
c3pl3
ctap
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctap=(356*Sqrt(2)*c3pl3)/12311.
c3pl2=c3pl3
c3pl1=c3pl3
cpl3=3*c3pl3
cpl2=3*c3pl3
cpl1=3*c3pl3
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cpl1 >= 0
cpl2 >= 0
cpl3 >= 0
c3pl1 >= 0
c3pl2 >= 0
c3pl3 >= 0
ctap >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
16*c3pl3
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = lamSigmaf1
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(lamSigmaf1)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = 4*np.abs(np.emath.sqrt(c3pl3)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
lamSigmaf1 = -4*np.emath.sqrt(c3pl3)*m
