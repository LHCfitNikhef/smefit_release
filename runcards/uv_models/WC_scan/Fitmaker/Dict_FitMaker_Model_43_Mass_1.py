# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
U
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to set to zero
cpG
cpB
cpW
cpWB
cpd
cpD
cWWW
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
ctG
ccp
ctap
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpqMi
cpQM
c3pq
c3pQ3
ctp
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctp=(-17276*Sqrt(2)*c3pQ3)/12311.
c3pq=c3pQ3
cpQM=-2*c3pQ3
cpqMi=-2*c3pQ3
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cpqMi >= 0
cpQM >= 0
ctp >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
c3pq <= 0
c3pQ3 <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
-4*c3pQ3
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = lamUf1
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(lamUf1)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = 2*np.abs(np.emath.sqrt(c3pQ3)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
lamUf1 = (0-2J)*np.emath.sqrt(c3pQ3)*m