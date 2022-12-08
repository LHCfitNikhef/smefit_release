# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
W
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to set to zero
cpG
cpB
cpW
cpWB
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
ctG
ccp
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpd
cbp
ctp
ctap
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctap=(89*ctp)/8638.
cbp=(209*ctp)/8638.
cpd=(-36933*ctp)/(17276.*Sqrt(2))
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cpd >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cbp <= 0
ctp <= 0
ctap <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
(-12311*Sqrt(2)*ctp)/4319.
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = gWH
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(gWH)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = 2**0.25*np.emath.sqrt(2.8504283398934938)*np.abs(np.emath.sqrt(ctp)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
gWH = (0-1J)*2**0.25*np.emath.sqrt(2.8504283398934938)*np.emath.sqrt(ctp)*m
