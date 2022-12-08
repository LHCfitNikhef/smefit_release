# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\Xi
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
ctG
ccp
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpd
cpD
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
cpD=(-12311*ctp)/(4319.*Sqrt(2))
cpd=(-12311*ctp)/(17276.*Sqrt(2))
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cbp >= 0
ctp >= 0
ctap >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cpd <= 0
cpD <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
(12311*ctp*Power(m,2))/(8638.*Sqrt(2))
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = kXi
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(kXi)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = (np.emath.sqrt(2.8504283398934938)*np.abs(np.emath.sqrt(ctp)*m**2))/2**0.75
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
kXi = -((np.emath.sqrt(2.8504283398934938)*np.emath.sqrt(ctp)*m**2)/2**0.75)
