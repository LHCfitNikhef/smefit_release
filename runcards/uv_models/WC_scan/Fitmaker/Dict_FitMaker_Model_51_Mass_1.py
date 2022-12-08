# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\text{SpTB}
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
ctap
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
cpt
cbp
ctp
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctp=(8638*cbp)/209.
cpt=(-12311*cbp)/(209.*Sqrt(2))
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
cbp >= 0
ctp >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cpt <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
(12311*cbp)/(209.*Sqrt(2)*Power(m,2))
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = sRt
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(sRt)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = (np.emath.sqrt(58.90430622009569)*np.abs(np.emath.sqrt(cbp)*v))/2**0.25
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
sRt = -((np.emath.sqrt(58.90430622009569)*np.emath.sqrt(cbp)*v)/2**0.25)
