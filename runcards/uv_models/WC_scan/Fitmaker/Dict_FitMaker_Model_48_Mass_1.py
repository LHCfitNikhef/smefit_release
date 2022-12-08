# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\text{T1}
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
cbp
ctp
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
1
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctp=(17276*cbp)/209.
c3pQ3=(12311*cbp)/(418.*Sqrt(2))
c3pq=(12311*cbp)/(418.*Sqrt(2))
cpQM=(-12311*Sqrt(2)*cbp)/209.
cpqMi=(-12311*Sqrt(2)*cbp)/209.
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
c3pq >= 0
c3pQ3 >= 0
cbp >= 0
ctp >= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
cpqMi <= 0
cpQM <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
(49244*Sqrt(2)*cbp)/209.
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = lamT1f1
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(lamT1f1)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = 2*2**0.25*np.emath.sqrt(58.90430622009569)*np.abs(np.emath.sqrt(cbp)*m)
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
lamT1f1 = -2*2**0.25*np.emath.sqrt(58.90430622009569)*np.emath.sqrt(cbp)*m
