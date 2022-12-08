# UV dictionary for FitMaker Models (2012.02779). All at tree-level.
# ~~~~~~~~~~~~~~~~~~~~~~
# Model:
\varphi
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
cQQ1
cQQ8
ctt1
cll
ctG
ccp
ctW
ctZ
# ~~~~~~~~~~~~~~~~~~~~~~
# WCs to be constrained
c8qd
c1qd
cQt1
cQt8
cbp
ctp
ctap
# ~~~~~~~~~~~~~~~~~~~~~~
# Number of L.I. WCs
2
# ~~~~~~~~~~~~~~~~~~~~~~
# Unrelated WCs
 
# ~~~~~~~~~~~~~~~~~~~~~~
# Linear Relations among WCs
ctap=(-89*ctp)/8638.
cbp=(-209*ctp)/8638.
cQt8=6*cQt1
c1qd=(43681*cQt1)/7.4615044e7
c8qd=(131043*cQt1)/3.7307522e7
# ~~~~~~~~~~~~~~~~~~~~~~
# Positively defined WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# Negatively defined WCs
c8qd <= 0
c1qd <= 0
cQt1 <= 0
cQt8 <= 0
# ~~~~~~~~~~~~~~~~~~~~~~
# Multiplicative relations among WCs
# ~~~~~~~~~~~~~~~~~~~~~~
# A possible basis of L.I. WCs
(-454682163*cQt1)/7.4615044e7
(-12311*ctp)/8638.
# ~~~~~~~~~~~~~~~~~~~~~~
# UV couplings in this model
u1 = cotBeta
u2 = Z6
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities in this model
invar1 = np.abs(cotBeta)
invar2 = (cotBeta*Z6)/np.abs(cotBeta)
# ~~~~~~~~~~~~~~~~~~~~~~
# UV coupling combinations invariant under sign ambiguities, in terms of the WCs
invar1 = (12311*np.emath.sqrt(3)*np.abs(np.emath.sqrt(cQt1)*m))/8638.
invar2 = -((ctp*m**2)/(np.emath.sqrt(6)*np.abs(np.emath.sqrt(cQt1)*m)))
# ~~~~~~~~~~~~~~~~~~~~~~
# One possible solution of the equation relating the UV couplings to the WCs.
cotBeta = (0-1.4252141699467469J)*np.emath.sqrt(3)*np.emath.sqrt(cQt1)*m
Z6 = ((0-1J)*ctp*m)/(np.emath.sqrt(6)*np.emath.sqrt(cQt1))
