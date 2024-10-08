import json 
import kappa
import sys


#create the different kappas combination
kappas0 = ["kb2","ktau2","kc2","kW2","kZ2","kg2","kgamma2","kZgamma2","kmu2","kt2"]
#kappas0_combination=kappa_combiner(kappas0)

#We read the array from an external file, is a dictionary with I (initial) and F (final) states.
dict_dataset=kappa.Exp[str(sys.argv[1])]

n=len((dict_dataset["I"])["kb2"]) #number of point in the dataset
print(f"Found {n} point")
EMPTY=[0. for _ in range(n)]

kappacombination_dict={}
I=dict_dataset["I"]
F=dict_dataset["F"]

for i_,istate in enumerate(I.keys()):
    for fstate in list(F.keys())[i_:]:
        if not(istate==fstate):
            arr1=[i_val*f_val for i_val,f_val in zip(I[istate],F[fstate])]
            arr2=[i_val*f_val for i_val,f_val in zip(I[fstate],F[istate])] #swith to consider them all if not the same label
            kappacombination_dict[istate+"*"+fstate]=[a1+a2 for a1,a2 in zip(arr1,arr2)]
        else:
            kappacombination_dict[istate+"*"+fstate]=[i_val*f_val for i_val,f_val in zip(I[istate],F[fstate])]


#create the dictionary and print it in a file

kappa_dict={}
LO_dict={}
kappa_dict["best_sm"]=EMPTY
kappa_dict["theory_cov"]=[EMPTY for _ in range(n)]
LO_dict["SM"]=EMPTY
for k in kappas0:
     LO_dict[k]=EMPTY
LO_dict.update(kappacombination_dict)
kappa_dict["LO"]=LO_dict
with open(str(sys.argv[1])+".json", "w") as outfile: 
    json.dump(kappa_dict, outfile,indent="")