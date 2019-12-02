import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import pandas as pd
#Reading data
week=4
data1=pd.read_csv("July_Week1_Python.csv",sep=";")
data2=pd.read_csv("July_Week2_Python.csv",sep=";")
data3=pd.read_csv("July_Week3_Python.csv",sep=";")
data4=pd.read_csv("July_Week4_Python.csv",sep=";")
y1=np.array(data1["Streams"])[0:191]
y2=np.array(data2["Streams"])[0:191]
y3=np.array(data3["Streams"])[0:191]
y4=np.array(data4["Streams"])[0:191]
data1=data1.drop(columns="Streams", axis=0)
data2=data2.drop(columns="Streams", axis=0)
data3=data3.drop(columns="Streams", axis=0)
data4=data4.drop(columns="Streams", axis=0)
rel1=np.array(data1["Release_Date"])[0:191]
rel2=np.array(data2["Release_Date"])[0:191]
rel3=np.array(data3["Release_Date"])[0:191]
rel4=np.array(data4["Release_Date"])[0:191]
data1=data1.drop(columns="Release_Date", axis=0)
data2=data2.drop(columns="Release_Date", axis=0)
data3=data3.drop(columns="Release_Date", axis=0)
data4=data4.drop(columns="Release_Date", axis=0)

data1=np.array(data1)
data2=np.array(data2)[0:191]
data3=np.array(data3)[0:191]
data4=np.array(data4)[0:191]
rel=np.concatenate([rel1,rel2,rel3,rel4])
rel=np.reshape(rel,(week,191,1))
y=np.concatenate([y1,y2,y3,y4])
y=np.reshape(y,(week,191,1))
data=np.concatenate([data1,data2,data3,data4])
data=np.reshape(data,(week,191,33))


#Indices
H=week #Number of weeks
S=len(data[0]) #Number of samples
M=len(data[0][1]) #Number of features
R=6 #Number of regions
nsong=10
#Indices and Sets
h=[i for i in range(0,H)] #Set of weeks w = {1..W}
s=[i for i in range(0,S)] #Set of samples s = {1..S}
m=[i for i in range(0, M)] #Set of features m = {1..M}
r=[i for i in range(0,R)] #Set of regions r={1..R}
r1=[i for i in range(1,R)] #r1={2..R}
r2=[i for i in range(0,R-1)] #r1={1..R-1}
#
smh = [(j,k,l) for j in s for k in m for l in h]
sh = [(j,l) for j in s for l in h]
mh = [(k,l) for k in m for l in h]
rh = [(i,l) for i in r for l in h]
rsh=[(i,j,l) for i in r for j in s for l in h]
mm=[k for k in m]
rr=[i for i in r]
#Parameters
U=int(max(max(rel[0]),max(rel[1]),max(rel[2]),max(rel[3])))+1 #Arbitraryly large positive number
U2=float(2*max(max(y[0]),max(y[1]),max(y[2]),max(y[3]))) #Arbitraryly large positive number

A = {(j,k,l):float(data[l,j,k]) for j,k,l in smh} #Numeric value of sample s on feature m
Y = {(j,l):float(y[l,j]) for j,l in sh} #Output variable
Rel = {(j,l):int(rel[l,j]) for j,l in sh} #Release Date
V=0.05
mdl = Model('OPLRA_4_Weeks')

W=mdl.continuous_var_dict(mh,name="W",lb=-9) #W[m,h]:regression coefficient for feature m at week h
B=mdl.continuous_var_dict(rh,name="R",lb=-9) #B[r,h]: Intercept of regression in region r at week h
Pred=mdl.continuous_var_dict(rsh,name="Pred") #Pred[r,s,h]: Predicted output for sample s in region r at week h
X=mdl.continuous_var_dict(rh,name="X") #X[r,h]: Break point r at week h
D=mdl.continuous_var_dict(sh,name="D") #D[s,h]: Training error for sample s at week h
F=mdl.binary_var_dict(rsh,name="F") #F[r,s,h]: 1 if sample s falls into region r at week h
Max=mdl.continuous_var_dict(mm,name="Max")
Min=mdl.continuous_var_dict(mm,name="Min")
BMax=mdl.continuous_var_dict(rr,name="BMax")
BMin=mdl.continuous_var_dict(rr,name="BMin")

objective=mdl.sumsq(D[j,l] for j in s for l in h)
mdl.set_objective("min",objective)
#Constraints
mdl.add_constraints(X[i-1,l]+0.1<=X[i,l] for i in r1 for l in h)#1
mdl.add_constraints(X[i-1, l] - U*(1-F[i,j,l])+0.1<=Rel[j,l] for i in r1 for j in s for l in h)#2
mdl.add_constraints(Rel[j,l]<=X[i,l]+U*(1-F[i,j,l]) for i in r2 for j in s for l in h)#3
mdl.add_constraints(mdl.sum(F[i,j,l] for i in r)==1 for j in s for l in h)#4
mdl.add_constraints(mdl.sum(F[i,j,l] for j in s)>=nsong for i in r for l in h)#4
mdl.add_constraints(mdl.sum(W[k,l]*A[j,k,l] for k in m) + B[i,l] == Pred[i,j,l] for i in r for j in s for l in h)#5
mdl.add_constraints(Y[j,l]-Pred[i,j,l]-U2*(1-F[i,j,l])<=D[j,l] for i in r for j in s for l in h)
mdl.add_constraints((Pred[i,j,l]-Y[j,l]-U2*(1-(F[i,j,l]))<=D[j,l]) for i in r for j in s for l in h)
mdl.add_constraints(X[R-1,l]==2000 for l in h)#MAX RELEASE

mdl.add_constraints(W[k,l]<=Max[k] for l in h for k in m)
mdl.add_constraints(W[k,l]>=Min[k] for l in h for k in m)
mdl.add_constraints(B[i,l]<=BMax[i] for i in r for l in h)
mdl.add_constraints(B[i,l]<=BMax[i] for i in r for l in h)

mdl.add_constraints(BMax[i]-BMin[i]<=V for i in r)
mdl.add_constraints(Max[k]-Min[k]<=V for k in m)
solution=mdl.solve(log_output=True)
#print(solution.solve_status) #if it says feasible, it is not optimal
#mdl.solve_details

F_values=np.zeros((R,S))
Pred_values=np.zeros((R,S))
X_values=np.zeros((R))
B_values=np.zeros((R))
W_values=np.zeros((M))
D_values=np.zeros((S))
for i in r:
    X_values[i]=X[i].solution_value
    B_values[i]=B[i].solution_value
    for j in s:
        F_values[i,j]=F[i,j].solution_value
        Pred_values[i,j]=Pred[i,j].solution_value
for k in m:
    W_values[k]=W[k].solution_value
for j in s:
    D_values[j]=D[j].solution_value

R_total=np.zeros((R))
for i in r:
    R_total[i]=sum(F_values[i])

print("Objective Function Value:")
print(solution.get_objective_value())
