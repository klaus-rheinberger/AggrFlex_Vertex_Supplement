# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:30:53 2024

@author: ozem
"""

import numpy as np
import gurobipy as gp
import pandas as pd
from collections import deque

def get_x_max_alpha(S0,Smax,Pmax,dt,alpha):
    if (S0*alpha + Pmax*dt) > Smax:
        x_max = (Smax - alpha*S0)/dt #- 0.01
    else:
        x_max = Pmax
    SOC = alpha*S0 + x_max*dt
    return x_max,SOC

def get_x_min_alpha(S0,Smin,Pmin,dt,alpha):
    if (S0*alpha + Pmin*dt) < Smin:
        x_min = (Smin - alpha*S0)/dt #+ 0.01
    else:
        x_min = Pmin
    SOC = alpha*S0 + x_min*dt
    return x_min,SOC

def get_vertices(S0,Smax,Smin,Pmax,Pmin,alpha,S_f,cfg,arr):
    p_list = []
    for item in arr:
        SOC_list = [S0]
        p_elem_list = []
        for elem in item:
            if elem == 1:
                x_opt,SOC = get_x_max_alpha(SOC_list[-1],Smax,Pmax,cfg["dt"],alpha)
            else:
                x_opt,SOC = get_x_min_alpha(SOC_list[-1],Smin,Pmin,cfg["dt"],alpha)
            p_elem_list.append(x_opt)
            SOC_list.append(SOC)           
        i = 1
        if SOC < alpha**cfg["Time periods"]*S_f:
            while SOC < alpha**cfg["Time periods"]*S_f - 10**(-15): # correction # (- 10**(-15) due to numerical issues)
                p_elem_list_temp = p_elem_list.copy()
                SOC_list_temp = SOC_list.copy()
                del p_elem_list_temp[-i:]
                del SOC_list_temp[-i:]
                SOC = SOC_list_temp[-1]
                for j in range(i,1,-1):
                    x_opt = Pmax
                    SOC = alpha*SOC + Pmax*cfg["dt"]
                    p_elem_list_temp.append(x_opt)           
                    SOC_list_temp.append(SOC)
                x_opt,SOC = get_x_max_alpha(SOC,alpha**cfg["Time periods"]*S_f,Pmax,cfg["dt"],alpha)
                p_elem_list_temp.append(x_opt)
                SOC_list_temp.append(SOC)                
                i += 1
            p_elem_list = p_elem_list_temp
        p_list.append(np.array(p_elem_list))
    return np.array(p_list).T

def get_random_signals(dim, n): # random generation of vectors in {-1,1}^d
    nums = np.unique(np.random.choice([-1,1], [n, dim]),axis=1)
    while len(nums) < n:
        nums = np.unique(np.stack([nums, np.random.choice([-1,1], [n - len(nums), dim])]),axis=1)
    return nums

def costPeakReduction(A_list,b_list,edge_points,D,c,delta_t,dimension,households,opt_type):
    numb_edge_points = np.shape(edge_points)[1]
    if opt_type == 0:
        model = gp.Model("optimizaiton approximation")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape = dimension, lb = -float("inf"))
        alpha = model.addMVar(shape = numb_edge_points, ub = 1)
        model.setObjective(c@x*delta_t + c@sum(D)*delta_t, gp.GRB.MINIMIZE)
        model.addConstr(x == edge_points@alpha)
        model.addConstr(gp.quicksum(alpha) == 1)
        model.optimize()
        sol = x.X
        
        model = gp.Model("optimizaiton exact")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape = (dimension,households), lb = -float("inf"))
        model.setObjective(sum(c@(x[:,i])*delta_t + c@D[i] for i in range(households)), gp.GRB.MINIMIZE) # add demand !!!
        model.addConstrs(A_list[i]@x[:,i] <= b_list[i] for i in range(households))
        model.optimize()
        sol_exact = np.sum(x.X,axis=1)
    
    elif opt_type == 1:
        model = gp.Model("optimizaiton approximation")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape = dimension, lb = -float("inf"))
        alpha = model.addMVar(shape = numb_edge_points, ub = 1)
        t = model.addVar(lb = 0.0)
        model.setObjective(t,gp.GRB.MINIMIZE)
        model.addConstrs(-t <= x[i] + sum(D)[i] for i in range(dimension))
        model.addConstrs(x[i] + sum(D)[i] <= t for i in range(dimension))
        model.addConstr(x == edge_points@alpha)
        model.addConstr(sum(alpha) == 1)
        model.optimize()
        sol = x.X
        
        model = gp.Model("optimizaiton exact")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape = (dimension,households), lb = -float("inf"))
        t = model.addVar(lb = 0.0)
        model.setObjective(t,gp.GRB.MINIMIZE)
        for i in range(dimension):
            model.addConstr(-t <= gp.quicksum(x[i,j] + D[j][i] for j in range(households)))
            model.addConstr(gp.quicksum(x[i,j] + D[j][i] for j in range(households)) <= t)
        model.addConstrs(A_list[i]@x[:,i] <= b_list[i] for i in range(households))
        model.optimize()
        sol_exact = np.sum(x.X,axis=1)
    return sol,sol_exact

def get_Ab(cfg,batts):
    b_list = []
    A_list = []
    I = np.identity(cfg["Time periods"])
    for x_min,x_max,S0,Smax,Smin,alpha,S_f in zip(batts["x_min"],batts["x_max"],batts["S_0"],batts["S_max"],batts["S_min"],batts["alpha"],batts["S_f"]):
        v = deque([alpha**i for i in range(cfg["Time periods"])])
        v_list = [list(v)]
        for i in range(cfg["Time periods"]-1):
            v.rotate(1)
            v_list.append(list(v))
        Gamma = np.tril(np.array(v_list).T,0)
        A = np.concatenate((-I,I,Gamma,-Gamma),axis=0)
        
        v = np.array([alpha**i for i in range(1,cfg["Time periods"]+1)])
        b_1 = np.ones(cfg["Time periods"])*(-x_min)
        b_2 = np.ones(cfg["Time periods"])*(x_max)
        b_3 = (np.ones(cfg["Time periods"])*Smax-S0*np.ones(cfg["Time periods"])*v)/cfg["dt"]
        b_4 = (S0*np.ones(cfg["Time periods"]-1)*v[:-1]-Smin*np.ones(cfg["Time periods"]-1))/cfg["dt"]
        b_5 = np.array([(S0*v[-1] - S_f)/cfg["dt"]])
        b = np.concatenate((b_1,b_2,b_3,b_4,b_5))
        
        b_list.append(b)
        A_list.append(A)
    return A_list,b_list

def importData(cfg):
    my_file = "residential_IDs_file1.pickle"
    my_participants = pd.read_pickle(cfg["path HH"] + my_file)
    D_list = []
    selection = np.random.choice(len(my_participants),cfg["Households"],replace=True) # random participants choice
    for k in selection:
        my_file = f"hh_df_{my_participants[k]}.pickle"
        D = pd.read_pickle(cfg["path HH"] + my_file).values[0:cfg["Time periods"]].flatten()
        D_list.append(D)
    
    my_file = "da_df.pickle"
    my_prices = pd.read_pickle(cfg["path DA"] + my_file)
    my_prices = my_prices.to_numpy()
    my_prices = my_prices.reshape(int(len(my_prices)/96),96)/1000 # convert to EUR/kWh
    c_list = []
    for i in range(cfg["Numb Cost Vectors"]):
        c_list.append(my_prices[i,0:cfg["Time periods"]])
    return c_list,D_list

def createRandomBatteryValues(cfg):
    batts = {}
    batts["x_max"] = np.random.uniform(5.8,7.6,cfg["Households"]).tolist()
    batts["x_min"] = np.random.uniform(-7.6,-5.8,cfg["Households"]).tolist()
    batts["S_min"] = np.random.uniform(0,0,cfg["Households"]).tolist()
    batts["S_max"] = np.random.uniform(10.5,13.5,cfg["Households"]).tolist()
    batts["S_0"] = np.random.uniform(0,10.5,cfg["Households"]).tolist()
    batts["S_f"] = ((np.array(batts["S_0"]) + np.array(batts["S_min"]))/2).tolist()
    batts["alpha"] = np.random.uniform(1,1,cfg["Households"]).tolist()
    return batts
