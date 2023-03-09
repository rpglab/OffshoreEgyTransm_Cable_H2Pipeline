### Run the simulations for three cases: HVDC case, Hybrid case, and HP Case.
### Assume hydrogen line capital cost is c, annual operation cost is 7%c
## Author: Jin Lu, University of Houston.
## link: https://rpglab.github.io/resources/

### Import
from pyomo.environ import *
import os
import numpy as np
from HVDC_Case_Model import *
from Hybrid_Case_Model import *
from HP_Case_Model import *

### function to solve the pyomo model
def solve_model(pym_model,savefl_nm):
    solver = SolverFactory('knitro',executable='C:\\Users\\lujin\\OneDrive - University Of Houston\\work folder\\ampl new\\ampl_mswin64\\knitro.exe')
    #solver.options.mipgap = 0.005
    results = solver.solve(pym_model)
    # display solution
    print("\nresults.Solution.Status: " + str(results.Solution.Status))
    print("\nresults.solver.status: " + str(results.solver.status))
    print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
    print("\nresults.solver.termination_message: " + str(results.solver.termination_message) + '\n')
    # write result
    if savefl_nm=='results_HVDC':
        write_result_nhydg(pym_model, savefl_nm)
    elif savefl_nm=='results_Hybrid':
        write_result_pchydg(pym_model, savefl_nm)
    elif savefl_nm == 'results_HP':
        write_result_hchydg(pym_model, savefl_nm)

### The input data were provided/investigated by Jesus Silva-Rodriguez except for wind power data (by Jin Lu).
# LMP ($/KWh), wind power (KW)
## following multipliers are used in sensitivity analysis
dist_mul = 1000/387 # multiplier for transmission line distance
EFcst_mul = 0.5 # multiplier for electrolyzer cost
EFeff_mul = 2.12   # multiplier for electrolyzer efficiency
hlc_mul = 0.5   # multiplier for hydrogen line cost
p_mul = 500 # multiplier for wind farm capacity
lmp_mul = 0.5   # multiplier for locational marginal price
hstrg_mul = 2   # multiplier for hydrogen storage
## grid configuration data
grid_config = {'w_tnum':3,'dist_wf_sub':[91*dist_mul,104*dist_mul,192*dist_mul]}
## cost data
cost_data = {'elct_cap':800*EFcst_mul,'elct_op':16*EFcst_mul,'fc_cap':1180*EFcst_mul,'fc_op':15.34*EFcst_mul,
             'pline_cap':2.02*(10**6),'pline_op':10100,'hline_cap':0.96*(10**6)*hlc_mul,'hline_op':67.200*hlc_mul,
             'hstrg_cap':55.5*hstrg_mul,'hstrg_op':44783*hstrg_mul,'cvt_wf_cap':220,'cvt_sub_cap':92,
             'hpcmp_cap':4717,'lpcmp_cap':4717}
## size/capacity data
size_data = {'elct_p':250,'fc_p':250,'pline_p':1000*(10**3),'hline_h':30000,
             'hstrg_min':0,'hstrg_max':202500*hstrg_mul}
## other data
other_data = {'elct_cp':53,'fc_cp':20*EFeff_mul,'chp_cp':3.3,'clp_cp':3.3,'y_tnum':30,'cvt_cp':0.99}
flow_data = {'vo':320, 'r':0.011}
## low pressure hydrogen pipeline data
cost_data['lphl_cap']=0.96*(10**6)*hlc_mul  # low pressure hydrogen line cost data
cost_data['lphl_op']=67200*hlc_mul
size_data['lphl_h']=30000
## hourly locational marginal price at onshore substation
LMP_24h = [61.22,56.55,53.32,53.07,53.65,53.25,58.27,61.16,61.88,61.03,
    64.6,66.59,87.94,97.65,176.06,229.99,255.01,210.65,199.21,218.59,
    212.36,81.27,82.59,77.34]
LMP_24h = [p/1000*lmp_mul for p in LMP_24h] # unit from $/MWh to $/kWh
other_data['LMP'] = LMP_24h
## wind farm hourly generation profile
# hourly generation coefficients
wind_chg = [0.715110621,0.825731525,0.733212223,0.70842795,0.748394213,0.742165704,0.744112113,0.764419646,0.582884578,0.434633102,0.314020632,0.309479011,0.310776617,0.313955752,0.197560501,0.358204113,0.946279115,0.4955557,0.456887043,0.792318173,0.845065854,0.907091416,0.966002725,1]
wf1_p = list(np.zeros((24)))
wf2_p = list(np.zeros((24)))
wf3_p = list(np.zeros((24)))
# Each wind farm's capacity is 3.6MW
for h in range(24):
    wf1_p[h] = 3636590.72*wind_chg[h]
    wf2_p[h] = 3636590.72 * wind_chg[h]
    wf3_p[h] = 3636590.72 * wind_chg[h]
# apply the multiplier
wf1_p = [p/(10**3)*p_mul for p in wf1_p]  # unit from W to KW, assume 100 turbines in one farm
wf2_p = [p/(10**3)*p_mul for p in wf2_p]
wf3_p = [p/(10**3)*p_mul for p in wf3_p]
wf_p = [wf1_p,wf2_p,wf3_p]
other_data['wf_p'] = wf_p

# Additional data for hybrid case
# distance from wind farm to hydrogen super center
grid_config['dist_wf_hsc'] = [69.3*dist_mul,84.1*dist_mul,140*dist_mul]
# distance from hydrogen super center to onshore substation
grid_config['dist_hsc_sub'] = 55.9*dist_mul
# data for flow equation
flow_data['ps_sub'] = 200
flow_data['lambda'] = 0.02
flow_data['T'] = 300
flow_data['z_h'] = 1.133
flow_data['tho_h'] = 38.642
flow_data['D_h'] = 1219.2

# Additional data for HP case
flow_data['ps_hsc'] = 100
flow_data['z_l'] = 1.067
flow_data['tho_l'] = 20.25
flow_data['D_l'] = 914.4

### Main
nhydg_model = build_pym_nhydg(grid_config, cost_data, size_data, flow_data, other_data)
solve_model(nhydg_model,'results_HVDC')
pchydg_model = build_pym_pchydg(grid_config,cost_data,size_data, flow_data, other_data)
solve_model(pchydg_model,'results_Hybrid')
hchydg_model = build_pym_hchydg(grid_config,cost_data,size_data, flow_data, other_data)
solve_model(hchydg_model,'results_HP')