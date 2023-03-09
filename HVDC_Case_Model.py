### The pyomo model & saving results function for HVDC case
## Author: Jin Lu, University of Houston.
## link: https://rpglab.github.io/resources/

### Import
from pyomo.environ import *
import os
import numpy as np

### model of HVDC case
def build_pym_nhydg(grid_config, cost_data, size_data, flow_data, other_data):
    model = AbstractModel()
    # Set
    W = list(range(1,grid_config['w_tnum']+1))
    T = list(range(1,25))
    model.W = Set(initialize=W)
    model.T = Set(initialize=T)
    # Variable
    model.p_del = Var(model.T)
    model.pl_num = Var(model.W,domain=PositiveIntegers)
    model.c_t = Var(domain=Reals)
    model.wf_pout = Var(model.W,model.T)
    # Objective
    def objfunc(model):
        obj = 365*other_data['y_tnum']*sum(other_data['LMP'][t-1]*model.p_del[t] for t in model.T)-model.c_t
        return obj
    model.object = Objective(rule=objfunc, sense=maximize)
    ## Constraint
    # capital cost constraint (*** add converter station cost)
    def consfunc_ct(model):
        eqright = (cost_data['pline_cap']+cost_data['pline_op']*other_data['y_tnum'])*\
                sum(grid_config['dist_wf_sub'][w-1]*model.pl_num[w] for w in model.W)
        eqright += sum(cost_data['cvt_wf_cap']*size_data['pline_p']*model.pl_num[w] for w in model.W)
        eqright += cost_data['cvt_sub_cap']*sum(size_data['pline_p']*model.pl_num[w] for w in model.W)
        return model.c_t == eqright
    model.cons_ct = Constraint(rule=consfunc_ct)
    # deliver power constraint  (*** add converter efficiency)
    def consfunc_pdel(model,t):
        return sum(model.wf_pout[w,t] for w in model.W)==model.p_del[t]/other_data['cvt_cp']
    model.cons_pdel = Constraint(model.T, rule=consfunc_pdel)
    # line capacity constraint  (*** wind power output replace the wind power generation)
    def consfuc_plcap(model,w,t):
        return size_data['pline_p']*model.pl_num[w]>=model.wf_pout[w,t]
    model.cons_plcap = Constraint(model.W,model.T,rule=consfuc_plcap)
    # dc power flow constraint
    def consfuc_dcpf(model, w, t):
        eq_left = other_data['wf_p'][w - 1][t - 1]*other_data['cvt_cp'] - model.wf_pout[w, t]
        eq_right = 1000 * ((model.wf_pout[w, t] / flow_data['vo']/1000) ** 2) * grid_config['dist_wf_sub'][w - 1] * \
                   flow_data['r']/model.pl_num[w]
        return eq_left == eq_right
    model.cons_dcpf = Constraint(model.W, model.T, rule=consfuc_dcpf)
    # load case data and create instance
    print('start creating the instance')
    case_pyomo = model.create_instance()
    print('finish creating the instance')
    # case_pyomo.pprint()
    return case_pyomo

### function to save the results of HVDC case
def write_result_nhydg(pym_model, savefl_nm):
    fdpath = os.getcwd() + '\\' + savefl_nm
    # Check whether the specified path exists or not
    isExist = os.path.exists(fdpath)
    if not isExist:
        os.makedirs(fdpath)  # Create a new directory because it does not exist
    # print objective value
    flpath = fdpath + '\\' + 'objvalue.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.object()))
    f.close()
    # print pdel
    flpath = fdpath + '\\' + 'p_del.txt'
    f = open(flpath, 'w')
    pdel_str = ''
    for t in pym_model.T:
            pdel_str = pdel_str + str(pym_model.p_del[t]()) + ' '
    f.write(pdel_str)
    f.close()
    # print number of power line
    flpath = fdpath + '\\' + 'pl_num.txt'
    f = open(flpath, 'w')
    plnum_str = ''
    for w in pym_model.W:
        plnum_str = plnum_str + str(pym_model.pl_num[w]()) + ' '
    f.write(plnum_str)
    f.close()
    # print power input at substation
    flpath = fdpath + '\\' + 'wf_pout.txt'
    f = open(flpath, 'w')
    wfpout_str = ''
    for w in pym_model.W:
        for t in pym_model.T:
            wfpout_str = wfpout_str + str(pym_model.wf_pout[w,t]()) + ' '
        wfpout_str += '\n'
    f.write(wfpout_str)
    f.close()