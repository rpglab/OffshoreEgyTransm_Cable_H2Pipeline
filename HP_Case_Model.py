### The pyomo model & saving results function for HP case
## Author: Jin Lu, University of Houston.
## link: https://rpglab.github.io/resources/

### Import
from pyomo.environ import *
import os
import numpy as np

### model of HP case
def build_pym_hchydg(grid_config,cost_data,size_data, flow_data, other_data):
    model = AbstractModel()
    # Set
    W = list(range(1,grid_config['w_tnum']+1))
    T = list(range(1,25))
    model.W = Set(initialize=W)
    model.T = Set(initialize=T)
    # Variable
    model.p_del = Var(model.T,domain=PositiveReals)
    model.lphl_num = Var(model.W,domain=PositiveIntegers)    # lphl: low pressure hydrogen line
    model.hphl_num = Var(domain=PositiveIntegers)  # hphl: high pressure hydrogen line
    model.c_t = Var(domain=PositiveReals)
    model.elct_num = Var(model.W,domain=PositiveIntegers)
    model.fc_num = Var(domain=PositiveIntegers)
    model.h_elct = Var(model.W,model.T,domain=PositiveReals) # in case 3, electrolyzer at each wind farm
    model.h_hstrg = Var(model.T,domain=PositiveReals)
    model.h_hstrg_in = Var(model.T,domain=PositiveReals)
    model.h_fc = Var(model.T,domain=PositiveReals)
    model.p_fc = Var(model.T,domain=PositiveReals)
    model.p_c = Var(model.W,model.T,domain=PositiveReals)
    model.h_c = Var(model.T,domain=PositiveReals)
    model.ps_h = Var(model.T,domain=PositiveReals)
    model.ps_l = Var(model.W,model.T,domain=PositiveReals)

    # Objective
    def objfunc(model):
        obj = 365*other_data['y_tnum']*sum(other_data['LMP'][t-1]*model.p_del[t] for t in model.T)-model.c_t
        return obj
    model.object = Objective(rule=objfunc, sense=maximize)
    ## Constraint
    # capital cost constraint (*** add compressor cost)
    def consfunc_ct(model):
        # # C_E
        c_t_sum = (cost_data['elct_cap']+cost_data['elct_op']*other_data['y_tnum'])* \
                  sum(size_data['elct_p']*model.elct_num[w] for w in model.W)
        # C_FC
        c_t_sum += (cost_data['fc_cap']+cost_data['fc_op']*other_data['y_tnum'])* \
                   size_data['fc_p'] * model.fc_num
        # C_HP
        c_t_sum += (cost_data['lphl_cap']+cost_data['lphl_op']*other_data['y_tnum'])* \
                    sum(grid_config['dist_wf_hsc'][w-1]*model.lphl_num[w] for w in model.W)
        c_t_sum +=        (cost_data['hline_cap'] + cost_data['hline_op'] * other_data['y_tnum']) * \
                   grid_config['dist_hsc_sub']*model.hphl_num
        # C_HS
        c_t_sum += cost_data['hstrg_cap']*size_data['hstrg_max']+cost_data['hstrg_op']*other_data['y_tnum']
        # Compressor cost
        c_t_sum += cost_data['lpcmp_cap']*sum(size_data['lphl_h']*model.lphl_num[w] for w in model.W)
        c_t_sum += cost_data['hpcmp_cap']*size_data['hline_h'] * model.hphl_num
        return model.c_t == c_t_sum
    model.cons_ct = Constraint(rule=consfunc_ct)
    # Wind farm / electrolyzer hydrogen output
    def consfunc_pEout(model,w,t):
        return other_data['wf_p'][w-1][t-1]-model.p_c[w,t]==other_data['elct_cp']*model.h_elct[w,t]
    model.cons_pEout = Constraint(model.W,model.T, rule=consfunc_pEout)
    # HSC input hydrogen constraint
    def consfunc_hHSCin(model,t):
        return sum(model.h_elct[w,t] for w in model.W)==model.h_hstrg_in[t]+model.h_c[t]
    model.cons_hHSCin = Constraint(model.T, rule=consfunc_hHSCin)
    # low pressure hydrogen flow constraint
    def consfuc_lphyfl(model,w, t):
        eq_left = (model.ps_l[w,t]** 2) - (flow_data['ps_hsc'] ** 2)
        eq_right = 5007.7 * flow_data['lambda'] * flow_data['z_l'] * flow_data['T'] /flow_data['tho_l']*\
                   ((model.h_elct[w,t]/model.lphl_num[w])** 2) * \
                   grid_config['dist_wf_hsc'][w-1] / (flow_data['D_l'] ** 5)
        return eq_left == eq_right
    model.cons_lphyfl = Constraint(model.W, model.T, rule=consfuc_lphyfl)
    # high pressure hydrogen flow constraint
    def consfuc_hyfl(model, t):
        eq_left = (model.ps_h[t] ** 2) - (flow_data['ps_sub'] ** 2)
        eq_right = 5007.7 * flow_data['lambda'] * flow_data['z_h'] * flow_data['T'] /flow_data['tho_h']*\
                   ((model.h_hstrg_in[t]/model.hphl_num) ** 2) * \
                   grid_config['dist_hsc_sub'] / (flow_data['D_h'] ** 5)
        return eq_left == eq_right
    model.cons_hyfl = Constraint(model.T, rule=consfuc_hyfl)
    # low pressure compressor consumption constraint
    def consfuc_lpc(model, w, t):
        return model.p_c[w,t] == model.h_elct[w,t]*other_data['clp_cp']
    model.cons_lpc = Constraint(model.W, model.T, rule=consfuc_lpc)
    # high pressure compressor consumption constraint
    def consfuc_hpc(model, t):
        return model.h_c[t] == model.h_hstrg_in[t] * other_data['chp_cp']/other_data['fc_cp']
    model.cons_hpc = Constraint(model.T, rule=consfuc_hpc)
    # low pressure hydrogen line capacity constraint
    def consfuc_lphlcap(model,w,t):
        return size_data['lphl_h']*model.lphl_num[w]>=model.h_elct[w,t]
    model.cons_lphlcap = Constraint(model.W,model.T,rule=consfuc_lphlcap)
    # high pressure hydrogen line capacity constraint
    def consfuc_hphlcap(model, t):
        return size_data['hline_h'] * model.hphl_num >= model.h_hstrg_in[t]
    model.cons_hphlcap = Constraint( model.T, rule=consfuc_hphlcap)
    # Electrolyzer capacity constraint
    def consfuc_elctcap(model, w, t):
        return size_data['elct_p'] * model.elct_num[w] >= other_data['wf_p'][w-1][t-1]
    model.cons_elctcap = Constraint(model.W,model.T, rule=consfuc_elctcap)
    # Hydrogen storage constraint, delt_t is 1 and ignored
    def consfuc_hstrg(model, t):
        if t >= 2:
            return model.h_hstrg[t] == model.h_hstrg[t - 1] + (model.h_hstrg_in[t] - model.h_fc[t])
        if t == 1:
            return model.h_hstrg[t] == model.h_hstrg_in[t] - model.h_fc[t]
    model.cons_hstrg = Constraint(model.T, rule=consfuc_hstrg)
    # Hydrogen storage min constraint
    def consfuc_hstrg_min(model, t):
        return model.h_hstrg[t] >= size_data['hstrg_min']
    model.cons_hstrg_min = Constraint(model.T, rule=consfuc_hstrg_min)
    # Hydrogen storage max constraint
    def consfuc_hstrg_max(model, t):
        return model.h_hstrg[t] <= size_data['hstrg_max']
    model.cons_hstrg_max = Constraint(model.T, rule=consfuc_hstrg_max)
    # Fuel cell ouput constraint
    def consfuc_pFCout(model, t):
        return model.p_fc[t] == model.h_fc[t] * other_data['fc_cp']
    model.cons_pFCout = Constraint(model.T, rule=consfuc_pFCout)
    # Fuel cell capacity constraint
    def consfuc_fccap(model, t):
        return size_data['fc_p'] * model.fc_num >= model.p_fc[t]
    model.cons_fccap = Constraint(model.T, rule=consfuc_fccap)
    # Deliver power constraint
    def consfuc_pdel(model, t):
        return model.p_fc[t] == model.p_del[t]
    model.cons_pdel = Constraint(model.T, rule=consfuc_pdel)

    # load case data and create instance
    print('start creating the instance')
    case_pyomo = model.create_instance()
    print('finish creating the instance')
    # case_pyomo.pprint()
    return case_pyomo

### function to save the results of HP case
def write_result_hchydg(pym_model, savefl_nm):
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
    # print number of hydrogen lines from wind farm to hsc
    flpath = fdpath + '\\' + 'lphl_num.txt'
    f = open(flpath, 'w')
    lphlnum_str = ''
    for w in pym_model.W:
        lphlnum_str = lphlnum_str + str(pym_model.lphl_num[w]()) + ' '
    f.write(lphlnum_str)
    f.close()
    # print number of hydrogen lines from hsc to substation
    flpath = fdpath + '\\' + 'hphl_num.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.hphl_num()))
    f.close()
    # print number of elctrolyzer
    flpath = fdpath + '\\' + 'elct_num.txt'
    f = open(flpath, 'w')
    elctnum_str = ''
    for w in pym_model.W:
        elctnum_str = elctnum_str + str(pym_model.elct_num[w]())+' '
    f.write(elctnum_str)
    f.close()
    # print number of fuel cell
    flpath = fdpath + '\\' + 'fc_num.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.fc_num()))
    f.close()
    # print hydrogen ouput of all electrolyzers combined at t
    flpath = fdpath + '\\' + 'h_elct.txt'
    f = open(flpath, 'w')
    helct_str = ''
    for w in pym_model.W:
        for t in pym_model.T:
            helct_str = helct_str + str(pym_model.h_elct[w,t]()) + ' '
        helct_str += '\n'
    f.write(helct_str)
    f.close()
    # print hydrogen input to storage at t
    flpath = fdpath + '\\' + 'h_hstrg_in.txt'
    f = open(flpath, 'w')
    hstrgin_str = ''
    for t in pym_model.T:
        hstrgin_str = hstrgin_str + str(pym_model.h_hstrg_in[t]()) + ' '
    f.write(hstrgin_str)
    f.close()
    # print hydrogen storage combined at t [kW]
    flpath = fdpath + '\\' + 'h_hstrg.txt'
    f = open(flpath, 'w')
    hstrg_str = ''
    for t in pym_model.T:
        hstrg_str = hstrg_str + str(pym_model.h_hstrg[t]()) + ' '
    f.write(hstrg_str)
    f.close()
    # print Power output of all fuel cells combined at t [kW]
    flpath = fdpath + '\\' + 'p_fc.txt'
    f = open(flpath, 'w')
    pfc_str = ''
    for t in pym_model.T:
        pfc_str = pfc_str + str(pym_model.p_fc[t]()) + ' '
    f.write(pfc_str)
    f.close()
    # print hydrogen input of all fuel cells combined at t
    flpath = fdpath + '\\' + 'h_fc.txt'
    f = open(flpath, 'w')
    hfc_str = ''
    for t in pym_model.T:
        hfc_str = hfc_str + str(pym_model.h_fc[t]()) + ' '
    f.write(hfc_str)
    f.close()
    # print consumption of the compressor
    flpath = fdpath + '\\' + 'p_c.txt'
    f = open(flpath, 'w')
    pc_str = ''
    for w in pym_model.W:
        for t in pym_model.T:
            pc_str = pc_str + str(pym_model.p_c[w,t]()) + ' '
        pc_str += '\n'
    f.write(pc_str)
    f.close()