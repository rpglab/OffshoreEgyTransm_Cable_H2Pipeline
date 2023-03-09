### The pyomo model & saving results function for Hybrid case
## Author: Jin Lu, University of Houston.
## link: https://rpglab.github.io/resources/

### Import
from pyomo.environ import *
import os
import numpy as np

### model of hybrid case
def build_pym_pchydg(grid_config,cost_data,size_data, flow_data, other_data):
    model = AbstractModel()
    # Set
    W = list(range(1,grid_config['w_tnum']+1))
    T = list(range(1,25))
    model.W = Set(initialize=W)
    model.T = Set(initialize=T)
    # Variable
    model.p_del = Var(model.T,domain=PositiveReals)
    model.pl_num = Var(model.W,domain=PositiveIntegers)
    model.c_t = Var(domain=PositiveReals)
    model.hl_num = Var(domain=PositiveIntegers)
    model.elct_num = Var(domain=PositiveIntegers)
    model.fc_num = Var(domain=PositiveIntegers)
    model.p_elct = Var(model.T,domain=PositiveReals)
    model.h_elct = Var(model.T,domain=PositiveReals)
    model.h_hstrg = Var(model.T,domain=PositiveReals)
    model.h_fc = Var(model.T,domain=PositiveReals)
    model.p_fc = Var(model.T,domain=PositiveReals)
    model.wf_pout = Var(model.W,model.T,domain=PositiveReals)
    model.p_c = Var(model.T,domain=PositiveReals)    # power consumed by compressor
    model.ps_h = Var(model.T,domain=PositiveReals)   # High variable pressure at hydrogen super center at time t [bar]

    # Objective
    def objfunc(model):
        obj = 365*other_data['y_tnum']*sum(other_data['LMP'][t-1]*model.p_del[t] for t in model.T)-model.c_t
        return obj
    model.object = Objective(rule=objfunc, sense=maximize)
    ## Constraint
    # capital cost constraint   (*** add converter cost and compressor capital cost)
    def consfunc_ct(model):
        # C_PL
        c_t_sum = (cost_data['pline_cap']+cost_data['pline_op']*other_data['y_tnum'])*\
                sum(grid_config['dist_wf_sub'][w-1]*model.pl_num[w] for w in model.W)
        # C_E
        c_t_sum += (cost_data['elct_cap']+cost_data['elct_op']*other_data['y_tnum'])*\
                size_data['elct_p']*model.elct_num
        # C_FC
        c_t_sum += (cost_data['fc_cap']+cost_data['fc_op']*other_data['y_tnum'])* \
                   size_data['fc_p'] * model.fc_num
        # C_HP
        c_t_sum += (cost_data['hline_cap']+cost_data['hline_op']*other_data['y_tnum'])* \
                    grid_config['dist_hsc_sub']*model.hl_num
        # C_HS
        c_t_sum += cost_data['hstrg_cap']*size_data['hstrg_max']+cost_data['hstrg_op']*other_data['y_tnum']
        # cvt cap cost
        c_t_sum += sum(cost_data['cvt_wf_cap']*size_data['pline_p']*model.pl_num[w] for w in model.W)
        c_t_sum += cost_data['cvt_sub_cap']*sum(size_data['pline_p']*model.pl_num[w] for w in model.W)
        # compressor cap cost
        c_t_sum += cost_data['hpcmp_cap']*size_data['hline_h']*model.hl_num
        return model.c_t == c_t_sum
    model.cons_ct = Constraint(rule=consfunc_ct)
    # HSC input power constraint
    def consfunc_pEin(model,t):
        return sum(model.wf_pout[w,t] for w in model.W)==(model.p_elct[t]+model.p_c[t])/other_data['cvt_cp']
    model.cons_pEin = Constraint(model.T, rule=consfunc_pEin)
    # power line capacity constraint
    def consfuc_plcap(model,w,t):
        return size_data['pline_p']*model.pl_num[w]>=model.wf_pout[w,t]
    model.cons_plcap = Constraint(model.W,model.T,rule=consfuc_plcap)
    # dc power flow constraint
    def consfuc_dcpf(model, w, t):
        eq_left = other_data['cvt_cp']*other_data['wf_p'][w - 1][t - 1] - model.wf_pout[w, t]
        eq_right = 1000 * ((model.wf_pout[w, t] / flow_data['vo']/1000) ** 2) * grid_config['dist_wf_hsc'][w - 1] * \
                   flow_data['r']/model.pl_num[w]
        return eq_left == eq_right
    model.cons_dcpf = Constraint(model.W, model.T, rule=consfuc_dcpf)
    # compressor consumption constraint
    def consfuc_pc(model,t):
        return model.p_c[t]==model.h_elct[t]*other_data['chp_cp']
    model.cons_pc = Constraint(model.T, rule=consfuc_pc)
    # Electrolyzer Output constraint
    def consfuc_hEout(model,t):
        return model.p_elct[t]==model.h_elct[t]*other_data['elct_cp']
    model.cons_hEout = Constraint(model.T,rule=consfuc_hEout)
    # Hydrogen line capacity constraint
    def consfuc_hlcap(model, t):
        return size_data['hline_h']*model.hl_num >= model.h_elct[t]
    model.cons_hlcap = Constraint(model.T, rule=consfuc_hlcap)
    # hydrogen flow constraint
    def consfuc_hyfl(model, t):
        eq_left = (model.ps_h[t]**2)-(flow_data['ps_sub']**2)
        eq_right = 5007.7*flow_data['lambda']*flow_data['z_h']*flow_data['T']/flow_data['tho_h']*\
                   ((model.h_elct[t]/model.hl_num)**2)*\
                   grid_config['dist_hsc_sub']/(flow_data['D_h']**5)
        return eq_left==eq_right
    model.cons_hyfl = Constraint(model.T, rule=consfuc_hyfl)
    # Elctrolyzer capacity constraint
    def consfuc_elctcap(model, t):
        return size_data['elct_p'] * model.elct_num >= model.p_elct[t]
    model.cons_elctcap = Constraint(model.T, rule=consfuc_elctcap)
    # Hydrogen storage constraint, delt_t is 1 and ignored
    def consfuc_hstrg(model, t):
        if t>=2:
            return model.h_hstrg[t] == model.h_hstrg[t-1] + (model.h_elct[t]-model.h_fc[t])
        if t==1:
            return model.h_hstrg[t] == model.h_elct[t]-model.h_fc[t]
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
        return model.p_fc[t] == model.h_fc[t]*other_data['fc_cp']
    model.cons_pFCout = Constraint(model.T, rule=consfuc_pFCout)
    # Fuel cell capacity constraint
    def consfuc_fccap(model, t):
        return size_data['fc_p']*model.fc_num >= model.p_fc[t]
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

### function to save the results of hybrid case
def write_result_pchydg(pym_model, savefl_nm):
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
    # print number of hydrogen lines
    flpath = fdpath + '\\' + 'hl_num.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.hl_num()))
    f.close()
    # print number of elctrolyzer
    flpath = fdpath + '\\' + 'elct_num.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.elct_num()))
    f.close()
    # print number of fuel cell
    flpath = fdpath + '\\' + 'fc_num.txt'
    f = open(flpath, 'w')
    f.write(str(pym_model.fc_num()))
    f.close()
    # print Power input of all electrolyzers combined at t [kW]
    flpath = fdpath + '\\' + 'p_elct.txt'
    f = open(flpath, 'w')
    pelct_str = ''
    for t in pym_model.T:
            pelct_str = pelct_str + str(pym_model.p_elct[t]()) + ' '
    f.write(pelct_str)
    f.close()
    # print hydrogen ouput of all electrolyzers combined at t
    flpath = fdpath + '\\' + 'h_elct.txt'
    f = open(flpath, 'w')
    helct_str = ''
    for t in pym_model.T:
        helct_str = helct_str + str(pym_model.h_elct[t]()) + ' '
    f.write(helct_str)
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
    # print hydrogen storage combined at t [kW]
    flpath = fdpath + '\\' + 'h_hstrg.txt'
    f = open(flpath, 'w')
    hstrg_str = ''
    for t in pym_model.T:
        hstrg_str = hstrg_str + str(pym_model.h_hstrg[t]()) + ' '
    f.write(hstrg_str)
    f.close()
    # print consumption of the compressor
    flpath = fdpath + '\\' + 'p_c.txt'
    f = open(flpath, 'w')
    pc_str = ''
    for t in pym_model.T:
        pc_str = pc_str + str(pym_model.p_c[t]()) + ' '
    f.write(pc_str)
    f.close()
    # print wind farm pout after power line consumption
    flpath = fdpath + '\\' + 'wf_pout.txt'
    f = open(flpath, 'w')
    wfpout_str = ''
    for w in pym_model.W:
        for t in pym_model.T:
            wfpout_str = wfpout_str + str(pym_model.wf_pout[w,t]()) + ' '
        wfpout_str += '\n'
    f.write(wfpout_str)
    f.close()