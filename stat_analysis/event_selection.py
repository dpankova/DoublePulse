import numpy as np
import pandas as pd

def EventSelection(data_tc0,data_mc0,data_ec0,data_eg0,data_n0,data_ac0,data_ap0,data_c0):
    LLH = -0.1
    Qst1 = 2000
    Qst2 = 10
    Qst3 = 10
    NET1 =0.99
    NET2 =0.98
    NET3 =0.85

    data_tc0 = data_tc0[np.isfinite(data_tc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_tc0['logan_veto']['Cascade_rlogl'])]
    data_mc0 = data_mc0[np.isfinite(data_mc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_mc0['logan_veto']['Cascade_rlogl'])]
    data_ec0 = data_ec0[np.isfinite(data_ec0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ec0['logan_veto']['Cascade_rlogl'])]
    data_eg0 = data_eg0[np.isfinite(data_eg0['logan_veto']['SPE_rlogl']) & np.isfinite(data_eg0['logan_veto']['Cascade_rlogl'])]
    data_n0 = data_n0[np.isfinite(data_n0['logan_veto']['SPE_rlogl']) & np.isfinite(data_n0['logan_veto']['Cascade_rlogl'])]
    data_ac0 = data_ac0[np.isfinite(data_ac0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ac0['logan_veto']['Cascade_rlogl'])]
    data_ap0 = data_eg0[np.isfinite(data_ap0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ap0['logan_veto']['Cascade_rlogl'])]
    data_c0 = data_c0[np.isfinite(data_c0['logan_veto']['SPE_rlogl']) & np.isfinite(data_c0['logan_veto']['Cascade_rlogl'])]
    
    maskl_tc = data_tc0['logan_veto']['SPE_rlogl']-data_tc0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_mc = data_mc0['logan_veto']['SPE_rlogl']-data_mc0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ec = data_ec0['logan_veto']['SPE_rlogl']-data_ec0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_eg = data_eg0['logan_veto']['SPE_rlogl']-data_eg0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_n = data_n0['logan_veto']['SPE_rlogl']-data_n0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_ac = data_ac0['logan_veto']['SPE_rlogl']-data_ac0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ap = data_ap0['logan_veto']['SPE_rlogl']-data_ap0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_c = data_c0['logan_veto']['SPE_rlogl']-data_c0['logan_veto']['Cascade_rlogl'] > LLH

    maskq_tc = (data_tc['qst']['q'][:,0] >= Qst1) & (data_tc['qst']['q'][:,1] >= Qst2) & (data_tc['qst']['q'][:,2] >= Qst3)
    maskq_mc = (data_mc['qst']['q'][:,0] >= Qst1) & (data_mc['qst']['q'][:,1] >= Qst2) & (data_mc['qst']['q'][:,2] >= Qst3)
    maskq_ec = (data_ec['qst']['q'][:,0] >= Qst1) & (data_ec['qst']['q'][:,1] >= Qst2) & (data_ec['qst']['q'][:,2] >= Qst3)
    maskq_eg = (data_eg['qst']['q'][:,0] >= Qst1) & (data_eg['qst']['q'][:,1] >= Qst2) & (data_eg['qst']['q'][:,2] >= Qst3) 
    maskq_n = (data_n['qst']['q'][:,0] >= Qst1) & (data_n['qst']['q'][:,1] >= Qst2) & (data_n['qst']['q'][:,2] >= Qst3)
    maskq_ac = (data_ac['qst']['q'][:,0] >= Qst1) & (data_ac['qst']['q'][:,1] >= Qst2) & (data_ac['qst']['q'][:,2] >= Qst3)
    maskq_ap = (data_ap['qst']['q'][:,0] >= Qst1) & (data_ap['qst']['q'][:,1] >= Qst2) & (data_ap['qst']['q'][:,2] >= Qst3) 
    maskq_c = (data_c['qst']['q'][:,0] >= Qst1) & (data_c['qst']['q'][:,1] >= Qst2) & (data_c['qst']['q'][:,2] >= Qst3)

    masks_tc = (data_tc['preds']['n1'] >= NET1) & (data_tc['preds']['n2'] >= NET2) & (data_tc['preds']['n3'] >= NET3) 
    masks_mc = (data_mc['preds']['n1'] >= NET1) & (data_mc['preds']['n2'] >= NET2) & (data_mc['preds']['n3'] >= NET3)
    masks_ec = (data_ec['preds']['n1'] >= NET1) & (data_ec['preds']['n2'] >= NET2) & (data_ec['preds']['n3'] >= NET3)
    masks_eg = (data_eg['preds']['n1'] >= NET1) & (data_eg['preds']['n2'] >= NET2) & (data_eg['preds']['n3'] >= NET3)
    masks_n = (data_n['preds']['n1'] >= NET1) & (data_n['preds']['n2'] >= NET2) & (data_n['preds']['n3'] >= NET3)
    masks_ac = (data_ac['preds']['n1'] >= NET1) & (data_ac['preds']['n2'] >= NET2) & (data_ac['preds']['n3'] >= NET3)
    masks_ap = (data_ap['preds']['n1'] >= NET1) & (data_ap['preds']['n2'] >= NET2) & (data_ap['preds']['n3'] >= NET3) 
    masks_c =  (data_c['preds']['n1'] >= NET1) & (data_c['preds']['n2_1'] >= NET2) & (data_c['preds']['n3'] >= NET3)
   
    mask_tc = maskl_tc & maskq_tc & masks_tc
    mask_mc = maskl_mc & maskq_mc & masks_mc
    mask_ec = maskl_ec & maskq_ec & masks_ec
    mask_eg = maskl_eg & maskq_eg & masks_eg
    mask_n = maskl_n & maskq_n & masks_n
    mask_ac = maskl_ac & maskq_ac & masks_ac
    mask_ap = maskl_ap & maskq_ap & masks_ap
    mask_c = maskl_c & maskq_c & masks_c
    
    data_tc = data_tc0[mask_tc]
    data_mc = data_mc0[mask_mc]
    data_ec = data_ec0[mask_ec]
    data_eg = data_eg0[mask_eg]
    data_n = data_n0[mask_n]
    data_ac = data_ac0[mask_ac]
    data_ap = data_ap0[mask_ap]
    data_c = data_c0[mask_c]
    
    return data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c

def EventSelectionPreCut(data_tc0,data_mc0,data_ec0,data_eg0,data_n0,data_ac0,data_ap0,data_c0):
    LLH = -0.1
    
    data_tc0 = data_tc0[np.isfinite(data_tc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_tc0['logan_veto']['Cascade_rlogl'])]
    data_mc0 = data_mc0[np.isfinite(data_mc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_mc0['logan_veto']['Cascade_rlogl'])]
    data_ec0 = data_ec0[np.isfinite(data_ec0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ec0['logan_veto']['Cascade_rlogl'])]
    data_eg0 = data_eg0[np.isfinite(data_eg0['logan_veto']['SPE_rlogl']) & np.isfinite(data_eg0['logan_veto']['Cascade_rlogl'])]
    data_n0 = data_n0[np.isfinite(data_n0['logan_veto']['SPE_rlogl']) & np.isfinite(data_n0['logan_veto']['Cascade_rlogl'])]
    data_ac0 = data_ac0[np.isfinite(data_ac0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ac0['logan_veto']['Cascade_rlogl'])]
    data_ap0 = data_ap0[np.isfinite(data_ap0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ap0['logan_veto']['Cascade_rlogl'])]
    data_c0 = data_c0[np.isfinite(data_c0['logan_veto']['SPE_rlogl']) & np.isfinite(data_c0['logan_veto']['Cascade_rlogl'])]
    
    maskl_tc = data_tc0['logan_veto']['SPE_rlogl']-data_tc0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_mc = data_mc0['logan_veto']['SPE_rlogl']-data_mc0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ec = data_ec0['logan_veto']['SPE_rlogl']-data_ec0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_eg = data_eg0['logan_veto']['SPE_rlogl']-data_eg0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_n = data_n0['logan_veto']['SPE_rlogl']-data_n0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_ac = data_ac0['logan_veto']['SPE_rlogl']-data_ac0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ap = data_ap0['logan_veto']['SPE_rlogl']-data_ap0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_c = data_c0['logan_veto']['SPE_rlogl']-data_c0['logan_veto']['Cascade_rlogl'] > LLH
    
    data_tc = data_tc0[maskl_tc]
    data_mc = data_mc0[maskl_mc]
    data_ec = data_ec0[maskl_ec]
    data_eg = data_eg0[maskl_eg]
    data_n = data_n0[maskl_n]
    data_ac = data_ac0[maskl_ac]
    data_ap = data_ap0[maskl_ap]
    data_c = data_c0[maskl_c]
    
    return data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c

def EventSelectionPreCutNoTail(data_tc0,data_mc0,data_ec0,data_eg0,data_n0,data_ac0,data_ap0,data_c0):
    LLH = -0.1
    LLHt = 0.5
    
    data_tc0 = data_tc0[np.isfinite(data_tc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_tc0['logan_veto']['Cascade_rlogl'])]
    data_mc0 = data_mc0[np.isfinite(data_mc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_mc0['logan_veto']['Cascade_rlogl'])]
    data_ec0 = data_ec0[np.isfinite(data_ec0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ec0['logan_veto']['Cascade_rlogl'])]
    data_eg0 = data_eg0[np.isfinite(data_eg0['logan_veto']['SPE_rlogl']) & np.isfinite(data_eg0['logan_veto']['Cascade_rlogl'])]
    data_n0 = data_n0[np.isfinite(data_n0['logan_veto']['SPE_rlogl']) & np.isfinite(data_n0['logan_veto']['Cascade_rlogl'])]
    data_ac0 = data_ac0[np.isfinite(data_ac0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ac0['logan_veto']['Cascade_rlogl'])]
    data_ap0 = data_ap0[np.isfinite(data_ap0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ap0['logan_veto']['Cascade_rlogl'])]
    data_c0 = data_c0[np.isfinite(data_c0['logan_veto']['SPE_rlogl']) & np.isfinite(data_c0['logan_veto']['Cascade_rlogl'])]
    
    maskl_tc = data_tc0['logan_veto']['SPE_rlogl']-data_tc0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_mc = data_mc0['logan_veto']['SPE_rlogl']-data_mc0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ec = data_ec0['logan_veto']['SPE_rlogl']-data_ec0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_eg = data_eg0['logan_veto']['SPE_rlogl']-data_eg0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_n = data_n0['logan_veto']['SPE_rlogl']-data_n0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_ac = data_ac0['logan_veto']['SPE_rlogl']-data_ac0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ap = data_ap0['logan_veto']['SPE_rlogl']-data_ap0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_c = data_c0['logan_veto']['SPE_rlogl']-data_c0['logan_veto']['Cascade_rlogl'] > LLH
    
    masklt_tc = data_tc0['logan_veto']['SPE_rlogl']-data_tc0['logan_veto']['Cascade_rlogl'] < LLHt
    masklt_mc = data_mc0['logan_veto']['SPE_rlogl']-data_mc0['logan_veto']['Cascade_rlogl'] < LLHt
    masklt_ec = data_ec0['logan_veto']['SPE_rlogl']-data_ec0['logan_veto']['Cascade_rlogl'] < LLHt
    masklt_eg = data_eg0['logan_veto']['SPE_rlogl']-data_eg0['logan_veto']['Cascade_rlogl'] < LLHt
    masklt_n = data_n0['logan_veto']['SPE_rlogl']-data_n0['logan_veto']['Cascade_rlogl'] < LLHt
    masklt_ac = data_ac0['logan_veto']['SPE_rlogl']-data_ac0['logan_veto']['Cascade_rlogl'] < LLHt 
    masklt_ap = data_ap0['logan_veto']['SPE_rlogl']-data_ap0['logan_veto']['Cascade_rlogl'] < LLHt 
    masklt_c = data_c0['logan_veto']['SPE_rlogl']-data_c0['logan_veto']['Cascade_rlogl'] < LLHt
    
    data_tc = data_tc0[maskl_tc & masklt_tc]
    data_mc = data_mc0[maskl_mc & masklt_mc]
    data_ec = data_ec0[maskl_ec & masklt_ec]
    data_eg = data_eg0[maskl_eg & masklt_eg]
    data_n = data_n0[maskl_n & masklt_n]
    data_ac = data_ac0[maskl_ac & masklt_ac]
    data_ap = data_ap0[maskl_ap & masklt_ap]
    data_c = data_c0[maskl_c & masklt_c]
    
    return data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c


def EventSelectionTwoBin(data_tc0,data_mc0,data_ec0,data_eg0,data_n0,data_ac0,data_ap0,data_c0,COR1,COR2):

    LLH = -0.1
    Qst1 = 2000
    Qst2 = 10
    Qst3 = 10
    
    NET2 =0.98
    
    data_tc0 = data_tc0[np.isfinite(data_tc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_tc0['logan_veto']['Cascade_rlogl'])]
    data_mc0 = data_mc0[np.isfinite(data_mc0['logan_veto']['SPE_rlogl']) & np.isfinite(data_mc0['logan_veto']['Cascade_rlogl'])]
    data_ec0 = data_ec0[np.isfinite(data_ec0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ec0['logan_veto']['Cascade_rlogl'])]
    data_eg0 = data_eg0[np.isfinite(data_eg0['logan_veto']['SPE_rlogl']) & np.isfinite(data_eg0['logan_veto']['Cascade_rlogl'])]
    data_n0 = data_n0[np.isfinite(data_n0['logan_veto']['SPE_rlogl']) & np.isfinite(data_n0['logan_veto']['Cascade_rlogl'])]
    data_ac0 = data_ac0[np.isfinite(data_ac0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ac0['logan_veto']['Cascade_rlogl'])]
    data_ap0 = data_eg0[np.isfinite(data_ap0['logan_veto']['SPE_rlogl']) & np.isfinite(data_ap0['logan_veto']['Cascade_rlogl'])]
    data_c0 = data_c0[np.isfinite(data_c0['logan_veto']['SPE_rlogl']) & np.isfinite(data_c0['logan_veto']['Cascade_rlogl'])]
    
    maskl_tc = data_tc0['logan_veto']['SPE_rlogl']-data_tc0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_mc = data_mc0['logan_veto']['SPE_rlogl']-data_mc0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ec = data_ec0['logan_veto']['SPE_rlogl']-data_ec0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_eg = data_eg0['logan_veto']['SPE_rlogl']-data_eg0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_n = data_n0['logan_veto']['SPE_rlogl']-data_n0['logan_veto']['Cascade_rlogl'] > LLH
    maskl_ac = data_ac0['logan_veto']['SPE_rlogl']-data_ac0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_ap = data_ap0['logan_veto']['SPE_rlogl']-data_ap0['logan_veto']['Cascade_rlogl'] > LLH 
    maskl_c = data_c0['logan_veto']['SPE_rlogl']-data_c0['logan_veto']['Cascade_rlogl'] > LLH

    maskq_tc = (data_tc['qst']['q'][:,0] >= Qst1) & (data_tc['qst']['q'][:,1] >= Qst2) & (data_tc['qst']['q'][:,2] >= Qst3)
    maskq_mc = (data_mc['qst']['q'][:,0] >= Qst1) & (data_mc['qst']['q'][:,1] >= Qst2) & (data_mc['qst']['q'][:,2] >= Qst3)
    maskq_ec = (data_ec['qst']['q'][:,0] >= Qst1) & (data_ec['qst']['q'][:,1] >= Qst2) & (data_ec['qst']['q'][:,2] >= Qst3)
    maskq_eg = (data_eg['qst']['q'][:,0] >= Qst1) & (data_eg['qst']['q'][:,1] >= Qst2) & (data_eg['qst']['q'][:,2] >= Qst3) 
    maskq_n = (data_n['qst']['q'][:,0] >= Qst1) & (data_n['qst']['q'][:,1] >= Qst2) & (data_n['qst']['q'][:,2] >= Qst3)
    maskq_ac = (data_ac['qst']['q'][:,0] >= Qst1) & (data_ac['qst']['q'][:,1] >= Qst2) & (data_ac['qst']['q'][:,2] >= Qst3)
    maskq_ap = (data_ap['qst']['q'][:,0] >= Qst1) & (data_ap['qst']['q'][:,1] >= Qst2) & (data_ap['qst']['q'][:,2] >= Qst3) 
    maskq_c = (data_c['qst']['q'][:,0] >= Qst1) & (data_c['qst']['q'][:,1] >= Qst2) & (data_c['qst']['q'][:,2] >= Qst3)

    maskn2_tc = (data_tc0['preds']['n2'] >= NET2)  
    maskn2_mc = (data_mc0['preds']['n2'] >= NET2) 
    maskn2_ec = (data_ec0['preds']['n2'] >= NET2) 
    maskn2_eg = (data_eg0['preds']['n2'] >= NET2)
    maskn2_n = (data_n0['preds']['n2'] >= NET2) 
    maskn2_ac = (data_ac0['preds']['n2'] >= NET2) 
    maskn2_ap = (data_ap0['preds']['n2'] >= NET2)
    maskn2_c =  (data_c0['preds']['n2_1'] >= NET2) 
    
    maskb1_tc = (data_tc0['preds']['n1'] <= COR1[0]) & (data_tc0['preds']['n3'] >= COR1[1])
    maskb1_mc = (data_mc0['preds']['n1'] <= COR1[0]) & (data_mc0['preds']['n3'] >= COR1[1])
    maskb1_ec = (data_ec0['preds']['n1'] <= COR1[0]) & (data_ec0['preds']['n3'] >= COR1[1])
    maskb1_eg = (data_eg0['preds']['n1'] <= COR1[0]) & (data_eg0['preds']['n3'] >= COR1[1])
    maskb1_n = (data_n0['preds']['n1'] <= COR1[0]) & (data_n0['preds']['n3'] >= COR1[1])
    maskb1_ac = (data_ac0['preds']['n1'] <= COR1[0]) & (data_ac0['preds']['n3'] >= COR1[1])
    maskb1_ap = (data_ap0['preds']['n1'] <= COR1[0]) & (data_ap0['preds']['n3'] >= COR1[1])
    maskb1_c = (data_c0['preds']['n1'] <= COR1[0]) & (data_c0['preds']['n3'] >= COR1[1])
   
    maskb2_tc = (data_tc0['preds']['n1'] >= COR2[0]) & (data_tc0['preds']['n3'] <= COR2[1])
    maskb2_mc = (data_mc0['preds']['n1'] >= COR2[0]) & (data_mc0['preds']['n3'] <= COR2[1])
    maskb2_ec = (data_ec0['preds']['n1'] >= COR2[0]) & (data_ec0['preds']['n3'] <= COR2[1])
    maskb2_eg = (data_eg0['preds']['n1'] >= COR2[0]) & (data_eg0['preds']['n3'] <= COR2[1])
    maskb2_n = (data_n0['preds']['n1'] >= COR2[0]) & (data_n0['preds']['n3'] <= COR2[1])
    maskb2_ac = (data_ac0['preds']['n1'] >= COR2[0]) & (data_ac0['preds']['n3'] <= COR2[1])
    maskb2_ap = (data_ap0['preds']['n1'] >= COR2[0]) & (data_ap0['preds']['n3'] <= COR2[1])
    maskb2_c = (data_c0['preds']['n1'] >= COR2[0]) & (data_c0['preds']['n3'] <= COR2[1])
   
    maskb_tc = maskb1_tc | maskb2_tc
    maskb_mc = maskb1_mc | maskb2_mc
    maskb_ec = maskb1_ec | maskb2_ec
    maskb_eg = maskb1_eg | maskb2_eg
    maskb_n = maskb1_n | maskb2_n
    maskb_ac = maskb1_ac | maskb2_ac
    maskb_ap = maskb1_ap | maskb2_ap
    maskb_c = maskb1_c | maskb2_c

    
    mask_tc = maskl_tc & maskq_tc & maskn2_tc & maskb_tc
    mask_mc = maskl_mc & maskq_mc & maskn2_mc & maskb_mc
    mask_ec = maskl_ec & maskq_ec & maskn2_ec & maskb_ec
    mask_eg = maskl_eg & maskq_eg & maskn2_eg & maskb_eg
    mask_n = maskl_n & maskq_n & maskn2_n & maskb_n
    mask_ac = maskl_ac & maskq_ac & maskn2_ac & maskb_ac
    mask_ap = maskl_ap & maskq_ap & maskn2_ap & maskb_ap
    mask_c = maskl_c & maskq_c & maskn2_c & maskb_c
    
    data_tc = data_tc0[mask_tc]
    data_mc = data_mc0[mask_mc]
    data_ec = data_ec0[mask_ec]
    data_eg = data_eg0[mask_eg]
    data_n = data_n0[mask_n]
    data_ac = data_ac0[mask_ac]
    data_ap = data_ap0[mask_ap]
    data_c = data_c0[mask_c]
    
    return data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c


def MakeDataFrames(data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c,path,livetime,weight_name='weight_val_0',weight_name_c = 'weight_val',wtype ='nom',save = True):

    data_tc_label = [0]*len(data_tc)
    data_mc_label = [1]*len(data_mc)
    data_ec_label = [2]*len(data_ec)
    data_eg_label = [3]*len(data_eg)
    data_n_label = [4]*len(data_n)
    data_ac_label = [5]*len(data_ac)
    data_ap_label = [6]*len(data_ap)
    data_c_label = [7]*len(data_c)

    array_tc = np.array([data_tc_label,data_tc['preds']['n3'],data_tc['preds']['n1'],data_tc[weight_name][wtype]*livetime]).T 
    array_mc = np.array([data_mc_label,data_mc['preds']['n3'],data_mc['preds']['n1'],data_mc[weight_name][wtype]*livetime]).T
    array_ec = np.array([data_ec_label,data_ec['preds']['n3'],data_ec['preds']['n1'],data_ec[weight_name][wtype]*livetime]).T 
    array_eg = np.array([data_eg_label,data_eg['preds']['n3'],data_eg['preds']['n1'],data_eg[weight_name][wtype]*livetime]).T 
    array_n = np.array([data_n_label,data_n['preds']['n3'],data_n['preds']['n1'],data_n[weight_name][wtype]*livetime]).T
    array_ac = np.array([data_ac_label,data_ac['preds']['n3'],data_ac['preds']['n1'],data_ac[weight_name_c]*livetime]).T  
    array_ap = np.array([data_ap_label,data_ap['preds']['n3'],data_ap['preds']['n1'],data_ap[weight_name_c]*livetime]).T  
    array_c = np.array([data_c_label,data_c['preds']['n3'],data_c['preds']['n1'],data_c[weight_name_c]*livetime]).T 

    arrs =  array_mc
    arrs =  np.append(arrs,array_ec, axis = 0)
    arrs =  np.append(arrs,array_eg, axis = 0)
    arrs =  np.append(arrs,array_n, axis = 0)
    arrs =  np.append(arrs,array_ac, axis = 0)
    arrs =  np.append(arrs,array_ap, axis = 0)
    arrs =  np.append(arrs,array_c, axis = 0)

    index_values_s = range(len(array_tc))
    index_values_b = range(len(arrs))

    column_values = ['label', 'n3', 'n1','weight'] 
    sig_df = pd.DataFrame(data = array_tc,  index = index_values_s, columns = column_values) 
    bkg_df = pd.DataFrame(data = arrs,  index = index_values_b, columns = column_values)
    
    if save:
        sig_df.to_csv(path+'sig_out.csv',index=False)
        bkg_df.to_csv(path+'bkg_out.csv',index=False)
    
    return sig_df, bkg_df

def MakeDataFramesWeights(data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c,path,livetime,weight_name='weight_val_0',weight_name_c = 'weight_val', wtype ='nom',save = True):

    data_tc_label = [0]*len(data_tc)
    data_mc_label = [1]*len(data_mc)
    data_ec_label = [2]*len(data_ec)
    data_eg_label = [3]*len(data_eg)
    data_n_label = [4]*len(data_n)
    data_ac_label = [5]*len(data_ac)
    data_ap_label = [6]*len(data_ap)
    data_c_label = [7]*len(data_c)

    array_tc = np.array([data_tc_label,data_tc['preds']['n3'],data_tc['preds']['n1'],data_tc[weight_name][wtype]*livetime,\
                         data_tc['weight']['PrimaryNeutrinoEnergy'],data_tc['weight']['OneWeight'],data_tc['weight']['NEvents']]).T 
    array_mc = np.array([data_mc_label,data_mc['preds']['n3'],data_mc['preds']['n1'],data_mc[weight_name][wtype]*livetime,\
                         data_mc['weight']['PrimaryNeutrinoEnergy'],data_mc['weight']['OneWeight'],data_mc['weight']['NEvents']]).T  
    array_ec = np.array([data_ec_label,data_ec['preds']['n3'],data_ec['preds']['n1'],data_ec[weight_name][wtype]*livetime,\
                         data_ec['weight']['PrimaryNeutrinoEnergy'],data_ec['weight']['OneWeight'],data_ec['weight']['NEvents']]).T 
    array_eg = np.array([data_eg_label,data_eg['preds']['n3'],data_eg['preds']['n1'],data_eg[weight_name][wtype]*livetime,\
                         data_eg['weight']['PrimaryNeutrinoEnergy'],data_eg['weight']['OneWeight'],data_eg['weight']['NEvents']]).T 
    array_n = np.array([data_n_label,data_n['preds']['n3'],data_n['preds']['n1'],data_n[weight_name][wtype]*livetime,\
                         data_n['weight']['PrimaryNeutrinoEnergy'],data_n['weight']['OneWeight'],data_n['weight']['NEvents']]).T 
    array_ac = np.array([data_ac_label,data_ac['preds']['n3'],data_ac['preds']['n1'],data_ac[weight_name_c]*livetime,\
                         data_ac['weight']['PrimaryNeutrinoEnergy'],data_ac['weight']['OneWeight'],data_ac['weight']['NEvents']]).T 
    array_ap = np.array([data_ap_label,data_ap['preds']['n3'],data_ap['preds']['n1'],data_ap[weight_name_c]*livetime,\
                         data_ap['weight']['PrimaryNeutrinoEnergy'],data_ap['weight']['OneWeight'],data_ap['weight']['NEvents']]).T 
    array_c = np.array([data_c_label,data_c['preds']['n3'],data_c['preds']['n1'],data_c[weight_name_c]*livetime,\
                        np.zeros(len(data_c)),np.zeros(len(data_c)),np.zeros(len(data_c))]).T
    
    arrs =  array_mc
    arrs =  np.append(arrs,array_ec, axis = 0)
    arrs =  np.append(arrs,array_eg, axis = 0)
    arrs =  np.append(arrs,array_n, axis = 0)
    arrs =  np.append(arrs,array_ac, axis = 0)
    arrs =  np.append(arrs,array_ap, axis = 0)
    arrs =  np.append(arrs,array_c, axis = 0)

    index_values_s = range(len(array_tc))
    index_values_b = range(len(arrs))

    column_values = ['label', 'n3', 'n1','weight','PrimaryNeutrinoEnergy','OneWeight','NEvents'] 
    sig_df = pd.DataFrame(data = array_tc,  index = index_values_s, columns = column_values) 
    bkg_df = pd.DataFrame(data = arrs,  index = index_values_b, columns = column_values)
    
    if save:
        sig_df.to_csv(path+'sig_out.csv',index=False)
        bkg_df.to_csv(path+'bkg_out.csv',index=False)
    
    return sig_df, bkg_df

def MakeSimSum(data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c,path, livetime, weight_name='weight_val_0',\
                        weight_name_c = 'weight_val', wtype ='nom', save= 'False'):
    
    data_tc_label = [0]*len(data_tc)
    data_mc_label = [1]*len(data_mc)
    data_ec_label = [2]*len(data_ec)
    data_eg_label = [3]*len(data_eg)
    data_n_label = [4]*len(data_n)
    data_ac_label = [5]*len(data_ac)
    data_ap_label = [6]*len(data_ap)
    data_c_label = [7]*len(data_c)
   
    
    array_tc = np.array([data_tc_label,data_tc['qtot'],data_tc['logan_veto']['SPE_rlogl']-data_tc['logan_veto']['Cascade_rlogl'],\
                         data_tc['qst']['q'][:,0],data_tc['qst']['q'][:,1],data_tc['qst']['q'][:,2],data_tc['preds']['n1'],data_tc['preds']['n2'],\
                         data_tc['preds']['n3'],data_tc[weight_name][wtype]]).T 
    array_mc = np.array([data_mc_label, data_mc['qtot'],data_mc['logan_veto']['SPE_rlogl']-data_mc['logan_veto']['Cascade_rlogl'],\
                         data_mc['qst']['q'][:,0],data_mc['qst']['q'][:,1],data_mc['qst']['q'][:,2],data_mc['preds']['n1'],data_mc['preds']['n2'],\
                         data_mc['preds']['n3'],data_mc[weight_name][wtype]]).T 
    array_ec = np.array([data_ec_label, data_ec['qtot'],data_ec['logan_veto']['SPE_rlogl']-data_ec['logan_veto']['Cascade_rlogl'],\
                         data_ec['qst']['q'][:,0],data_ec['qst']['q'][:,1],data_ec['qst']['q'][:,2],data_ec['preds']['n1'],data_ec['preds']['n2'],\
                         data_ec['preds']['n3'],data_ec[weight_name][wtype]]).T 
    array_eg = np.array([data_eg_label, data_eg['qtot'],data_eg['logan_veto']['SPE_rlogl']-data_eg['logan_veto']['Cascade_rlogl'],\
                         data_eg['qst']['q'][:,0],data_eg['qst']['q'][:,1],data_eg['qst']['q'][:,2],data_eg['preds']['n1'],data_eg['preds']['n2'],\
                         data_eg['preds']['n3'],data_eg[weight_name][wtype]]).T 
    array_n = np.array([data_n_label, data_n['qtot'],data_n['logan_veto']['SPE_rlogl']-data_n['logan_veto']['Cascade_rlogl'],\
                         data_n['qst']['q'][:,0],data_n['qst']['q'][:,1],data_n['qst']['q'][:,2],data_n['preds']['n1'],data_n['preds']['n2'],\
                         data_n['preds']['n3'],data_n[weight_name][wtype]]).T 
    array_ac = np.array([data_ac_label, data_ac['qtot'],data_ac['logan_veto']['SPE_rlogl']-data_ac['logan_veto']['Cascade_rlogl'],\
                        data_ac['qst']['q'][:,0], data_ac['qst']['q'][:,1],data_ac['qst']['q'][:,2],data_ac['preds']['n1'],data_ac['preds']['n2'],\
                        data_ac['preds']['n3'],data_ac[weight_name_c]]).T 
    array_ap = np.array([data_ap_label, data_ap['qtot'],data_ap['logan_veto']['SPE_rlogl']-data_ap['logan_veto']['Cascade_rlogl'],\
                        data_ap['qst']['q'][:,0], data_ap['qst']['q'][:,1],data_ap['qst']['q'][:,2],data_ap['preds']['n1'],data_ap['preds']['n2'],\
                        data_ap['preds']['n3'],data_ap[weight_name_c]]).T 
    array_c = np.array([data_c_label, data_c['qtot'],data_c['logan_veto']['SPE_rlogl']-data_c['logan_veto']['Cascade_rlogl'],\
                        data_c['qst']['q'][:,0], data_c['qst']['q'][:,1],data_c['qst']['q'][:,2],data_c['preds']['n1'],data_c['preds']['n2_1'],\
                        data_c['preds']['n3'],data_c[ weight_name_c]]).T 
   
    
    array_sim =  array_tc
    array_sim  =  np.append(array_sim ,array_mc, axis = 0)
    array_sim  =  np.append(array_sim ,array_ec, axis = 0)
    array_sim  =  np.append(array_sim ,array_eg, axis = 0)
    array_sim  =  np.append(array_sim ,array_n, axis = 0)
    array_sim  =  np.append(array_sim ,array_ac, axis = 0)
    array_sim  =  np.append(array_sim ,array_ap, axis = 0)
    array_sim  =  np.append(array_sim ,array_c, axis = 0)

    
    index_values_tc = range(len(array_tc))
    index_values_sim = range(len(array_sim))
   

    column_values = ['label', 'qtot', 'llh_diff', 'qst0','qst1','qst2','n1', 'n2', 'n3','weight'] 
   
    
    sig_df = pd.DataFrame(data = array_tc,  index = index_values_tc, columns = column_values) 
    sim_df = pd.DataFrame(data = array_sim,  index = index_values_sim, columns = column_values)
    
    if save:
        sig_df.to_csv(path+'sig_out.csv',index=False)
        sim_df.to_csv(path+'sim_out.csv',index=False)

    return sig_df, sim_df

def GetRates(Name,livetime,data_tc,data_mc,data_ec,data_eg,data_n,data_ac,data_ap,data_c,data_g,data_b,weight_name='weight_val_0',\
                        weight_name_c = 'weight_val', wtype ='nom', burnsample = False):
    
    
    rate_tc = np.sum(data_tc[weight_name][wtype]*livetime) 
    rate_mc = np.sum(data_mc[weight_name][wtype]*livetime) 
    rate_ec = np.sum(data_ec[weight_name][wtype]*livetime) 
    rate_eg = np.sum(data_eg[weight_name][wtype]*livetime) 
    rate_n  = np.sum(data_n[weight_name][wtype]*livetime)
    rate_ac = np.sum(data_ac[weight_name_c]*livetime)
    rate_ap = np.sum(data_ap[weight_name_c]*livetime)
    rate_c  = np.sum(data_c[weight_name_c]*livetime)
    rate_g  = np.sum(data_g[weight_name_c]*livetime)
   
    rate_tc_err = np.sqrt(np.sum(np.square(data_tc[weight_name][wtype]*livetime))) 
    rate_mc_err = np.sqrt(np.sum(np.square(data_mc[weight_name][wtype]*livetime))) 
    rate_ec_err = np.sqrt(np.sum(np.square(data_ec[weight_name][wtype]*livetime)))
    rate_eg_err = np.sqrt(np.sum(np.square(data_eg[weight_name][wtype]*livetime))) 
    rate_n_err = np.sqrt(np.sum(np.square(data_n[weight_name][wtype]*livetime)))
    rate_ac_err = np.sqrt(np.sum(np.square(data_ac[weight_name_c]*livetime))) 
    rate_ap_err = np.sqrt(np.sum(np.square(data_ap[weight_name_c]*livetime))) 
    rate_c_err = np.sqrt(np.sum(np.square(data_c[weight_name_c]*livetime)))
    rate_g_err = np.sqrt(np.sum(np.square(data_g[weight_name_c]*livetime)))
    

    print(Name)
    print("NuTauCC = {0:.5f} +/- {1:.5f}".format(rate_tc,rate_tc_err))
    print("NuMuCC = {0:.5f} +/- {1:.5f}".format(rate_mc,rate_mc_err))
    print("NuECC =  {0:.5f} +/- {1:.5f}".format(rate_ec,rate_ec_err))
    print("NuEGR =  {0:.5f} +/- {1:.5f}".format(rate_eg,rate_eg_err))
    print("NuNC =   {0:.5f} +/- {1:.5f}".format(rate_n,rate_n_err))
    print("NuConv =   {0:.5f} +/- {1:.5f}".format(rate_ac,rate_ac_err))
    print("NuPrompt =   {0:.5f} +/- {1:.5f}".format(rate_ap,rate_ap_err))
    print("Corsika =   {0:.5f} +/- {1:.5f}".format(rate_c,rate_c_err))
    print("MuonGun =   {0:.5f} +/- {1:.5f}".format(rate_g,rate_g_err))
    
    if burnsample:
        rate_b = np.sum(data_b.shape[0])
        rate_b_err = np.sqrt(data_b.shape[0])
        print("BurnSample =   {0:.5f} +/- {1:.5f}".format(rate_b,rate_b_err))
        
  
        
def GetRatesSim(Name,arr,lt):
    
    rate = np.sum(arr['weight']*lt) 
    rate_err = np.sqrt(np.sum(np.square(arr['weight']*lt))) 
    
    print(Name)
    print("SumSim = {0:.3f} +/- {1:.3f}".format(rate,rate_err))
    
    
def GetRatesAtmos(livetime,data_tc,data_tn,data_mc,data_mn,data_ec,data_en,data_eg,data_c,weight_name='weight_c',\
                  weight_name2='weight_p',weight_name_c = 'weight_val', burnsample = False,Name ='Conv', Name2 = 'Prompt'):
    
    
    rate_tc = np.sum(data_tc[weight_name]*livetime) 
    rate_tn = np.sum(data_tn[weight_name]*livetime) 
    rate_mc = np.sum(data_mc[weight_name]*livetime) 
    rate_mn = np.sum(data_mn[weight_name]*livetime) 
    rate_ec = np.sum(data_ec[weight_name]*livetime)
    rate_en = np.sum(data_en[weight_name]*livetime)
    rate_eg = np.sum(data_eg[weight_name]*livetime)
    rate_c = np.sum(data_c[weight_name_c]*livetime)
   
   
    rate_tc_err = np.sqrt(np.sum(np.square(data_tc[weight_name]*livetime))) 
    rate_tn_err = np.sqrt(np.sum(np.square(data_tn[weight_name]*livetime))) 
    rate_mc_err = np.sqrt(np.sum(np.square(data_mc[weight_name]*livetime)))
    rate_mn_err = np.sqrt(np.sum(np.square(data_mn[weight_name]*livetime))) 
    rate_ec_err = np.sqrt(np.sum(np.square(data_ec[weight_name]*livetime)))
    rate_en_err = np.sqrt(np.sum(np.square(data_en[weight_name]*livetime))) 
    rate_eg_err = np.sqrt(np.sum(np.square(data_eg[weight_name]*livetime))) 
    rate_c_err = np.sqrt(np.sum(np.square(data_c[weight_name_c]*livetime)))
    
    print(Name)
    print("NuTauCC = {0:.4f} +/- {1:.4f}".format(rate_tc,rate_tc_err))
    print("NuTauCN = {0:.4f} +/- {1:.4f}".format(rate_tn,rate_tn_err))
    print("NuMuCC =  {0:.4f} +/- {1:.4f}".format(rate_mc,rate_mc_err))
    print("NuMuCN =  {0:.4f} +/- {1:.4f}".format(rate_mn,rate_mn_err))
    print("NuECC =   {0:.4f} +/- {1:.4f}".format(rate_ec,rate_ec_err))
    print("NuECN =   {0:.4f} +/- {1:.4f}".format(rate_en,rate_en_err))
    print("NuEGR =   {0:.4f} +/- {1:.4f}".format(rate_eg,rate_eg_err))
    
    
    rate_tc2 = np.sum(data_tc[weight_name2]*livetime) 
    rate_tn2 = np.sum(data_tn[weight_name2]*livetime) 
    rate_mc2 = np.sum(data_mc[weight_name2]*livetime) 
    rate_mn2 = np.sum(data_mn[weight_name2]*livetime) 
    rate_ec2 = np.sum(data_ec[weight_name2]*livetime)
    rate_en2 = np.sum(data_en[weight_name2]*livetime)
    rate_eg2 = np.sum(data_eg[weight_name2]*livetime)
    rate_c2 = np.sum(data_c[weight_name_c]*livetime)
   
   
    rate_tc_err2 = np.sqrt(np.sum(np.square(data_tc[weight_name2]*livetime))) 
    rate_tn_err2 = np.sqrt(np.sum(np.square(data_tn[weight_name2]*livetime))) 
    rate_mc_err2 = np.sqrt(np.sum(np.square(data_mc[weight_name2]*livetime)))
    rate_mn_err2 = np.sqrt(np.sum(np.square(data_mn[weight_name2]*livetime))) 
    rate_ec_err2 = np.sqrt(np.sum(np.square(data_ec[weight_name2]*livetime)))
    rate_en_err2 = np.sqrt(np.sum(np.square(data_en[weight_name2]*livetime))) 
    rate_eg_err2 = np.sqrt(np.sum(np.square(data_eg[weight_name2]*livetime))) 
    rate_c_err2 = np.sqrt(np.sum(np.square(data_c[weight_name_c]*livetime)))
    
    print(Name2)
    print("NuTauCC = {0:.4f} +/- {1:.4f}".format(rate_tc2,rate_tc_err2))
    print("NuTauCN = {0:.4f} +/- {1:.4f}".format(rate_tn2,rate_tn_err2))
    print("NuMuCC =  {0:.4f} +/- {1:.4f}".format(rate_mc2,rate_mc_err2))
    print("NuMuCN =  {0:.4f} +/- {1:.4f}".format(rate_mn2,rate_mn_err2))
    print("NuECC =   {0:.4f} +/- {1:.4f}".format(rate_ec2,rate_ec_err2))
    print("NuECN =   {0:.4f} +/- {1:.4f}".format(rate_en2,rate_en_err2))
    print("NuEGR =   {0:.4f} +/- {1:.4f}".format(rate_eg2,rate_eg_err2))
   
        
    