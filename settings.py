SEED1=1685077
SEED2=4120939
SEED3=7755379
CORR_GROUP = {
    'P_SUM': #Var to Predict
        ['S_SUM', # Sum of apparent power S1, S2, S3
        'S_L3', # Apparent power S3 
        'S_L2', # Apparent Power S2
        'S_L1', # Apparent power S1
        'C_phi_L1', #Fund power CosPhi factor L1
        'C_phi_L2', #Fund power CosPhi factor L2
        'P_SUM', # Sum of powers P1, P2, P3
        'P_L1', # Real Power 1
        'P_L2', # Real Power 2
        'P_L3', # Real Power 3
        'Q_SUM', # SUm of fund reactive power
        'Q_L1', #Fundamental Reactive Power Q1
        'Q_L2', #Fundamental Reactive Power Q2
        'Q_L3', #Fundamental Reactive Power Q3
        'I_L1', # Current L1
        'I_L2', # Current L2
        'I_L3'], # Current L3
    'U_L1_N':
        ['U_L1_L2', # Voltage L1_l2
        'U_L3_L1', # Voltage L3_l1
        'U_L3_N', # Voltage L3_N
        'U_L2_L3', # Voltage L2_l3
        'U_L2_N', # Voltage L2_N
        'U_L1_N'], # VOltage L1_N 
    'I_SUM': 
        ['I_SUM'], # Current Sum 
    'F': 
        ['F'], # Measured Freq
    'RealE_SUM':
        ['RealEc_SUM', # Sum of Consumed Energy 
        'RealEc_L1', # Real Energy Consumed L1
        'RealEc_L2', # Real Energy Consumed L2
        'RealEc_L3', # Real Energy Consumed L3
        'RealE_SUM', # Sum of Real Energy 
        'RealE_L2', # Real Energy L2
        'RealE_L3', # Real Energy L3
        'RealE_L1', # Real Energy L1
        'AE_SUM', # Apparent Energy Sum
        'AE_L1', # Apparent Energy L1
        'AE_L2', # Apparent Energy L2
        'AE_L3', # Apparent Energy L3
        'ReacE_L1'], #Reactive Energy L1
    'C_phi_L3': 
        ['C_phi_L3'] #Fund power CosPhi factor L3
}