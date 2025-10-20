import numpy as np
import scipy.stats as stats
def point_rank_weight(hcp):
    if hcp == 0: hons = np.array([0,0,0,0])
    elif hcp == 1: hons = np.array([0,0,0,408/522])
    elif hcp == 2: hons = np.array([0,0,232/522,408/522])
    elif hcp == 3: hons = np.array([0,100/522,232/522,408/522])
    elif hcp <= 5: hons = np.array([232/1388,476/1388,852/1388,1200/1388])
    elif hcp <= 7: hons = np.array([888/2896,1568/2896,2016/2896,2648/2896])
    elif hcp <= 12: hons = np.ones(4)
    elif hcp <= 15: hons = np.array([4392/2673, 3964/2673, 3288/2673, 2676/2673])
    elif hcp <= 18: hons = np.array([2160/941, 1612/941, 1356/941, 976/941])
    elif hcp <= 21: hons = np.array([704/233,536/233,384/233,240/233])
    else: hons = np.array([1116/396,804/396,740/396,396/396])

    return np.concatenate([hons, np.ones(9)])

def heuristic_rank_weight(hcp_max, hcp_min):
    v_list = []; w_list = []
    for hcp in range(hcp_min, hcp_max+1):
        v_list.append(point_rank_weight(hcp))
        w_list.append(stats.norm.pdf(hcp, loc = 10, scale = 4.127))
    return np.average(v_list, axis=0, weights=w_list)