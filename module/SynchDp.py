# import sys
# sys.path.append('/home/qotmd01/CovSF_2/')
# sys.path.append('/home/qotmd01/CovSF_2/Data/')
# sys.path.append('/home/qotmd01/CovSF_2/Multi_Omics/')
# sys.path.append('/home/qotmd01/CovSF_2/module/')

# import pandas as pd
# import numpy as np
# import copy
# import math
# from enum import IntEnum
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns

# class Trace(IntEnum):
#     STOP = 0
#     LEFT = 1
#     UP = 2
#     DIAGONAL = 3

# class SynchDp:
#     def __init__(self,data1,data2,day_gap,pcut,ucut,window=3,candidate=10):
#         #Sequence initialize
#         self.day_gap=day_gap
#         self.pcut=pcut
#         self.ucut=ucut
#         self.candidate=candidate
        
#         seq1,seq2= data1.iloc[0].to_list(), data2.iloc[0].to_list()
        
#         self.seq1,self.seq2 = seq1,seq2
#         self.data1, self.data2 = data1,data2
        
#         #Calculate day gap score    
#         self.x,self.x_gap = self.seq1,self.calc_gap_score(self.data1)
#         self.y,self.y_gap = self.seq2,self.calc_gap_score(self.data2)
        
        
#         #Alignment
#         self.window = window
#         self.row, self.col = len(self.x) + 1 , len(self.y) + 1
#         self.score_matrix,self.backtrace_matrix,self.matchlen_matrix = self.alingment()
#         self.avg_matrix = self.score_matrix / self.matchlen_matrix
#         self.avg_matrix[np.isnan(self.avg_matrix)] = 0
        
#         #Multiple Backtrace
#         self.results = [] # 0=score,
#         temp_score_matrix = copy.deepcopy(self.score_matrix)
#         temp_avg_matrix = copy.deepcopy(self.avg_matrix)

#         for row in range(len(self.avg_matrix)):
#             for col in range(len(self.avg_matrix[0])):
#                 if self.avg_matrix[row][col] < self.pcut:
#                     self.avg_matrix[row][col] = 0
                    
#         while not np.all(temp_avg_matrix == 0):
#             p_score, path, temp_score_matrix,temp_avg_matrix = self.backtrace(self.pcut, self.backtrace_matrix,self.matchlen_matrix,temp_score_matrix,temp_avg_matrix)
#             if len(path) >= 1:
#                 if p_score > 0:
                    
#                     final_x = []
#                     final_y = []
#                     start_x,start_y= path[0][0], path[0][1] 
#                     for i in range(self.window,1,-1):
#                         final_x.append(self.x[start_x - i])
#                         final_y.append(self.y[start_y - i])
#                     for node in path:
#                         final_x.append(self.x[node[0]-1])
#                         final_y.append(self.y[node[1]-1])

#                     path_len = len(path) + self.window - 1
#                     u_dist_sum = 0
#                     for i in range(len(final_x)):
#                         if final_x[i] >= 10:
#                             u_dist_sum += abs(10 - final_y[i])
#                         else:
#                             u_dist_sum += abs(final_x[i] - final_y[i])

#                     u_dist = u_dist_sum / path_len

#                     x = path[0][0]
#                     y = path[0][1]

#                     for i in range(0, window-1):
#                         x = x - 1
#                         y = y - 1
#                         wind_path = (x,y)
#                         path = [wind_path] + path
                    
#                     score = p_score * math.log2(len(path))
#                     if u_dist < self.ucut:
#                         self.results.append([score, u_dist, p_score, path])
                    
#         self.results.sort(key= lambda x:-x[0])

        
#         to_remove = []

#         for num in self.results:
#             if num[1] >= 4:
#                 to_remove.append(num)

#         for num in to_remove:
#             self.results.remove(num)
        
#     def alingment(self):
#         score_matrix = np.zeros((self.row, self.col), dtype=float)
#         backtrace_matrix = np.zeros((self.row, self.col), dtype=float)
#         matchlen_matrix = np.zeros((self.row, self.col),dtype=float)
        
#         for i in range(self.window,self.row):
#             for j in range(self.window,self.col):
#                 pcor_gap = abs(round(sum(self.y_gap[j - self.window : j-1]) - sum(self.x_gap[i - self.window : i-1]),3))
                
#                 x_window,y_window = self.x[i-self.window:i],self.y[j - self.window:j]

                
#                 if np.std(x_window) == 0 and np.std(y_window) == 0:
#                     pcor = 1
#                 elif np.std(x_window) == 0 or np.std(y_window) == 0:
#                     pcor = 0
#                 else:
#                     pcor = stats.pearsonr(x_window, y_window)[0]
                
#                 diagonal_score = score_matrix[i-1,j-1] + pcor - pcor_gap
                
#                 vertical_score = score_matrix[i-1,j] + (-1 * self.x_gap[i-2])
                
#                 horizontal_score = score_matrix[i,j-1] + (-1 * self.y_gap[j-2])
                
#                 score_matrix[i,j] = max(0, diagonal_score, vertical_score, horizontal_score)
                
#                 if score_matrix[i,j] == 0:
#                     backtrace_matrix[i,j] = Trace.STOP
#                     matchlen_matrix[i,j] = 0
                    
#                 elif score_matrix[i,j] == horizontal_score:
#                     backtrace_matrix[i,j] = Trace.LEFT
#                     matchlen_matrix[i,j] = matchlen_matrix[i,j-1]
                
#                 elif score_matrix[i,j] == vertical_score:
#                     backtrace_matrix[i,j] = Trace.UP
#                     matchlen_matrix[i,j] = matchlen_matrix[i-1,j]
                    
#                 elif score_matrix[i,j] == diagonal_score:
#                     backtrace_matrix[i,j] = Trace.DIAGONAL
#                     matchlen_matrix[i,j] = matchlen_matrix[i-1,j-1] + 1
                    
                    
#         return score_matrix,backtrace_matrix,matchlen_matrix
    

#     def backtrace(self, pcut, backtrace_matrix,matchlen_matrix,score_matrix,avg_matrix):
#         path = []
#         i, j = max_xy(avg_matrix)

#         score = avg_matrix[i,j]
#         while backtrace_matrix[i,j] != Trace.STOP:
            
#             avg_matrix[i,j] = 0

#             path.append((i,j))
            
#             if score_matrix[i,j] < (pcut * matchlen_matrix[i,j]):
#                 path.pop(-1)
#                 return score, path[::-1],score_matrix, avg_matrix
    
#             if backtrace_matrix[i,j] == Trace.DIAGONAL:
#                 i -= 1
#                 j -= 1
            
#             elif backtrace_matrix[i,j] == Trace.LEFT:
#                 j -= 1
            
#             elif backtrace_matrix[i,j] == Trace.UP:
#                 i -= 1

#         return score,path[::-1],score_matrix,avg_matrix

#     def calc_gap_score(self,data):
#         gscore=[]
#         cur=int(data.columns[0])
#         day_scaler=self.day_gap
#         for i in data.columns.to_list()[1:]:
#             day=int(i)
#             gscore.append((day-cur)*day_scaler)
#             cur=day
#         return gscore
    

# def acc_mean(ps):
#     DAYS = list(ps[0].keys())
#     acc_mean = []
#     for day in DAYS:
#         _ = np.array([ps[p][day] for p in range(4) if ps[p][day] is not None])
#         if len(_) != 0: acc_mean.append(_.mean())
        
#     return acc_mean


# def max_xy(ndarr):
#     mode = ndarr.shape[1]
#     ndarr_p = copy.deepcopy(ndarr)
#     max,argmax = np.max(ndarr_p), np.argmax(ndarr_p)
#     i,j = (argmax // mode, argmax % mode)
#     ndarr_p[i,j] = 0
#     while np.max(ndarr_p) == max:
#         argmax = np.argmax(ndarr_p)
#         i,j = (argmax // mode, argmax % mode)
#         ndarr_p[i,j] = 0
        
#     return i,j


# def ref_seq(x_length): #generate reference sequence
#     # x 축 포인트 생성
#     x_points = np.linspace(0, x_length - 1, x_length)
    
#     # 정규분포의 평균과 표준편차 설정
#     mean = (x_length - 1) / 2
#     std_dev = x_length / 10  # x 길이에 따라 적절한 표준편차 설정

#     # 정규분포 확률 밀도 함수(PDF)를 사용하여 y 값 계산 및 정규화
#     y_values = norm.pdf(x_points, mean, std_dev)
#     y_values_normalized = y_values / max(y_values)  # y값을 최대값으로 나누어 정규화


#     return pd.DataFrame([y_values_normalized],columns=range(x_length))

import pandas as pd
import numpy as np
import copy
import math
from enum import IntEnum
from scipy import stats

class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    UP = 2
    DIAGONAL = 3

class synchdp:
    def __init__(self, data1, data2, penalty, pcut, window = 3):
        
        #Sequence initialize
        self.penalty = penalty
        self.pcut = pcut
        self.window = window
        self.data1, self.data2 = data1, data2
        self.seq1, self.seq2 = data1.iloc[0].to_list(), data2.iloc[0].to_list()
        self.x, self.y = self.seq1, self.seq2
        
        #Initialization Matrix
        self.row, self.col = len(self.x) + 1 , len(self.y) + 1
        self.cumulative_matrix, self.backtrace_matrix, self.length_matrix = self.alingment()
        self.average_matrix = self.cumulative_matrix / self.length_matrix
        self.average_matrix[np.isnan(self.average_matrix)] = 0
        
        self.results = []
        temp_cumulative_matrix = copy.deepcopy(self.cumulative_matrix)
        temp_average_matrix = copy.deepcopy(self.average_matrix)

        for row in range(len(self.average_matrix)):
            for col in range(len(self.average_matrix[0])):
                if self.average_matrix[row][col] < self.pcut:
                    self.average_matrix[row][col] = 0
                    
        self.results = self.backtrace(self.window, self.pcut, self.backtrace_matrix, self.length_matrix, temp_cumulative_matrix, temp_average_matrix)
        self.results.sort(key= lambda x:-x[0])
        
        
    def calc_TGS(self, data):
        TGS = []
        current = int(data.columns[0])
        penalty_scale = self.penalty
        
        for t in data.columns.to_list()[1:]:
            time = int(t)
            TGS.append((time - current) * penalty_scale)
            current = time
        return TGS
    
    
    def alingment(self):
        #Calculate Time Gap Score
        self.TGS_x, self.TGS_y = self.calc_TGS(self.data1), self.calc_TGS(self.data2)
        
        cumulative_matrix = np.zeros((self.row, self.col), dtype=float)
        backtrace_matrix = np.zeros((self.row, self.col), dtype=float)
        length_matrix = np.zeros((self.row, self.col),dtype=float)
        average_matrix = np.zeros((self.row, self.col),dtype=float)
        
        for i in range(self.window, self.row):
            for j in range(self.window, self.col):                
                x_window, y_window = self.x[i - self.window : i], self.y[j - self.window : j]
                if np.std(x_window) == 0 and np.std(y_window) == 0:
                    Pscore = 1
                elif np.std(x_window) == 0 or np.std(y_window) == 0:
                    Pscore = 0
                else:
                    Pscore = stats.pearsonr(x_window, y_window)[0]
                IP = abs(round(sum(self.TGS_y[j - self.window : j-1]) - sum(self.TGS_x[i - self.window : i-1]), 3))
                MatchScore = Pscore - IP
                
                diagonal_score = cumulative_matrix[i - 1, j - 1] + MatchScore
                vertical_score = cumulative_matrix[i - 1, j] + (-1 * self.TGS_x[i - 2])
                horizontal_score = cumulative_matrix[i, j - 1] + (-1 * self.TGS_y[j - 2])
                
                cumulative_matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)
                
                if cumulative_matrix[i, j] == 0:
                    backtrace_matrix[i, j] = Trace.STOP
                    length_matrix[i, j] = 0
                elif cumulative_matrix[i, j] == horizontal_score:
                    backtrace_matrix[i, j] = Trace.LEFT
                    length_matrix[i, j] = length_matrix[i, j - 1]
                elif cumulative_matrix[i, j] == vertical_score:
                    backtrace_matrix[i, j] = Trace.UP
                    length_matrix[i, j] = length_matrix[i - 1, j]
                elif cumulative_matrix[i, j] == diagonal_score:
                    backtrace_matrix[i, j] = Trace.DIAGONAL
                    length_matrix[i, j] = length_matrix[i - 1, j - 1] + 1        
        return cumulative_matrix, backtrace_matrix, length_matrix
    
    
    def backtrace(self, window, pcut, backtrace_matrix, length_matrix, cumulative_matrix, average_matrix):
        result = []
        
        while not np.all(average_matrix == 0):
            path = []
        
            argamx = np.argmax(average_matrix)
            i, j = (argamx // average_matrix.shape[1], argamx % average_matrix.shape[1])

            avgscore = average_matrix[i,j]
            
            while backtrace_matrix[i,j] != Trace.STOP:
                average_matrix[i,j] = 0

                path.append((i,j))

                if cumulative_matrix[i,j] < (pcut * length_matrix[i,j]):
                    path.pop(-1)
                    break
                else:
                    if backtrace_matrix[i,j] == Trace.DIAGONAL:
                        i -= 1
                        j -= 1
                    elif backtrace_matrix[i,j] == Trace.LEFT:
                        j -= 1
                    elif backtrace_matrix[i,j] == Trace.UP:
                        i -= 1
            path = path[::-1]

            if len(path) >= 1 and avgscore > 0:
                final_x, final_y = [], []
                start_x, start_y = path[0][0], path[0][1] 

                for i in range(self.window, 1, -1):
                    final_x.append(self.x[start_x - i])
                    final_y.append(self.y[start_y - i])

                for node in path:
                    final_x.append(self.x[node[0] - 1])
                    final_y.append(self.y[node[1] - 1])

                x = path[0][0]
                y = path[0][1]

                for i in range(0, window - 1):
                    x = x - 1
                    y = y - 1
                    wind_path = (x, y)
                    path = [wind_path] + path

                alignmentscore = avgscore * math.log2(len(path))
                self.results.append([alignmentscore, avgscore, path])
        return self.results