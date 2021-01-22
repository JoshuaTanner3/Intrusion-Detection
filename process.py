# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:55:23 2020

@author: Joshua
"""

import numpy as np
import glob
import pandas as pd

files = glob.glob('./*.txt')



cnt1 = 10

while cnt1 < len(files):
    
    important = []
    keep_phrases = ['val_accuracy']
    
    
    infile = files[cnt1]
      
    with open(infile) as f:
        f = f.readlines()
        
    for line in f:
        for phrase in keep_phrases:
            if phrase in line:
                important.append(line)
                break
            

    
    total = []
    data = []
    cnt = 0
    
    while cnt < len(important):

        #ave = float(important[cnt][127:-1])
        ave = float(important[cnt][-7:-1])
        
        value = (ave)
        
        data.append(value)
        if len(data) == 50:
            total.append(data)
            data = []
            
        
        
        
        cnt = cnt + 1
    
    column_name = ['1', '2', '3', '4', '5']
    total = np.array(total).T.tolist()
    df = pd.DataFrame(total, columns=column_name)
        
    df.to_csv( files[cnt1][2:-4] + '.csv', index=None)
    # print('Successfully made csv.')
    cnt1 = cnt1 + 1

# new_array = np.vstack([all_files])
# new_array = np.transpose(new_array)


#np.savetxt('50_oid.csv', new_array, delimiter=',')
