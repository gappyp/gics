# -*- coding: utf-8 -*-
"""
Created on Fri May 04 08:34:35 2018


spike remove a folder full of TS files from LEMI format (remote referenced)

@author: u81670
"""

import numpy as np
import os.path as op
import os
import csv


def readTS(timeSeries):
    with open(timeSeries, 'r') as openfile:
        data_iter = csv.reader(openfile, delimiter = ' ')
        data = [data for data in data_iter]
    TS = np.asarray(data, dtype = float)
    return(TS)

#spike rejection function, takes channel and returns filtered and number of rejects
def spikeReject(channel):
    st = 200    #define spike threshold
    n = 0   #count spikes removed
    for i, value in enumerate(channel):   #go through the channel values
        #pick out outliers larger than threshold (one sample width)
        if abs(channel[i]-channel[i-1]) > st and abs(channel[i]-channel[i+1]) > st:
            #replace value with 5pt median
            channel[i] = np.median(channel[(i-2):(i+3)])
            n+=1
            print 'x',
    return(channel, n)
    

#### inputs ####
workdir = r'M:\AusLAMP\AusLAMP_NSW\Data_Processed\April_June2018\SavedTS\Spike_remove'


batchTxt = [] #collate all the filenames for batch processing
fileList = [ff for ff in os.listdir(workdir) if ((ff.endswith('.txt')))] #find all the TS files

columns = ['year', 'month', 'day', 'hour', 'min', 'sec',
           'Bx', 'By', 'Bz', 'E1', 'E2', 'Rx', 'Ry']

#open file, pass columns through spikeReject and save resulting file.
for fname in fileList:
    
    timeSeries = readTS(op.join(workdir,fname))
#    timeSeries = np.genfromtxt(op.join(workdir,fname), names = columns, dtype=(int, int, int, int, 
#												int, int, float, float, 
#												float, float, float, float, 
#												float))
    print 'read in TS file ', fname
    print '===================================='
    
    #extract the site name from the file name
    for i, char in enumerate(fname):
        if char == '_':
            siteID = fname[:i]
            break
    
    spikes = 0
    for col in range(6,13):
        timeSeries[:,col], n = spikeReject(timeSeries[:,col])
        spikes+=n
        print 'Removed spikes from field, ', columns[col], n
    print 'Total spikes removed from TS, ', siteID, spikes
    
    np.savetxt(op.join(workdir,fname[:-4]+'_ds.txt'),timeSeries,fmt=['%1i']*6+['%.3f']*7)
    #save the spike removed TS, with original name _ds
    batchTxt.append((op.join(workdir[workdir.rfind('\\')+1:],fname[:-4]+'_ds.txt')))
    
    '''
    Now have to make separate config files for each as LEMI assumes config file has same name as site TS
    this makes the EDIS get named site.edi, so maybe just use the original cfg and then edit the EDI later to clean up the 
    'site' field.
    '''
    
    RR = open((op.join(workdir,fname[:-4]+'.cfg')), "r")
    refSite = RR.readlines()
    RR.close()
    refSite[0] = refSite[0][:4]+ '\t' + fname[:-4]+'_ds' +'\n'  
    
    configWrite = open(op.join(workdir,fname[:-4]+'_ds.cfg'), "w") #write config to file
    for line in refSite:
        configWrite.write(line)
    configWrite.close()

    
    print '\n================================'

with open(op.join(workdir,'batch_inputs_ds.txt'),'w') as writeBatch:
    for rows in batchTxt:
        writeBatch.write(rows+'\n')
    