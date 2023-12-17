import random
import sys
import csv
import pandas as pd #The pandas library
import pandas_datareader.data as web # For accessing web data
from pandas import Series, DataFrame #Main pandas data structures
import matplotlib.pyplot as plt #main plotting tool for python
import matplotlib as mpl
import seaborn as sns #A more fancy plotting library
from datetime import datetime #For handling dates
import scipy as sp #library for scientific computations
from scipy import stats #The statistics part of the library
import numpy as np
from numpy import genfromtxt
import scipy as sp
from operator import itemgetter, attrgetter
from universalHashFunctions import create_random_hash_function
import json



def create_random_permutation(K=50):

    myHashFunction = create_random_hash_function()

    hashList = []             # stores pairs (i , H(i) )...
    randomPermutation = []    # stores the permutation of [1,2,...,K]

    for i in range(0,K):
        j=int(myHashFunction(i))
        hashList.append( (i,j) )
        
    # sort the hashList by second argument of the pairs...
    sortedHashList = sorted( hashList, key=itemgetter(1) )

    for i in range(0,len(sortedHashList)):
        randomPermutation.append(1 + sortedHashList[i][0])

    return randomPermutation


def create_random_hash_function(p=2**33-355, m=100):#m=2**32-1
    a = random.randint(1,p-1)
    b = random.randint(0, p-1)
    return lambda x: 1 + (((a * x + b) % p) % m)


def create_pairs(list_of_same_movies):
    n=len(list_of_same_movies)
    
    pairs_list=[]
    for i in range(n):
        for j in range(i+1,n):
            lst=[]
            lst.append(list_of_same_movies[i])
            lst.append(list_of_same_movies[j])
            pairs_list.append(lst)
    return pairs_list    
    
    

def jaccardSimilarity(movieId1,movieId2):
    
        s1 = set(movieList[movieId1])
        s2 = set(movieList[movieId2])

        Jaccard = ( len(s1.intersection(s2)) / len(s1.union(s2)) )

        return Jaccard
    
def signatureSimilarity(movieId1,movieId2,n_tonos,sig):
     
    
    same=0
    for i in range(n_tonos):
        if(sig[i][movieMap[movieId1]-1]==sig[i][movieMap[movieId2]-1]):
            same=same+1
    sign_similarity=same/n_tonos
    
    return sign_similarity
        
def get_key(val,my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist for val: "+str(val)
def two_digits(i):
    if(i<=9):
        return '0'+str(i)
    elif(i>9):
        return str(i)

def minHash(n,N,k,movieMap,userList,movieList):
    sig = [[0 for x in range(N)] for y in range(n)]
    hij = []
    for col in range(N):
        for i in range(n):
            sig[i][col]=1000
                        
                        
    for l in range(n):
        hi= create_random_permutation(k)
        hij.append(hi)    
            
    for row in range(k):
        
        for col in range(N):
                        
            if(get_key(col+1,movieMap) in userList[row+1]):
                    
                for i in range(n):
                                    
                    if(hij[i][row]<sig[i][col]):
                        sig[i][col]=hij[i][row]
    return sig


def LSH(n,b,r,sig,first_n_movies):
    
    f_hash=create_random_hash_function()
    number_cols=len(sig[0])
    
    same_movies=[]
    
    for i in range(b):
        bin_dict={}
        for j in range(first_n_movies):
            mylist=[]
            mystring=''
            for k in range(r):
                mylist.append( two_digits(sig[k+i*r][j]) )
            mystring = ''.join(map(str,mylist))
            key1 = int(mystring,base=10)
            dict_key=f_hash(key1)
            bin_keys=bin_dict.keys()
            if(dict_key in bin_keys):
                bin_dict[dict_key].append(j+1)
            
            else:
                lst=[]
                lst.append(j+1)
                bin_dict[dict_key]=lst
        x = bin_dict.keys()
        for k in x:
            if(len(bin_dict.get(k))>2):
                
                pairs=create_pairs(bin_dict.get(k))
                for p in pairs:
                    if(p not in same_movies):
                        same_movies.append(p)
            elif(len(bin_dict[k])==2):
                if(bin_dict.get(k) not in same_movies):
                    same_movies.append(bin_dict.get(k))
                
    return same_movies
    

            
             
    
            
            
        
    
    


                                                

     
###################################################

# loading from file

####################################################

files=['ratings_100users.csv','ratings.csv']
print('0 : ratings_100users.csv ')
print('1 : ratings.csv')
choice_file=int(input('give 0 or 1 to to set file input: '))
print('\n')
s=float(input('give the value of s  in [0,1]: '))
with open(files[choice_file],'r') as rating:
    
    rating= csv.reader(rating,delimiter=',')#,header=None
    next(rating)
    
    ###########userList define
    userList={}
    previous_user=1
    movies_of_user=[]
    ###########movieMap define
    movieMap={}
    movie_id=1
    ###########movieList define
    movieList={}
    movies_of_user=[]
    for line in rating:
        ###########userList
        user_newi=int(line[0])
        if (previous_user != user_newi):
            
            userList[previous_user]=movies_of_user
            movies_of_user=[]
            previous_user=user_newi
        
        movie_i=int(line[1])
        movies_of_user.append(movie_i)
        ###########userList
        ###########movieMap
        if (not(movieMap.get(movie_i)) ):
            movieMap[movie_i]=movie_id
            movie_id=movie_id+1
        ###########movieMap
        ###########movieList
        all_movie_keys=movieList.keys()
        if(movie_i in all_movie_keys):
            users_of_movie=movieList.get(movie_i)
            if( not(user_newi in users_of_movie) ):
                users_of_movie.append(user_newi)
                users_of_movie.sort()
                movieList.update({movie_i: users_of_movie})
                

        else:
            li=[]
            li.append(user_newi)
            movieList[movie_i]=li
    userList[previous_user]=movies_of_user
    print('\n \n ')
    print('Users :  ',len(userList))
    print('Movies : ',len(movieMap))
    n=40
    N=len(movieMap)
    k=len(userList)
    
    '''out_files=['mitrwa(userList,movieList,movieMap)(100 users,s=0.25).csv','mitrwa(userList,movieList,movieMap)(s=0.25).csv']
    with open(out_files[choice_file],'w+') as file:
        #json=json.dumps(userList)
        file.write('---- userList------ \n')
        file.write('######################')
        file.write(str(userList))
        file.write('\n---- movieList------ \n')
        file.write('######################')
        file.write(str(movieList))
        file.write('\n---- movieMap------ \n')
        file.write('######################')
        file.write(str(movieMap))
        file.close()'''
        
   
    
##############################################
    
# first experiment

##############################################
    print('\n############first experiment###############')
    
    JSims_greater_s={}
    JSims_less_s={}
    
    #s=0.25
    for i in range(1,21):
        movie_id1=get_key(i,movieMap)
        for j in range(i+1,21):
            
            movie_id2=get_key(j,movieMap)
            similar=jaccardSimilarity(movie_id1,movie_id2)
            if (similar >=s):
                JSims_greater_s[str(i)+','+str(j)]=similar
            else:
                JSims_less_s[str(i)+','+str(j)]=similar
            
    
    SignSims_greater_s=[]
    SignSims_less_s=[]
    print('relative items (Jaccard > s)',len(JSims_greater_s))
    sig_table=minHash(n,N,k,movieMap,userList,movieList)
    
    #sig=genfromtxt('sig.csv',delimiter=',').astype(int)
    #sig_table=sig.tolist()
    
    num_set = [5,10,15,20,25,30,35,40]
    sig_dataframe=pd.DataFrame(sig_table)
    sig_dataframe.to_csv ('sig_table.csv', index = False, header=False)
    
    for n_tonos in num_set:
        print('n_tonos: ',n_tonos)
        sigSim_n_less_s={}
        sigSim_n_greater_s={}
        for i in range(1,21):
            movie_id1=get_key(i,movieMap)
            for j in range(i+1,21):
               
                movie_id2=get_key(j,movieMap)
                sign_similarity=signatureSimilarity(movie_id1,movie_id2,n_tonos,sig_table)
                if(sign_similarity >= s):
                    sigSim_n_greater_s[str(i)+','+str(j)]=sign_similarity
                else:
                    sigSim_n_less_s[str(i)+','+str(j)]=sign_similarity
                    
        SignSims_greater_s.append(sigSim_n_greater_s)
        SignSims_less_s.append(sigSim_n_less_s)

    
    
    JS_greater_keys=JSims_greater_s.keys()
    JS_less_keys=JSims_less_s.keys()
    
    metrics={}
    metrics['false_positives']=[]
    metrics['false_negatives']=[]
    metrics['true_positives']=[]
    metrics['PRECISION']=[]
    metrics['RECALL']=[]
    metrics['F1']=[]
    for n_tonos in num_set:
        true_positives=0
        false_positives=0
        false_negatives=0
        
        SigS_greater_keys=SignSims_greater_s[n_tonos//5 -1].keys()
        for pair in JS_greater_keys:
            
            ids=pair.split(',')
            reverse_pair=str(ids[1])+','+str(ids[0])
            if(pair in SigS_greater_keys or(reverse_pair in SigS_greater_keys )):
                true_positives=true_positives+1
            else:
                false_negatives=false_negatives+1
        for pair in JS_less_keys:
            ids=pair.split(',')
            reverse_pair=str(ids[1])+','+str(ids[0])
            if(pair in SigS_greater_keys or(reverse_pair in SigS_greater_keys)):
                false_positives=false_positives+1
                
        
        metrics['false_positives'].append(false_positives)
        metrics['false_negatives'].append(false_negatives)
        metrics['true_positives'].append(true_positives)
        
        PRECISION=true_positives / ( true_positives + false_positives )
        RECALL = true_positives / ( true_positives + false_negatives )
        if(true_positives==0):
            F1=0
            
        else:
            F1 = 2 * RECALL * PRECISION / ( RECALL + PRECISION )
        
            
        metrics['PRECISION'].append(PRECISION)
        metrics['RECALL'].append(RECALL)
        metrics['F1'].append(F1)
        
        
        
        
        
        
        
    
    metrics_dataframe=pd.DataFrame(metrics)
    
    
    metrics_dataframe['n_tonos']=[5,10,15,20,25,30,35,40]
    print('\n  First Experiment values \n')
    print(metrics_dataframe)
    
    ####  CHARTS FOR false_negatives,false_positives,true_positives
    
    metrics_dataframe.plot(x = 'n_tonos', y = ['false_negatives','false_positives','true_positives' ],logy=True);
    plt.show()
     ####  CHARTS FOR PRECISION,RECALL,F1
    #
    metrics_dataframe.plot(x = 'n_tonos', y = ['PRECISION','RECALL','F1' ],logy=True);
    plt.show()
    print('\n############End of first experiment###############')
    
    
    
    
    ##########################################

    # second experiment

    ##########################################
    
    
    print('\n############second experiment###############')
    
    b_values=[20,10,8,5,4,2]
    r_values=[2,4,5,8,10,20]
    n=40
    
    metrics2={}
    metrics2['true_positives']=[]
    metrics2['false_positives']=[]
    metrics2['false_negatives']=[]
    metrics2['PRECISION']=[]
    metrics2['RECALL']=[]
    metrics2['F1']=[]
    length=len(b_values)
    if(k==100):
        first_n_movies=20
    else:
        first_n_movies=100
        
    for i in range(length):
        
        LSH_pairs=LSH(n,b_values[i],r_values[i],sig_table,first_n_movies)
        
        LSH_Jaccard_greater_s={}
        LSH_Jaccard_less_s={}
        
                
                
        for j in range(len(LSH_pairs)):
            movie_id1=get_key(LSH_pairs[j][0],movieMap)
            movie_id2=get_key(LSH_pairs[j][1],movieMap)
            similar=jaccardSimilarity(movie_id1,movie_id2)
            
            
            if (similar >=s):
                LSH_Jaccard_greater_s[str(LSH_pairs[j][0])+','+str(LSH_pairs[j][1])]=similar
            else:
                LSH_Jaccard_less_s[str(LSH_pairs[j][0])+','+str(LSH_pairs[j][1])]=similar
                
           
        
        
        true_positives=len(LSH_Jaccard_greater_s)
        false_positives=len(LSH_Jaccard_less_s)
        false_negatives=len(JSims_greater_s.keys())-true_positives
        
        
        metrics2['true_positives'].append(true_positives)
        metrics2['false_positives'].append(false_positives)
        metrics2['false_negatives'].append(false_negatives)
                                                             
        
        PRECISION = true_positives / ( true_positives + false_positives )
        RECALL = true_positives / ( true_positives + false_negatives )
        
        if(true_positives==0):
            F1=0
            
        else:
            
            F1 = 2 * RECALL * PRECISION / ( RECALL + PRECISION )
        
        
        metrics2['PRECISION'].append(PRECISION)
        metrics2['RECALL'].append(RECALL)
        metrics2['F1'].append(F1)
        
        
        
        
    metrics2_dataframe=pd.DataFrame(metrics2)
    
    
    metrics2_dataframe['b']=b_values
    metrics2_dataframe['r']=r_values
    print('\n  Second Experiment values \n')
    print(metrics2_dataframe)
    print('\n')
    '''results=['final_results(100).csv','final_results.csv']
    with open(results[choice_file],'w+') as file:
        file.write('---- experiment 1------ \n')
        
        file.write(str(metrics_dataframe))
        file.write('\n---- experiment 2------ \n')
       
        file.write(str(metrics2_dataframe))
        
        file.close()'''
        
    #### BAR CHARTS FOR true_positives,false_positives,false_negatives
    
    plt.bar('b=20,r=2', metrics2_dataframe['true_positives'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['true_positives'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['true_positives'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['true_positives'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['true_positives'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['true_positives'][5], color = 'orange', width = 0.25)   
    
    plt.title('true_positives')
    plt.ylabel('Value')
    plt.show()


    plt.bar('b=20,r=2', metrics2_dataframe['false_positives'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['false_positives'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['false_positives'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['false_positives'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['false_positives'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['false_positives'][5], color = 'orange', width = 0.25)   
    
    plt.title('false_positives')
    plt.ylabel('Value')
    plt.show()

    plt.bar('b=20,r=2', metrics2_dataframe['false_negatives'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['false_negatives'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['false_negatives'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['false_negatives'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['false_negatives'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['false_negatives'][5], color = 'orange', width = 0.25)   
    
    plt.title('false_negatives')
    plt.ylabel('Value')
    plt.show()

    


    

    
    #### BAR CHARTS FOR PRECISION,RECALL,F1

    plt.bar('b=20,r=2', metrics2_dataframe['PRECISION'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['PRECISION'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['PRECISION'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['PRECISION'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['PRECISION'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['PRECISION'][5], color = 'orange', width = 0.25)   
    
    plt.title('PRECISION')
    plt.ylabel('Value')
    plt.show()

    plt.bar('b=20,r=2', metrics2_dataframe['RECALL'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['RECALL'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['RECALL'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['RECALL'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['RECALL'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['RECALL'][5], color = 'orange', width = 0.25)   
    
    plt.title('RECALL')
    plt.ylabel('Value')
    plt.show()

    plt.bar('b=20,r=2', metrics2_dataframe['F1'][0], color = 'b', width = 0.25)
    plt.bar('b=10,r=4', metrics2_dataframe['F1'][1], color = 'g', width = 0.25)
    plt.bar('b=8,r=5', metrics2_dataframe['F1'][2], color = 'r', width = 0.25)
    plt.bar('b=5,r=8', metrics2_dataframe['F1'][3], color = 'gray', width = 0.25)
    plt.bar('b=4,r=10', metrics2_dataframe['F1'][4], color = 'purple', width = 0.25)
    plt.bar('b=2,r=20', metrics2_dataframe['F1'][5], color = 'orange', width = 0.25)   
    
    plt.title('F1')
    plt.ylabel('Value')
    plt.show()
   
    print('############End of second experiment###############')
    print('Done')
    
         

    
       
        
        
