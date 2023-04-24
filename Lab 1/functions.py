# Tzortzis Evangelos AM: 3088


import math
import time
import random
import pickle
import os

def MyReadDataRoutine(filename, numDocuments):
    docIdList = list(range(numDocuments))
    docCount = 0
    previousCount = 1
    docList = [[] for _ in range(numDocuments)]
    with open(filename,'r') as f:
        d, w, nnz = [int(next(f)) for x in range(3)] # read 3 first lines: d, w, nnz
        #print(d, w, nnz)
        docList = [[] for _ in range(d)]
        for line in f:
            if int(line.split()[0]) != docCount:
                docCount +=1
                #print(docCount)

            if (not docList[int(line.split()[0])-1]) and docCount > 1:
                docList[int(line.split()[0])-2] = frozenset(docList[int(line.split()[0])-2])

            docList[int(line.split()[0])-1].append(int(line.split()[1]))

            previousCount = docCount

        docList[-1] = frozenset(docList[-1])
        docList = [docList[i] for i in range(numDocuments)]

        print('Number of documents: {0},from which {2} will be considered.\nNumber of words: {1}.'.format(d,w,numDocuments))
        return docList, d, w


def MyJacSimWithSets(docID1, docID2):
    intersectionCounter = 0
    doc1len = len(docID1)
    doc2len = len(docID2)
    for w1 in docID1:
        for w2 in docID2:
            if w1 == w2:
                intersectionCounter +=1

    jacSim = intersectionCounter / (doc1len + doc2len - intersectionCounter)
    return jacSim


def MyJacSimWithOrderedLists(docID1, docID2):
    pos1 = 0
    pos2 = 0
    intersectionCounter = 0
    doc1len = len(docID1)
    doc2len = len(docID2)   
    doc1List = list(docID1)
    doc2List = list(docID2)

    doc1List.sort()
    doc2List.sort()

    while pos1<doc1len and pos2<doc2len:
        if doc1List[pos1] == doc2List[pos2]:
            intersectionCounter+=1
            pos1+=1
            pos2+=1
        else:
            if doc1List[pos1] < doc2List[pos2]:
                pos1+=1
            else:
                pos2+=1

    jacSim = intersectionCounter / (doc1len + doc2len - intersectionCounter)
    return jacSim


def create_random_hash_function(p=2**33-355, m=2**32-1):
    a = random.randint(1, p-1)
    b = random.randint(0, p-1)
    return lambda x: 1 + (((a * x + b) % p) % m)


def MyMinHash(docList, k, w):
    # unique word count
    sig = []
    hashFunctions = []
    permutations = []

    # Initialize the sig list
    for i in range(k):
        sig.append([])
        for col in range(len(docList)):
            sig[-1].append(math.inf)

    # Create the hash permutations
    print('Creating the hash permutations...')
    b = time.time()
    for i in range(k):
        h  = create_random_hash_function()
        randomHash = { x+1:h(x+1) for x in range(w)}  # 1 to w

        hashFunctions.append(randomHash)

        myHashKeysOrderedByValues = sorted(randomHash, key=randomHash.get)
        myHash = { myHashKeysOrderedByValues[x]:x+1 for x in range(w)}  # 1 to w
        permutations.append(myHash)

    print('Permutations created in: ',time.time()-b)

    stepPrint = [int(x) for x in [1,w,w/8,(w/8)*2,(w/8)*3,(w/8)*4,(w/8)*5,(w/8)*6,(w/8)*7]]  # nine prints 
    print('MinHash:')
    b = time.time()

    # signatures computation
    for row in range(w):
        for col in range(len(docList)):
            if (row+1) in docList[col]:
                for i in range(k):
                    if permutations[i].get(row+1) < sig[i][col]:
                        sig[i][col] = permutations[i].get(row+1)

        if (row+1) in stepPrint:
            p = ((row+1)/w)*100
            t = time.time() - b
            print('{0:.1f}% rows done in time {1}.'.format(p,t))

    return sig


#def MySigSim(docID1, docID2, numPermutations):  # asked function without siglist
def MySigSim(sigList, docID1, docID2, numPermutations):  # sigList is needed
    # this is slow
    # sigsimCount = 0
    # for i in range(numPermutations):
    #     sig_i = sigList[i]
    #     if sig_i[docID1-1] == sig_i[docID2-1]:
    #         sigsimCount +=1
    # sigsim = sigsimCount / numPermutations

    # this is 3 times faster with the same parameters
    tempList = []
    tempList = [i for i,j in zip(sigList[docID1-1],sigList[docID2-1]) if i==j]

    sigsim = len(tempList) / numPermutations
    return sigsim

def bruteForceNearestNeighborsWithJacSim(docID, documents, numNeighbors):
    d = len(documents)
    sim_list = []
    doc_ids = []

    # print('Nearest Neighbors with Jaccard Sim for docID: ',docID)
    b = time.time()
    for id in range(d):
        if id != docID-1:
            jc = MyJacSimWithOrderedLists(documents[docID-1],documents[id])
            sim_list.append(jc)
            doc_ids.append(id+1)
    a = time.time() - b

    # print('Done in: {0} secs.'.format(a))

    dist_dict = {doc_ids[x]:(1 - sim_list[x]) for x in range(len(doc_ids))}

    neighbor_counter = 0
    sorted_dict = {}
    for k,v in sorted(dist_dict.items(), key=lambda x:x[1]):
        if neighbor_counter < numNeighbors:
            sorted_dict[k] = v
            neighbor_counter += 1
        else: break

    # calculate the average similarity of the nearest neighbors
    avgSim = 0
    for v in sorted_dict.values():
        avgSim += (1-v)
    avgSim = avgSim / numNeighbors

    return avgSim


def bruteForceNearestNeighborsWithSigSim(sig, docID, documents, numNeighbors, numPermutations=200):
    d = len(documents)
    sim_list = []
    doc_ids = []

    # print('Nearest Neighbors with Signature Sim for docID: ',docID)
    b = time.time()
    for id in range(d):
        if id+1 != docID:
            sigsim = MySigSim(sig, docID, (id+1), numPermutations=numPermutations)
            sim_list.append(sigsim)
            doc_ids.append(id+1)
    a = time.time() - b

    # print('Done in: {0} secs.'.format(a))

    dist_dict = {doc_ids[x]:(1 - sim_list[x]) for x in range(len(doc_ids))}

    neighbor_counter = 0
    sorted_dict = {}
    for k,v in sorted(dist_dict.items(), key=lambda x:x[1]):
        if neighbor_counter < numNeighbors:
            sorted_dict[k] = v
            neighbor_counter += 1
        else: break

    # calculate the average similarity of the nearest neighbors for a document
    avgSim = 0
    for v in sorted_dict.values():
        avgSim += (1-v)
    avgSim = avgSim / numNeighbors

    return avgSim

def avgSim(numDocuments, docs, numNeighbors=10, simMeasure='Jaccard',sig=None, numSigPermutations=200):
    avgsim = 0

    print('Calculating Average {0} Similarity...'.format(simMeasure))

    b = time.time()
    if simMeasure == 'Jaccard':
        for i in range(numDocuments):
            if i % 500 == 0: # print every 500
                print('Document: '+str(i)+' out of: '+str(numDocuments)+' in time: '+str(time.time()-b))
            avgJacSim = bruteForceNearestNeighborsWithJacSim(i+1,docs,numNeighbors=numNeighbors)
            avgsim += avgJacSim
    elif simMeasure == 'Signature':
        
        # transpose the signature matrix
        sig = list(zip(*sig))
        
        for i in range(numDocuments):
            if i % 500 == 0: # print every 500
                print('Document: '+str(i)+' out of: '+str(numDocuments)+' in time: '+str(time.time()-b))
            avgSigSim = bruteForceNearestNeighborsWithSigSim(sig, i+1, documents=docs, numNeighbors=numNeighbors, numPermutations=numSigPermutations)
            avgsim += avgSigSim
    
    avgsim = avgsim / numDocuments

    a = time.time() - b
    print('Time of average {2} similarity for {0} documents: {1:.3f}'.format(numDocuments, a, simMeasure))
    return avgsim

def lsh(sig, rowsPerBand, w):
    docNum = len(sig[0])
    numBands = math.floor((len(sig)/rowsPerBand))
    s = (1/numBands)**(1/rowsPerBand)
    print('LSH Similarity threshold: ',s)
    
    LSHdicts = {}
    
    # canditateDocPairs = []
    canditateDocPairs = {}

    h  = create_random_hash_function()
    
    print('Calculating the pairs of similar documents...')
    before = time.time()
    remainder = 0
    for b in range(numBands):  # for each band
        LSHdicts[b] = {}

        if (len(sig) - (rowsPerBand*(b+1))) < rowsPerBand:  # calculate the remainder of elements after the last band in case the rowsPerBand doesn't divide the sig length 
            remainder = len(sig)-(rowsPerBand*(b+1))
            
        for d in range(docNum):  # for each document
            hashNum = hash(tuple([i[d] for i in sig[rowsPerBand*b:rowsPerBand*(b+1)+remainder]]))  # for each row and each document in the current band calculate the hash value of the signature vector
            LSHdicts[b][d+1] = h(hashNum)  # calculate the random hash of the vector hash number  
        
        LSHdicts[b] = {k:v for k,v in sorted(LSHdicts[b].items(), key=lambda item: item[1])}  # sort the dictionary in the current band

        keyList = list(LSHdicts[b].keys())
        valueList = list(LSHdicts[b].values())

        # first approach, works but is slow because of the "([keyList[i],keyList[i+1]] not in canditateDocPairs)"
        # for i in range(len(valueList)-1):  # find all pairs of similar documents 
        #     if (valueList[i] == valueList[i+1]) and ([keyList[i],keyList[i+1]] not in canditateDocPairs):
        #         canditateDocPairs.append([keyList[i],keyList[i+1]])
        #         canditateDocPairs.append([keyList[i+1],keyList[i]])

            #print('{0} of {1} done'.format(i,len(valueList)))

        # second approach using a dictionary is faster significantly
        for i in range(len(keyList)-1):
            if LSHdicts[b][keyList[i]] == LSHdicts[b][keyList[i+1]]:
                if keyList[i] in canditateDocPairs:
                    canditateDocPairs[keyList[i]].append(keyList[i+1])
                else:
                    canditateDocPairs[keyList[i]] = [keyList[i+1]]

                if keyList[i+1] in canditateDocPairs:
                    canditateDocPairs[keyList[i+1]].append(keyList[i])
                else:
                    canditateDocPairs[keyList[i+1]] = [keyList[i]]

        # remove duplicates by using sets
        for k,v in canditateDocPairs.items():
            s = list(set(v))
            canditateDocPairs[k] = s

        # print('{0} band(s) done out of {1}.'.format(b,numBands))
        
    timer = time.time() - before
    print('Time taken for the calculation: {:.3f}s'.format(timer))
    
    return canditateDocPairs

def calcJaccardSimListForLSH(lshSimilarCandidatesList, documents):
    jaccardSimList = {}
    counter = 0
    print('Calculating the similarity for all candidate pairs...')
    b = time.time()
    for k,v in lshSimilarCandidatesList.items():
        jaccardSimList[k] = []
        for neighbor in v:

            if neighbor in jaccardSimList: # find distance instead of calculating it a second time for the second element of the pair
                jc = jaccardSimList[neighbor][lshSimilarCandidatesList[neighbor].index(k)]
                jaccardSimList[k].append(jc)
            else:
                jaccardSimList[k].append(MyJacSimWithOrderedLists(documents[k-1],documents[neighbor-1]))
        counter +=1
    print('Done in {0}s.'.format(time.time()-b))

    return jaccardSimList


def calcSignatureSimListForLSH(lshSimilarCandidatesList,documents,signatureMatrix, permutations):
    signatureMatrix = list(zip(*signatureMatrix))

    sigsimList = {}
    counter = 0
    print('Calculating the similarity for all candidate pairs...')
    b = time.time()
    for k,v in lshSimilarCandidatesList.items():
        sigsimList[k] = []
        for neighbor in v:

            if neighbor in sigsimList: # find distance instead of calculating it a second time for the second element of the pair
                jc = sigsimList[neighbor][lshSimilarCandidatesList[neighbor].index(k)]
                sigsimList[k].append(jc)
            else:
                sigsimList[k].append(MySigSim(sigList=signatureMatrix ,docID1=k, docID2=neighbor,numPermutations=permutations))
        counter +=1
    print('Done in {0}s.'.format(time.time()-b))

    return sigsimList

# combine lsh candidates dictionary and their similarity dictionary eg {1:{2:0.234, 6:0.4526},2{1:0.234}}
def combineCandidateFilesWithSimilarities(candidates, simMatrix):
    candidates_plus_similarity = {}
    for k,v in candidates.items():
        candidates_plus_similarity[k] = {v[i]:simMatrix[k][i] for i in range(len(v))}

    return candidates_plus_similarity


# sort the neighbors based on their similarity
def sortCandidatesSimilarities(candidates_plus_sim):
    candidates_plus_similarity_sorted = {}
    for k,v in candidates_plus_sim.items():
        candidates_plus_similarity_sorted[k] = {k1: v1 for k1, v1 in sorted(v.items(), key=lambda item: item[1],reverse=True)}
    
    return candidates_plus_similarity_sorted


def checkNeighbors(candidates, numNeighbors, msg):
    lengthList = []
    for k,v in candidates.items():
        lengthList.append(len(v))

    counter = len([i for i in lengthList if i>=numNeighbors])
    if counter != len(candidates):
        print(msg)
        print('Number of files without requested number of neighbors({0}): {1}'.format(numNeighbors, len(candidates)-counter))
        
        counter1 = len([i for i in lengthList if i>=1])
        counter2 = len([i for i in lengthList if i>=2])
        counter3 = len([i for i in lengthList if i>=3])
        counter4 = len([i for i in lengthList if i>=4])
        counter5 = len([i for i in lengthList if i>=5])
        print('\nNote:')
        print('Documents with at least 1 neighbor:',counter1)
        print('Documents with at least 2 neighbor:',counter2)
        print('Documents with at least 3 neighbor:',counter3)
        print('Documents with at least 4 neighbor:',counter4)
        print('Documents with at least 5 neighbor:',counter5)
        print('\n')

        print('Please choose less rows per band number.')
        return 0
    return 1


def avgSimLSH(numDocuments, candidate_files_sorted, numNeighbors=5):
    avg_jsim = 0
    for d in range(numDocuments):
        avg_jsim += sum(list(candidate_files_sorted.get(d+1).values())[:numNeighbors]) / numNeighbors  # average similarity for one document with its numNeighbors with the closest similarity
    avg_jsim = avg_jsim / numDocuments  # average similarity derived as the average of all the documents
    return avg_jsim



if __name__ == '__main__':
    pass