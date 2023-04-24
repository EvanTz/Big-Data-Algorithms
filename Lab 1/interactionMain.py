# Tzortzis Evangelos AM: 3088

# used python 3.9.7

# to run in windows: python interactiveMain.py
# to run in linux: python3 interactiveMain.py

# selection of the parameters is done from the menu

import pickle
import time
import functions
import os


if __name__ == "__main__":
    menu = ["Type filename to open or <quit> to quit: ",
            'Type the number of documents to consider: ',
            'Type the number of neighbors to locate (eg 2,3,4,5) with minimum 1: ',
            'Type the number of permutations for the construction of the signature matrix: ',
            'Select the similarity metric, (1) Jaccard or (2) Signature: ',
            'Select which method to use, (1) Brute Force or (2) LSH: ',
            'Select rows per band for the LSH algorithm: ',
            'Compare two documents\' similarities with each other: ',
            'Document 1 id: ',
            'Document 2 id: ']
    

    msg_warning = '/!\\ WARNING: selected rows per band does not ensure selected number of neighbors /!\\'

    main_menu = """ 
    1. Choose a file.
    2. Choose the number of documents to consider.
    3. Choose the number of neighbors to locate.
    4. Choose the number of permutations of the signature matrix.
    5. Choose the similarity metric.
    6. Choose the method to use for the average similarity calculation.
    7. Calculate/Load the signature matrix.
    8. Run the calculation of the average similarity.
    9. Compare two documents with each other.
    10. Quit.
    """ 

    fl = ''
    d = -1
    neigh = -1
    perms = -1
    simMetric = -1
    calcMethod = -1
    rowsperband = -1

    sigmatrix = False
    lsh_created = False
    lsh_similarities = False

    while(True):
        print(main_menu)
        selection = -1
        while selection not in [1,2,3,4,5,6,7,8,9,10]:
            selection = int(input('Select an option from the menu:') or '-1')

        print('\n')

        if selection == 10:
            quit()

        elif selection == 1:
            print('Some file suggestions (copy-paste them in the field below):')
            print('DATA_1-docword.enron.txt')
            print('DATA_2-docword.nips.txt')

            fl = ''
            while not os.path.isfile(fl):
                fl = input(menu[0])
                if os.path.isfile(fl):
                    break
                if fl == 'quit':
                    quit()
                print('File does not exist! Try again.')
        
        elif selection == 2:
            if fl != '':
                # choose the number of documents
                d = -1
                while d <= 0:
                    d = int(input(menu[1]) or '-1')
                    if d <= 0:
                        print('Invalid number. Try again.')
            
                # open the file here
                docs, docNum ,words = functions.MyReadDataRoutine(fl,d) 
                print('File opened successfully!')
            else:
                print('No filename selected. Select a filename first.')

        elif selection == 3:
            # choose the number of neighbors
            neigh = -1
            while neigh <= 0:
                neigh = int(input(menu[2]) or '-1')
                if neigh <= 0:
                    print('Invalid number. Try again.')

        elif selection == 4:
            # choose the number of permutations
            perms = -1
            while perms <= 0:
                perms = int(input(menu[3])  or '-1')
                if perms <= 0:
                    print('Invalid number. Try again.')

        elif selection == 5:
            # select the similarity metric
            simMetric = -1
            while simMetric != 1 and simMetric != 2:
                simMetric = int(input(menu[4]) or '-1')
                if simMetric != 1 and simMetric != 2:
                    print('Invalid number. Try again.')

            if simMetric == 1:
                simMetric = 'Jaccard'
            else:
                simMetric = 'Signature'


        elif selection == 6:
            # select the similarity calculation method
            calcMethod = -1
            while calcMethod != 1 and calcMethod != 2:
                calcMethod = int(input(menu[5]) or '-1')
                if calcMethod != 1 and calcMethod != 2:
                    print('Invalid number. Try again.')

        if selection == 7:
            if perms != -1:
                # create or load the sig sim file
                loadFlagSig = False
                selSig = '-1'
                while selSig != '1' and selSig != '2':
                    selSig = input('Press (1) to load or (2) to create the signature permutations file: ') or '-1'
                    if selSig != '1' and selSig != '2':
                        print('Invalid choice, try again.')
                    else:
                        if selSig == '1':
                            loadFlagSig = True

                if loadFlagSig==True:
                    fileList = []
                    print('Available files in the directory:')
                    for fi in os.listdir():
                        if fi.endswith('signatureMatrix.pkl'):
                            fileList.append(fi)
                            print(fi)   
                    
                    fileSelection = ''
                    while fileSelection not in fileList:
                        fileSelection = input('Type the signatures file to load or <back> to go to the main menu:') or ''
                        if fileSelection == 'back':
                            break
                        if fileSelection not in fileList:
                            print('Wrong file name, try again.')
                    
                    if fileSelection == 'back':
                        print('Returning to main menu.')
                        continue

                    # Load signature matrix
                    with open(fileSelection,'rb') as f:
                        sig = pickle.load(f)
                    print('Loaded signature matrix from file '+fileSelection)
                
                else:
                    total_perms = -1
                    while total_perms <= 0 and total_perms < perms:
                        total_perms = int(input('Select the total number of signatures to be saved \
                        in the signature matrix (must be at least the same as the current number of permutations: '+str(perms)+'):') or '-1')
                        if total_perms <= 0 and total_perms < perms:
                            print('Invalid number. Try again.')

                    b = time.time()
                    sig = functions.MyMinHash(docs,total_perms, w=words)
                    print('MinHash took: ', time.time()-b)

                    # Store MinHash signature matrix to a file for later usage
                    with open(fl[:-4]+'_'+str(d)+'_documents_'+str(total_perms)+'_permutations_'+'signatureMatrix.pkl','wb') as f:
                        pickle.dump(sig, f)

                # only get the requested number of permutations 
                sig = sig[:perms] 

                sigmatrix = True # signature matrix is created/loaded
            else:
                print('Number of permutations is not set.')

        if selection == 8:
            if neigh != -1 and perms != -1 and simMetric != -1 and calcMethod != -1:
                # if LSH is chosen, choose the selected rows per band
                if calcMethod == 2 :
                    if (sigmatrix==False and simMetric == 'Signature'):
                        print('Cannot perform similarity calculations, signature matrix not created/loaded.')
                        continue

                    if rowsperband > 0:
                        print('Current rows per band:{0}, press 1 if you want to change it.'.format(rowsperband))
                        a = int(input('') or '0')
                        if a == 1:
                            rowsperband = -1

                    loadFlag = False
                    sel = '-1'
                    while sel != '1' and sel != '2':
                        sel = input('Press (1) to load or (2) to create the lsh file: ') or '-1'
                        if sel != '1' and sel != '2':
                            print('Invalid choice, try again.')
                        else:
                            if sel == '1':
                                loadFlag = True

                    if loadFlag == True:
                        fileList = []
                        print('Available files in the directory:')
                        for fi in os.listdir():
                            if fi.startswith('lsh'):
                                fileList.append(fi)
                                print(fi)   
                        
                        fileSelection = ''
                        while fileSelection not in fileList:
                            fileSelection = input('Type the lsh file to load or <back> to go to the main menu:') or ''
                            if fileSelection == 'back':
                                break
                            if fileSelection not in fileList:
                                print('Wrong file name, try again.')
                        
                        if fileSelection == 'back':
                            continue

                        with open(fileSelection,'rb') as f:
                            candidates = pickle.load(f)
                            
                        print('\nWARNING: using another file for the selected parameters might cause errors (like not matching document size)\n')
                        # check that each doc has at least the number of selected neighbors 
                        check_neighbors = functions.checkNeighbors(candidates=candidates,numNeighbors=neigh,msg=msg_warning)
                        
                        finalSel = ''
                        if check_neighbors == 0:
                                while finalSel != '1' and finalSel != '2':
                                    finalSel = input('Load file anyway (1), create it (2) or press <back> to go to the main menu: ') or ''
                                    if finalSel == 'back':
                                        break
                                    if finalSel != '1' and finalSel != '2':
                                         print('Invalid choice, try again.')
                                
                                if finalSel == '2':
                                    rowsperband = -1
                                    loadFlag == False
                                    continue
                                
                                elif finalSel == 'back':
                                    continue
                                
                                else: 
                                    lsh_created = True
                        

                    if loadFlag == False:
                        ex = ''
                        while rowsperband <= 0:
                            rowsperband = int(input(menu[6]))
                            if rowsperband <= 0:
                                print('Invalid number. Try again.')
                                continue

                            # calculate lsh candidates 
                            candidates = functions.lsh(sig=sig,rowsPerBand=rowsperband, w=words) 
                            
                            # check that each doc has at least the number of selected neighbors , if not rowsperband = -1
                            check_neighbors = functions.checkNeighbors(candidates=candidates,numNeighbors=neigh,msg=msg_warning)
                                
                            if check_neighbors == 0:
                                rowsperband = -1
                                ex = input('\nType <back> to go to main menu or ignore to select another rows per band number:')
                                if ex == 'back':
                                    break
                                continue

                            lsh_created = True

                            outFileName = 'lsh_candidates_'+str(d)+'_docs_'+str(perms)+'_perms_'+str(rowsperband)+'_rpb_'+fl[:-4]
                            print('Saving lsh candidate dictionary as: '+outFileName+'.pkl')
                            with open(outFileName+'.pkl','wb') as f:
                                pickle.dump(candidates, f)
                            
                        if ex == 'back':
                            continue

                    t0 = time.time()
                    if simMetric == 'Jaccard':
                        # calculate jaccard similarities for the candidates
                        sim_lsh = functions.calcJaccardSimListForLSH(candidates, docs)
                    else:
                        # calculate signature similarities for the candidates   
                        sim_lsh = functions.calcSignatureSimListForLSH(candidates, docs, signatureMatrix=sig, permutations=perms)


                    lsh_similarities = True

                    # combine candidate dictionary and similarity dictionary
                    combined_candidates_sims = functions.combineCandidateFilesWithSimilarities(candidates, sim_lsh) 
                    
                    # sort the neighbors based on their similarity
                    combined_candidates_sims_sorted = functions.sortCandidatesSimilarities(combined_candidates_sims)

                    # calculate average similarity
                    avg_sim_lsh = functions.avgSimLSH(numDocuments=d,candidate_files_sorted=combined_candidates_sims_sorted,numNeighbors=neigh)
                    print('Total time for calculation: '+str(time.time() - t0))
                    
                    print('Average '+str(simMetric)+' similarity using LSH: '+str(avg_sim_lsh))


                elif calcMethod == 1:
                    if simMetric == 'Jaccard':
                        # calc average jaccard similarity 
                        avgJacSim1 = functions.avgSim(d, docs, numNeighbors=neigh, simMeasure='Jaccard')
                        print('Average Jaccard similarity: '+ str(avgJacSim1))
                            
                    else:
                        if (sigmatrix==False):
                            print('Cannot perform similarity calculations, signature matrix not created/loaded.')
                            continue
                        # calc average sig similarity
                        avgJacSim2 = functions.avgSim(d, docs, numNeighbors=neigh, simMeasure='Signature',sig=sig,numSigPermutations=perms)
                        print('Average Signature similarity: '+ str(avgJacSim2))

            else:
                print('Cannot perform similarity calculations, not all parameters are set.')

        # compare two files with each other
        if selection == 9:
            # check if all parameters are set
            if fl != '' and d != -1 and neigh != -1 and perms != -1 and simMetric != -1 and calcMethod != -1 and sigmatrix == True:
                docId1 = -1
                docId2 = -1
                while docId1 < 1 or docId1 > d:
                    docId1 = int(input(menu[8]) or '-1')
                    if docId1 < 1 or docId1 > d:
                        print('Document 1 id out of range {0}-{1}. Choose again.'.format(1,d))

                while docId2 < 1 or docId2 > d:
                    docId2 = int(input(menu[9]) or '-1')
                    if docId2 < 1 or docId2 > d:
                        print('Document 2 id out of range {0}-{1}. Choose again.'.format(1,d))
                
                if calcMethod == 1:
                    jac_sim1 = functions.MyJacSimWithOrderedLists(docID1=docs[docId1-1],docID2=[docId2-1])
                    print('Jaccard Similarity between documents {0} and {1}s: {2}'.format(docId1,docId2,jac_sim1))

                    sig_sim1 = functions.MySigSim(sigList=list(zip(*sig)), docID1=docId1, docID2=docId2, numPermutations=perms)
                    print('Signature Similarity between documents {0} and {1}: {2}'.format(docId1,docId2,sig_sim1))

                else:
                    if lsh_created == True and lsh_similarities == True:
                        d1d2sim = sim_lsh.get(docId1).get(docId2)
                        print('{0} similarity between documents {1} and {2} with LSH: {3}'.format(simMetric,docId1,docId2,d1d2sim))

                        jac_sim1 = functions.MyJacSimWithOrderedLists(docID1=docs[docId1-1],docID2=[docId2-1])
                        print('Jaccard Similarity between documents {0} and {1}s: {2}'.format(docId1,docId2,jac_sim1))

                        sig_sim1 = functions.MySigSim(sigList=list(zip(*sig)), docID1=docId1, docID2=docId2, numPermutations=perms)
                        print('Signature Similarity between documents {0} and {1}: {2}'.format(docId1,docId2,sig_sim1))
                    
                    else:
                        print('LSH and LSH similarities are not created.')
                    



            else:
                print('Cannot compare two files, not all parameters are set OR the calculations of the necessary structures (sig matrix, lsh candidate docs etc) has not happened.')


        print('\n================================================================================')

        calcMethodS = 'Undefined'
        if calcMethod == 1:
            calcMethodS = 'Brute Force'
        elif calcMethod == 2:
            calcMethodS = 'LSH'

        print('\nSelections:\n File selected: {0}\n Number of documents: {1}\n Neighbors:{2}\n Permutations: {3}\n Similarity Metric: {4}\n Calculation Method: {5}\n Sig Matrix created/loaded: {6}'.format(fl,d,neigh,perms,simMetric,calcMethodS,sigmatrix))

        print('\n=============================')
