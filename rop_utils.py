import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import h5py
import sys
sys.path.append('./iMIMIC-RCVs/')
sys.path.append('./iMIMIC-RCVs/rcv_utils.py')
sys.path.append('./iMIMIC-RCVs/scripts/keras_vis_rcv/')
sys.path.append('./iMIMIC-RCVs/scripts/keras_vis_rcv/vis/')

import rcv_utils
import keras
import keras.backend as K
import sklearn.model_selection
import sklearn.linear_model
PWD=''

def swap(img):
    #new_im = np.zeros((224,224,3))
    #new_im[:,:,0]=img[2,:,:]
    #new_im[:,:,1]=img[1,:,:]
    #new_im[:,:,2]=img[0,:,:]
    return np.transpose(img, (1,2,0))

def get_activations(inputs, model, layer, batch_size):
    get_layer_output = K.function([model.layers[0].input],
                              [model.get_layer(layer).output])
    feats = get_layer_output([inputs])
    return feats[0 ]
    
def get_concept_measure(meas, file_name, concept):
    if file_name[:-4] in meas:
        return meas[file_name[:-4]][concept]
    return 'Nan'

def print_info(name, obj):
    print name 

def load_rop_data(file_name):
    h5file = file_name
    db = h5py.File(os.path.join(PWD, h5file), 'r')
    os.path.join(PWD, h5file)
    db.visititems(print_info)
    images = db['data']
    labels = db['labels']
    original_files = db['original_files']
    classes = [get_sample_class(labels, i) for i in range(len(labels))]
    return images, labels, original_files, classes

def get_sample_class(labels, i):
    if labels[i]=='No':
        return 0
    if labels[i]=='Plus':
        return 2
    return 1

def show_image(images, i):
    plt.imshow(images[i][0,:,:], cmap='gray')

def py_ang(v1, v2):
    cos = np.dot(v1,v2)
    return np.arccos(cos/(np.linalg.norm(v1) * np.linalg.norm(v2)))

def analize_angles(splits, layer, max_repetition, rop_class, meas, original_files, acts, labels, concept):
    angles = []
    datasizes=[]
    split = 0
    print splits[split]
    repetition=0
    c = concept
    while split < len(splits):
        print 'Extracting RCV at layer: ', layer
        acts = np.load('./rcv/phis/0_concepts_phis_'+layer+'.npy')
        #for c in concepts:
        #    if c not in rcvs.keys():
        #        rcvs[c] = {}
        #        rscores[c] = {}
        print 'Analysing concept: ', c
        fvec=[]
        to_keep=[]
        classes=[]
        fvec, to_keep = get_concept_measures_vector(meas, original_files, c)
        X=(np.asarray([acts[i].ravel() for i in to_keep], dtype=np.float64))
        classes = [get_sample_class(labels, i) for i in to_keep]
        idxs = get_class_indices(classes, rop_class)
        angle, datasize = solve_regression_angles(
                                np.squeeze(X[idxs], axis=1), 
                                np.squeeze(fvec[idxs], axis=1), 
                                splits[split]
                                )
        angles.append(angle)
        datasizes.append(datasize)
        split += 1
    return angles, datasizes

def solve_regression(inputs, y, n_splits=3, n_repeats=1, random_state=12883823, verbose=1):
    scores=[]
    max_score = 0
    direction = None
    dirs=[]
    rkf = sklearn.model_selection.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    counter = 0
    for train, test in rkf.split(inputs):
        if verbose:
            print 'N. ', counter, '..'
            print len(inputs[train])
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(inputs[train], y[train])
        trial_score = reg.score(inputs[test], y[test])
        dirs.append(reg.coef_)
        #print 'y[train]', y[train]
        scores.append(trial_score)
        if trial_score > max_score:
            direction = reg.coef_
        if verbose:
            print trial_score
        counter += 1
    if verbose:
        print np.mean(scores)
        i=0
        while i+1<len(dirs):
            print 'angle: ', py_ang(dirs[i], dirs[i+1])
            i+=1
    return np.mean(scores), direction

def solve_regression_angles(inputs, y, n_splits, n_repeats=1, random_state=12883823, verbose=1):
    scores=[]
    angles=[]
    datasizes=[]
    max_score = 0
    direction = None
    dirs=[]
    length=len(inputs)/n_splits
    train1 = train2 = np.zeros((length,))
    idx = np.arange(len(inputs))
    np.random.shuffle(idx)
    train1 = idx[:length]
    train2 = idx[length:length*2]
    #import pdb; pdb.set_trace()    
    if verbose:
        #print 'N. ', counter, '..'
        print length
        datasizes.append(length)
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(inputs[train1], y[train1])
    dir1 = reg.coef_
    del reg
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(inputs[train2], y[train2])
    dir2 = reg.coef_
    if verbose:
        #print np.mean(scores)
        i=0
        #while i+1<len(dirs):
        angle = py_ang(dir1, dir2)
        print 'angle: ', angle
    return angle, length

def solve_regression_(inputs, y, n_splits=2, n_repeats=1, random_state=12883823, verbose=1):
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(inputs, y)
    return reg.coef_

def get_concept_measures_vector(meas, original_files, concept):
    fvec= np.array([get_concept_measure(meas, original_files[i], concept) for i in range(len(original_files))], dtype=np.float32)
    to_remove = np.argwhere(np.isnan(fvec))
    to_remove = [to_remove[i][0] for i in range(len(to_remove))]
    to_keep = list(set(range(len(fvec)))-set(to_remove))
    fvec = fvec[to_keep]
    return fvec, to_keep

def compute_regression(max_rep, layers, meas, original_files, acts, labels, concepts):
    regression_outputs = {}
    for repetition in range(max_rep):
        for l in layers[:]:
            print 'Layer: ', l
            acts=np.load('./rcv/phis/0_concepts_phis_'+l+'.npy')    
            print acts.shape
            
            if l not in regression_outputs.keys():
                regression_outputs[l]={}
            for c in concepts:
                if c not in regression_outputs[l].keys():
                    regression_outputs[l][c]={}
                    regression_outputs[l][c]['Normal']=[]
                    regression_outputs[l][c]['PrePlus']=[]
                    regression_outputs[l][c]['Plus']=[]
                fvec=[]
                to_keep=[]
                classes=[]
                fvec, to_keep = get_concept_measures_vector(meas, original_files, c)
                X=(np.asarray([acts[i].ravel() for i in to_keep], dtype=np.float64))
                classes = [get_sample_class(labels, i) for i in to_keep]
                idx_preplus = np.argwhere(np.array(classes)==1)
                idx_normal = np.argwhere(np.array(classes)==0)
                idx_plus = np.argwhere(np.array(classes)==2)
                
                reg_score, cv = solve_regression(
                            np.squeeze(X[idx_preplus], axis=1), 
                            np.squeeze(fvec[idx_preplus], axis=1), 
                            random_state=12345
                )

                nreg_score, ncv = solve_regression(
                            np.squeeze(X[idx_normal], axis=1), 
                            np.squeeze(fvec[idx_normal], axis=1), 
                            random_state=12345
                )
                
                preg_score, pcv = solve_regression(
                            np.squeeze(X[idx_plus], axis=1), 
                            np.squeeze(fvec[idx_plus], axis=1), 
                            random_state=12345
                )
                print c, ': '
                print 'images with concept measures: ', len(X)
                print 'Normal: ', classes.count(0), 'Pre-plus: ', classes.count(1), 'Plus: ', classes.count(2) 
                print 'regression output: ', nreg_score, ' (normal)',  reg_score, ' (preplus)', preg_score, '(plus)'
                regression_outputs[l][c]['Normal'].append(max(-5e-100, nreg_score))
                regression_outputs[l][c]['PrePlus'].append(max(-5e-100, reg_score))
                regression_outputs[l][c]['Plus'].append(max(-5e-100, preg_score))

    return regression_outputs

def get_class_indices(classes, rop_class):
    if rop_class=='Normal':
        return np.argwhere(np.array(classes)==0)
    if rop_class == 'Plus':
        return np.argwhere(np.array(classes)==2)
    return np.argwhere(np.array(classes)==1)

def get_rcv(layer, max_repetition, rop_class, meas, original_files, acts, labels, concepts):
    rcvs = {}
    rscores = {}
    repetition = 0
    while repetition < max_repetition:
        print 'Extracting RCV at layer: ', layer
        acts = np.load('./rcv/phis/0_concepts_phis_'+layer+'.npy')
        for c in concepts:
            if c not in rcvs.keys():
                rcvs[c] = {}
                rscores[c] = {}
            print 'Analysing concept: ', c
            fvec=[]
            to_keep=[]
            classes=[]
            fvec, to_keep = get_concept_measures_vector(meas, original_files, c)
            X=(np.asarray([acts[i].ravel() for i in to_keep], dtype=np.float64))
            classes = [get_sample_class(labels, i) for i in to_keep]
            idxs = get_class_indices(classes, rop_class)
            reg_score, cv = solve_regression(
                        np.squeeze(X[idxs], axis=1), 
                        np.squeeze(fvec[idxs], axis=1), 
                        random_state=repetition
            )
            del cv
            cv = solve_regression_(
                                    np.squeeze(X[idxs], axis=1), 
                                    np.squeeze(fvec[idxs], axis=1), 
                                    
                                    )
            print 'Regression output: ', reg_score, ' (', rop_class, ')'
            if rop_class not in rcvs[c].keys():
                rcvs[c][rop_class]=[]
                rscores[c][rop_class]=[]
            rcvs[c][rop_class].append(cv)
            rscores[c][rop_class].append(reg_score)
        repetition += 1
    return rcvs, rscores

def plot_dynamics(regression_outputs, rop_class, concepts, layers):
    for c in concepts:
        plt.plot([regression_outputs[l][c][rop_class][0] for l in layers])
    plt.legend(concepts)
    plt.xticks(np.arange(len(layers)),layers, rotation = '45')
    plt.ylabel('R^2')
    plt.xlabel('concepts')
    plt.title('Determination Coefficient, ' + rop_class)


def compute_sensitivity_scores(concept, rcv, model, layer, rop_class, repetition, inputs, meas, original_files, acts, labels ):
    r = 0 # iterating this only once atm
    print 'Computing sensitivity scores of concept ', concept, 'on samples from class ', rop_class
    rcv /= np.linalg.norm(rcv)
    #classes = [get_sample_class(labels, i) for i in range(len(inputs))]
    #idxs = get_class_indices(classes, rop_class)
    #for sample in range(len(idxs[:])):
    for sample in range(len(inputs)):
        #print sample
        derivative = tcav_utils.compute_tcav(
                                            model, -1, 0,
                                            np.expand_dims(inputs[sample], axis=0), 
                                            wrt_tensor=model.get_layer(layer).output
                                            )
        flattened_derivative = derivative.ravel()
        score = np.multiply(-1, np.dot(flattened_derivative, rcv))
        filet=open('plus_tcav_'+concept+'_'+str(r)+'.txt', 'a')
        filet.write(str(repetition)+','+str(score)+'\n')
        filet.close() 
