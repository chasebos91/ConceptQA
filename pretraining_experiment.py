
import numpy as np
from soft_CE_loss import *
from Concept_Module import *
from Concept_Module import pretrain, train, evaluate

from Noam import NoamOptim
from batching import *
from pretraining_experiment import *
from matplotlib import pyplot as plt
import torch
import numpy as np
import time
import random


def pretrain_outer(pt_model, batches, optim, crit):

    model_file = "cat_pretrained"
    epochs = 50
    
    prev = 1000
    #remove
    #batches = [[batches[0][0]]]
    cat_ll = []
    for e in range(epochs):
        start = time.time()
        random.shuffle(batches)
        num_batches = len(batches)
        batches = batches[:num_batches-2]
        val_batch = batches[-1]
        for b in batches:
            cat_ll.append(pretrain(pt_model, b, optim, crit))
        #TODO:make train and val REMEMBER to remove the index on batches
        val_loss, preds = evaluate(pt_model, val_batch, crit, pt=True)
        if val_loss < prev:
            prev = val_loss
            best_model = pt_model
        #scheduler.step()
        elapsed = time.time() - start
        print("Epoch:", e, "Elapsed:", elapsed, val_loss)
        start = time.time()


    torch.save(best_model.state_dict(), model_file)

    vanilla_model = copy.deepcopy(best_model)
    soft_model = copy.deepcopy(best_model)
    vanilla_model.fine_tune()
    soft_model.fine_tune()
    return vanilla_model, soft_model, cat_ll



def finetune(vanilla_model, soft_model, vanilla_batches, soft_batches, s_optim, v_optim, soft_crit, crit):
    epochs = 50
    ##remove
    s_prev = 1000
    v_prev = 1000
    #vanilla_batches = [[vanilla_batches[0][0]]]
    #soft_batches = [[soft_batches[0][0]]]
    s_ll = []
    v_ll = []
    for e in range(epochs):
        start = time.time()
        for s, v in zip(soft_batches, vanilla_batches):

            v_ll.append(train(vanilla_model, v, v_optim, crit))
            s_ll.append(train(soft_model, s, s_optim, soft_crit, soft=True))
        #T
        v_val_loss, v_preds = evaluate(vanilla_model, v, crit)
        s_val_loss, s_preds = evaluate(soft_model, s, soft_crit, soft=True)
        if v_val_loss < v_prev:
            prev = v_val_loss
            v_best_model = vanilla_model
        if s_val_loss < s_prev:
            prev = s_val_loss
            s_best_model = soft_model
        # scheduler.step()
        elapsed = time.time() - start
        print("Epoch:", e, "Elapsed:", elapsed, "Vanilla loss: ", v_val_loss, "Soft loss: ", s_val_loss)
        
        start = time.time()
    return v_best_model, s_best_model, v_ll, s_ll



