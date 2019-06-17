import numpy as np
import tensorflow as tf
import re
from kfac_helper import *



def searchApproxStratergy(factorTensors, varlist, config):

    factorTensors = searchHomogenousWeightsBiases(factorTensors, varlist, config['useHomogeneous'])

    factorTensors = searchWeightSharingThenConcat(factorTensors, varlist)
    ########
    ## label weight sharing types and approximation stratergy
    for param in factorTensors:
      if not factorTensors[param].has_key('KFACType'):
        fFac_list = factorTensors[param]['fpropFactors']
        bFac_list = factorTensors[param]['bpropFactors']
        if len(bFac_list) > 1:
          if len(bFac_list) == len(fFac_list):
            t_sample = fFac_list[0]
            #if 'RNN' in t_sample.name and 'Cell' in t_sample.name:
            factorTensors[param]['KFACType'] = 'RNN'
            #else:
              #factorTensors[param]['KFACType'] = 'IND'
          else:
            factorTensors[param]['KFACType'] = 'IND'
        else:
          factorTensors[param]['KFACType'] = 'STD'
        if factorTensors[param]['assnBias'] is not None:
          factorTensors[factorTensors[param]['assnBias']]['KFACType'] = factorTensors[param]['KFACType']
    ########
    #######
    ## label variable op types and approximation stratergy
    for param in factorTensors:
      opType = factorTensors[param]['opName']
      if opType in KFAC_OPS:
        if opType != 'BiasAdd':
          factorTensors
        
    ########
    return factorTensors

def searchHomogenousWeightsBiases(factorTensors, varlist, deleteBiasStats=False):
    ########
    ## check associated weights and bias for homogeneous coordinate representation
    ## and check redundent factors
    ## TO-DO: there may be a bug to detect associate bias and weights for forking layer, e.g. in inception models.
    for param in varlist:
      factorTensors[param]['assnWeights'] = None
      factorTensors[param]['assnBias'] = None
    for bias in varlist:
      ## find bias variables and search for its weight matrix
      if factorTensors[bias]['opName'] == 'BiasAdd':
        factorTensors[bias]['assnWeights'] = None
        for weights in varlist:
          if factorTensors[weights]['opName'] in homogenous_KFAC_OPS:
            if len(factorTensors[weights]['bpropFactors']) > 0:
              ### match with weights that have the same backpropagated gradients
              if (set(factorTensors[weights]['bpropFactors']) == set(factorTensors[bias]['bpropFactors'])) \
             and (len(factorTensors[weights]['fpropFactors'])>0):
                factorTensors[bias]['assnWeights'] = weights
                factorTensors[weights]['assnBias'] = bias
                factorTensors[bias]['bpropFactors'] = factorTensors[weights]['bpropFactors']
                if deleteBiasStats:
                  factorTensors[bias]['bpropFactors'] = []
                  factorTensors[bias]['fpropFactors'] = []
    ########
    return factorTensors

def searchWeightSharingThenConcat(factorTensors, varlist):
    ########
    ## concatenate/lump the additive gradients along the batch 
    ## dimension, for batch computation.
    for key in ['fpropFactors', 'bpropFactors']:
      for i,param in enumerate(varlist):
        if len(factorTensors[param][key]) > 0:
          if not factorTensors[param].has_key(key+'_concat'):
            name_scope = factorTensors[param][key][0].name.split(':')[0]
            with tf.name_scope(name_scope):
              if KFAC_DEBUG:
                print(param.name + ' ' + key)
                print(factorTensors[param][key])
              factorTensors[param][key+'_concat'] = tf.concat(axis=0, values=factorTensors[param][key])
            

        else:
          factorTensors[param][key+'_concat'] = None

        ## make sure the shared factors get a single concatenated tensor
        ## search over the rest of the factors 
        for j,param2 in enumerate(varlist[(i+1):]):
          if (len(factorTensors[param][key]) > 0) and (set(factorTensors[param2][key]) == set(factorTensors[param][key])):
             factorTensors[param2][key] = factorTensors[param][key]
             factorTensors[param2][key+'_concat'] = factorTensors[param][key+'_concat']
    ########
    ## add a unique identifier for each unique (factor_concat, op_meta_data)
    for key in ['fpropFactors', 'bpropFactors']:
      for param in factorTensors:
        if factorTensors[param].has_key(key+'_concat'):
          factor_op_meta = getOpMetaData(factorTensors[param]['op'], key)
          factorTensors[param][key+'_concat_id'] = (factorTensors[param][key+'_concat'], factor_op_meta)
    return factorTensors

def searchFactors(gradient, param, graph):
    ## hard coded search stratergy
    bpropOp = gradient.op
    bpropOp_name = bpropOp.name
    
    bTensors = []
    fTensors = []
    
    ## combining additive gradient, assume they are the same op type and indepedent 
    if 'AddN' in bpropOp_name:
        factors = []
        for g in gradient.op.inputs:
            factors.append(searchFactors(g, param, graph))
        op_names = [item['opName'] for item in factors]
        ### TO-DO: need to check all the attribute of the ops as well
        print(param.name)
        print(op_names)
        assert len(np.unique(op_names)) == 1, gradient.name+' is shared among different computation OPs'

        bTensors = reduce(lambda x,y:x+y, [item['bpropFactors'] for item in factors])
        if len(factors[0]['fpropFactors']) > 0:
          fTensors = reduce(lambda x,y:x+y, [item['fpropFactors'] for item in factors]) 
        fpropOp_name = op_names[0]
        fpropOp = factors[0]['op']
    else:
        fpropOp_name = re.search('gradientsSampled(_[0-9]+|)/(.+?)_grad', bpropOp_name).group(2)
        try:
          fpropOp = graph.get_operation_by_name(fpropOp_name)
          fpropOp_name = fpropOp.op_def.name 
          if fpropOp_name in KFAC_OPS:
              ### Known OPs
              ###
              bTensor_candidates = [i for i in bpropOp.inputs if 'gradientsSampled' in i.name]
              if KFAC_DEBUG:
                print '--------------'
                print(param.name)
                print('backward candidates')
                for item in bTensor_candidates:
                  print item.name
                  print item.get_shape()
                print '--------------'
              bTensor_candidates = cleanupCandidateList(bTensor_candidates)
              bTensor = bTensor_candidates[-1]
              ### restore shape information
              bTensorShape = fpropOp.outputs[0].get_shape()
              if bTensor.get_shape()[0].value == None:
                  bTensor.set_shape(bTensorShape)
              ###
              bTensors.append(bTensor)
              ###
              if fpropOp_name == 'BiasAdd':
                  fTensors =  []
              else:
                  fTensor_candidates = [i for i in fpropOp.inputs if param.op.name not in i.name]

                  if KFAC_DEBUG:
                    print('forward candidates')
                    for item in fTensor_candidates:
                      print item.name
                      print item.get_shape()
                    print '--------------'
                  fTensor_candidates = cleanupCandidateList(fTensor_candidates)
                  fTensors.append(fTensor_candidates[0])
          else:
              #print(bpropOp.name)
              #print(fpropOp.op_def.name)
              #print(gradient)
              #print(gradient.dense_shape)
              ### unknown OPs, block approximation used
              bInputsList = [i for i in bpropOp.inputs[0].op.inputs if 'gradientsSampled' in i.name if 'Shape' not in i.name.split('/')[-1]]
              if len(bInputsList) > 0:
                bTensor = bInputsList[0]
                bTensorShape = fpropOp.outputs[0].get_shape()
                if bTensor.get_shape()[0].value == None:
                    bTensor.set_shape(bTensorShape)
                bTensors.append(bTensor)
              fpropOp_name = 'UNK-'+fpropOp_name

        except:
          print('could not find fprop op for {}'.format(param.name))
          print(gradient.name)
          print(gradient)
          print([(i.name, i.get_shape()) for i in gradient.op.inputs])
          bpropOp_name = gradient.op.op_def.name
          fpropOp_name = KFAC_GRAD_OPS_LOOKUP[bpropOp_name]
          print(bpropOp_name)
          print(fpropOp_name)
          fpropOp = gradient.op
          if fpropOp_name in KFAC_OPS:
            bTensor_candidates = [i for i in gradient.op.inputs]
            bTensor = bTensor_candidates[-1]
            bTensors.append(bTensor)
            if fpropOp_name == 'BiasAdd':
                fTensors =  []
            else:
                fTensor_candidates = [i for i in gradient.op.inputs]
                if fpropOp_name == 'MatMul':
                  candidate_index = 0
                elif fpropOp_name == 'Conv2D':
                  candidate_index = 0

                print([i.name for i in fTensor_candidates])
                fTensor = fTensor_candidates[candidate_index]
                fTensors.append(fTensor)
          else:
            print("screwed")
            exit()

    return  {'opName':fpropOp_name, 'op':fpropOp, 'bop':bpropOp, 
             'fpropFactors':fTensors, 'bpropFactors':bTensors}

def getOpMetaData(op, factor_name):
  if (op.op_def.name == 'Conv2D' and factor_name == 'fpropFactors') \
   or (op.op_def.name == 'Conv2DBackpropInput' and factor_name == 'bpropFactors'):
    strides = tuple(op.get_attr("strides"))
    padding = op.get_attr("padding")
    return (strides, padding)
  else:
    return tuple()


def cleanupCandidateList(input_list):
  remove_keywords = ['Shape', '_shape', 'pack:']
  for remove_keyword in remove_keywords:
    input_list = [i for i in input_list if remove_keyword not in i.name.split('/')[-1]]
  return input_list
