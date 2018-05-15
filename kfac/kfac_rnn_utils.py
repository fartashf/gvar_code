import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS


def compute_rnn_xstats(factorTensor_concat, factor, stats,
                         fprop=False, bprop=False, fops=None,  
                         stats_var=None, homogeneous_coord=False):
    fstats_list = factor['fpropFactors']
    bstats_list = factor['bpropFactors']

    batch_size = (tf.shape(fstats_list[0])[0])
    seqLength = tf.to_float(len(fstats_list))

    ## TO-DO: current assume the direction of 
    ##   the tensorlist is backwards in time.
    ## Construct X_tt and Xt

    ## TO-DO: redundent computation to pad the act twice
    if homogeneous_coord:
        stats_var_dim = int(stats_var.get_shape()[0])
        fpropFactor_size = int(factorTensor_concat.get_shape()[-1])
        if stats_var_dim == (fpropFactor_size+1):
          factorTensor_concat = tf.concat(axis=1, values=[factorTensor_concat, tf.ones([tf.shape(factorTensor_concat)[0],1])]) ## pad ones

    X = factorTensor_concat
    H_size = int(factorTensor_concat.get_shape()[1])

    direction = 'backward'## Xtt = BX_t + b + e 
    Xt = factorTensor_concat[:-batch_size] # X_t, X_19
    Xtt = factorTensor_concat[batch_size:]  # X_t+1, X_18
    #direction = 'forward'
    #Xtt = factorTensor_concat[:-batch_size] # X_t, e.g. X_19
    #Xt = factorTensor_concat[batch_size:]  # X_t-1, e.g. X_18
    ###########

    if len(stats_var.get_shape()) > 1: 
      ## xcovariance matrix
      N = tf.to_float(tf.shape(Xtt)[0])
      xcov = tf.matmul(Xtt, Xt, transpose_a=True)/N
      print('computing cross covariance')
    else:
      ## diagonal xcovariance matrix
      xcov = tf.reduce_mean(Xtt*Xt, 0)
      print('computing diagonal cross covariance')
              
    return xcov


def traceNormFunc(tensor, average=False):
  if len(tensor.get_shape())>1:
    diag_tensor = tf.diag_part(tensor)
  else:
    diag_tensor = tensor
  if average:
      return tf.reduce_mean(diag_tensor)
  else:
      return tf.reduce_sum(diag_tensor)

def compute_rnn_whiten_tensor(cov_f, xcov_f, cov_b, xcov_b, factor, stats, var, damping_coeff, use_KroneckerSum_dict):
    fstats_list = factor['fpropFactors']
    bstats_list = factor['bpropFactors']

    
    ## TO-DO: current assume the mini-batch size is the same across weight sharing 
    batch_size = (tf.shape(fstats_list[0])[0])
    H_size = int(fstats_list[0].get_shape()[1])
    seqLength = tf.to_float(len(fstats_list))
    ###########
        
    damping_correction = tf.sqrt(traceNormFunc(cov_f, True) / traceNormFunc(cov_b, True))

    cov_f_is_diagonal = len(cov_f.get_shape())==1
    cov_b_is_diagonal = len(cov_b.get_shape())==1
    results = []
    #with tf.device('/gpu:0'):
    for cov, xcov, d_corr, is_diagonal, is_act in [(cov_f, xcov_f, damping_correction, cov_f_is_diagonal, True),(cov_b, xcov_b, 1./damping_correction, cov_b_is_diagonal, False),]:
        ## linear time-series regression coefficient
        #print_name = var.name+'_act' if is_act else var.name+'_grad'
        #cov = print_tracenorm(cov, print_name+'_cov')
        #xcov = print_tracenorm(xcov, print_name+'_xcov')
        if not is_diagonal: 
          I = tf.diag(tf.ones([tf.shape(cov)[0]]))
          with tf.device('/gpu:0'):
            cov += I * d_corr * tf.sqrt(damping_coeff) ## factored damping for now and apply damping in eigen space later
            #cov_damped = cov + I * d_corr * tf.sqrt(damping_coeff)
          #inv_cov = tf.matrix_inverse(cov_damped)
          inv_cov = tf.matrix_inverse(cov)
          
          #inv_cov = tf.matrix_inverse(cov + I*traceNormFunc(xcov, True)) ## make sure B has largest eig < 1
          #inv_cov = tf.matrix_inverse(cov + I*(traceNormFunc(xcov, True)/traceNormFunc(cov, True))*tf.sqrt(damping_coeff)) ## make sure B has largest eig < 1
          with tf.device('/gpu:0'):
            B = tf.matmul(xcov, inv_cov)

          Btilde_inv = tf.matrix_inverse(I - B)

          with tf.device('/gpu:0'):
            cov_tilde = cov

            a = tf.matmul(tf.matmul(Btilde_inv, cov_tilde), Btilde_inv, transpose_b=True)

            c = tf.matmul(tf.matmul(B, cov_tilde), B, transpose_b=True)
            c = tf.matmul(tf.matmul(Btilde_inv, c), Btilde_inv, transpose_b=True)
        else:
          #### diagonal stats ########
          I = tf.ones([tf.shape(cov)[0]])
          cov += I*d_corr*tf.sqrt(damping_coeff)
          inv_cov = 1./cov
          B = xcov*inv_cov
          Btilde_inv = 1./(I - B)
          cov_tilde = cov
          a = Btilde_inv*cov_tilde*Btilde_inv
          c = a*B*B
        results.extend([a,c])


    a,c,b,d = results
    print('using jimmys option factored RNN approximaition second stage')
    ######## DEBUG########
    #a = print_tracenorm(a, var.name+'_a')
    #b = print_tracenorm(b, var.name+'_b')
    #c = print_tracenorm(c, var.name+'_c')
    #d = print_tracenorm(d, var.name+'_d')

    use_KroneckerSum_dict[var] = False
    return kroneckerSumABminusCD_approx(a,b,c,d)
    #use_KroneckerSum_dict[var] = 'option2'
    #return kroneckerSumABminusCD(a,b,c,d)


def compute_rnn_whiten_tensor_option2(cov_f, xcov_f, cov_b, xcov_b, factor, stats, var, damping_coeff, use_KroneckerSum_dict):
    fstats_list = factor['fpropFactors']
    bstats_list = factor['bpropFactors']

    
    ## TO-DO: current assume the mini-batch size is the same across weight sharing 
    batch_size = (tf.shape(fstats_list[0])[0])
    H_size = int(fstats_list[0].get_shape()[1])
    seqLength = tf.to_float(len(fstats_list))
    ###########
        
    damping_correction = tf.sqrt(traceNormFunc(cov_f, True) / traceNormFunc(cov_b, True))

    cov_f_is_diagonal = len(cov_f.get_shape())==1
    cov_b_is_diagonal = len(cov_b.get_shape())==1
    results = []
    #with tf.device('/gpu:0'):
    for cov, xcov, d_corr, is_diagonal, is_act in [(cov_f, xcov_f, damping_correction, cov_f_is_diagonal, True),(cov_b, xcov_b, 1./damping_correction, cov_b_is_diagonal, False),]:
        ## linear time-series regression coefficient
        #print_name = var.name+'_act' if is_act else var.name+'_grad'
        #cov = print_tracenorm(cov, print_name+'_cov')
        #xcov = print_tracenorm(xcov, print_name+'_xcov')
        if not is_diagonal: 
          I = tf.diag(tf.ones([tf.shape(cov)[0]]))
          with tf.device('/gpu:0'):
            cov += I * d_corr * tf.sqrt(damping_coeff) ## factored damping for now and apply damping in eigen space later
            #cov_damped = cov + I * d_corr * tf.sqrt(damping_coeff)
            cov_damped = cov + I * 1.
          #cov_damped = tf.Print(cov_damped, [cov_damped])
          inv_cov = tf.matrix_inverse(cov_damped)
          #inv_cov = tf.matrix_inverse(cov)
          
          #inv_cov = tf.matrix_inverse(cov + I*traceNormFunc(xcov, True)) ## make sure B has largest eig < 1
          #inv_cov = tf.matrix_inverse(cov + I*(traceNormFunc(xcov, True)/traceNormFunc(cov, True))*tf.sqrt(damping_coeff)) ## make sure B has largest eig < 1
          with tf.device('/gpu:0'):
            B = tf.matmul(xcov, inv_cov)

          Btilde_inv = tf.matrix_inverse(I - B)

          with tf.device('/gpu:0'):
            cov_tilde = cov

            a = tf.matmul(tf.matmul(Btilde_inv, cov_tilde), Btilde_inv, transpose_b=True)

            c = tf.matmul(tf.matmul(B, cov_tilde), B, transpose_b=True)
            c = tf.matmul(tf.matmul(Btilde_inv, c), Btilde_inv, transpose_b=True)
        else:
          #### diagonal stats ########
          I = tf.ones([tf.shape(cov)[0]])
          cov += I*d_corr*tf.sqrt(damping_coeff)
          inv_cov = 1./cov
          B = xcov*inv_cov
          Btilde_inv = 1./(I - B)
          cov_tilde = cov
          a = Btilde_inv*cov_tilde*Btilde_inv
          c = a*B*B
        results.extend([a,c])


    a,c,b,d = results
    print('using option2 factored RNN approximaition second stage')
    ######## DEBUG########
    #a = print_tracenorm(a, var.name+'_a')
    #b = print_tracenorm(b, var.name+'_b')
    #c = print_tracenorm(c, var.name+'_c')
    #d = print_tracenorm(d, var.name+'_d')

    #use_KroneckerSum_dict[var] = False
    #return kroneckerSumABminusCD_approx(a,b,c,d)
    use_KroneckerSum_dict[var] = 'option2'
    return kroneckerSumABminusCD(a,b,c,d)





#def compute_rnn_whiten_tensor_v2(cov_f, xcov_f, cov_b, xcov_b, factor, stats, var, damping_coeff, use_KroneckerSum_dict):
#    fstats_list = factor['fpropFactors']
#    bstats_list = factor['bpropFactors']
#
#    
#    ## TO-DO: current assume the mini-batch size is the same across weight sharing 
#    batch_size = (tf.shape(fstats_list[0])[0])
#    H_size = int(fstats_list[0].get_shape()[1])
#    seqLength = tf.to_float(len(fstats_list))
#    ###########
#    centerAct = False
#    ###########
#    stats_var_dim = int(cov_f.get_shape()[0])
#    if centerAct and stats_var_dim == (H_size+1):
#      mu_act = tf.expand_dims(cov_f[-1], 0)
#      M_act = tf.matmul(mu_act, mu_act, transpose_a = True)
#    ###########
#
#    
#    damping_correction = tf.sqrt(traceNormFunc(cov_f, True) / traceNormFunc(cov_b, True))
#
#    cov_f_is_diagonal = len(cov_f.get_shape())==1
#    cov_b_is_diagonal = len(cov_b.get_shape())==1
#    results = []
#    #with tf.device('/gpu:0'):
#    for cov, xcov, d_corr, centering, is_diagonal in [(cov_f, xcov_f, damping_correction, True, cov_f_is_diagonal),(cov_b, xcov_b, 1./damping_correction, False, cov_b_is_diagonal),]:
#        ## linear time-series regression coefficient
#        if not is_diagonal: 
#          I = tf.diag(tf.ones([tf.shape(cov)[0]]))
#          with tf.device('/gpu:0'):
#            if centerAct and centering:
#              cov -= M_act
#              xcov -= M_act
#            cov += I * d_corr * tf.sqrt(damping_coeff)
#          #inv_cov = tf.matrix_inverse(cov + I * 10.)
#          inv_cov = tf.matrix_inverse(cov)
#          with tf.device('/gpu:0'):
#            B = tf.matmul(xcov, inv_cov)
#
#          Btilde_inv = tf.matrix_inverse(I - B)
#
#          with tf.device('/gpu:0'):
#            if centerAct and centering:
#              cov_tilde = cov + M_act
#            else:
#              cov_tilde = cov
#
#            a = tf.matmul(tf.matmul(Btilde_inv, cov_tilde), Btilde_inv, transpose_b=True)
#
#            c = tf.matmul(tf.matmul(B, cov_tilde), B, transpose_b=True)
#            c = tf.matmul(tf.matmul(Btilde_inv, c), Btilde_inv, transpose_b=True)
#        else:
#          #### diagonal stats ########
#          I = tf.ones([tf.shape(cov)[0]])
#          cov += I*d_corr*tf.sqrt(damping_coeff)
#          inv_cov = 1./cov
#          B = xcov*inv_cov
#          Btilde_inv = 1./(I - B)
#          cov_tilde = cov
#          a = Btilde_inv*cov_tilde*Btilde_inv
#          c = a*B*B
#        results.extend([a,c])
#
#
#    a,c,b,d = results
#    print('using proper factored RNN approximaition second stage')
#
#    use_KroneckerSum_dict[var] = False
#    return kroneckerSumABminusCD_approx(a,b,c,d)
#    #use_KroneckerSum_dict[var] = True
#    #return kroneckerSumABminusCD(a,b,c,d)

def compute_rnn_whiten_tensor_option1(cov_f, xcov_f, cov_b, xcov_b, factor, stats, var, damping_coeff, use_KroneckerSum_dict):
    fstats_list = factor['fpropFactors']
    bstats_list = factor['bpropFactors']

    
    ## TO-DO: current assume the mini-batch size is the same across weight sharing 
    batch_size = (tf.shape(fstats_list[0])[0])
    H_size = int(fstats_list[0].get_shape()[1])
    seqLength = tf.to_float(len(fstats_list))
    ###########
    ## add in seq length later
    #cov_f /= tf.sqrt(seqLength)
    #xcov_f /= tf.sqrt(seqLength)
    #cov_b /= tf.sqrt(seqLength)
    #xcov_b /= tf.sqrt(seqLength)
    cov_b /= seqLength
    xcov_b /= seqLength
    ###########
        
    damping_correction = tf.sqrt(traceNormFunc(cov_f, True) / traceNormFunc(cov_b, True))

    cov_f_is_diagonal = len(cov_f.get_shape())==1
    cov_b_is_diagonal = len(cov_b.get_shape())==1
    results = []
    #with tf.device('/gpu:0'):
    for cov, xcov, d_corr, is_diagonal, is_act in [(cov_f, xcov_f, damping_correction, cov_f_is_diagonal, True),(cov_b, xcov_b, 1./damping_correction, cov_b_is_diagonal, False),]:
        ## linear time-series regression coefficient
        #cov = print_tracenorm(cov, print_name+'_cov')
        #xcov = print_tracenorm(xcov, print_name+'_xcov')
        if not is_diagonal: 
          I = tf.diag(tf.ones([tf.shape(cov)[0]]))
          #with tf.device('/gpu:0'):
          cov += I * d_corr * tf.sqrt(damping_coeff) ## factored damping for now and apply damping in eigen space later
          #cov_damped = cov + I * d_corr * tf.sqrt(damping_coeff) ## factored damping for now and apply damping in eigen space later
          #cov += I * d_corr * 0.01
          #cov += I * d_corr * 100000.
          ##### 
          #### compute cov^{-0.5}
          #cov_damped = tf.Print(cov_damped, [tf.convert_to_tensor('{}'.format(var.name)),cov])
          #cov_e, cov_Q = tf.self_adjoint_eig(cov_damped)
          cov_e, cov_Q = tf.self_adjoint_eig(cov)
          cov_e_sqrt = tf.expand_dims(tf.sqrt(cov_e),0)
          #cov_e_diag = tf.diag(tf.sqrt(cov_e))
          #cov_e_inv_diag = tf.diag(1./tf.sqrt(cov_e))
          with tf.device('/gpu:0'):
            #cov_inv_sqrt = tf.matmul(tf.matmul(cov_Q, cov_e_inv_diag), cov_Q, transpose_b=True)
            #cov_sqrt = tf.matmul(tf.matmul(cov_Q, cov_e_diag), cov_Q, transpose_b=True)
            cov_inv_sqrt = tf.matmul((cov_Q/cov_e_sqrt), cov_Q, transpose_b=True)
            cov_sqrt = tf.matmul((cov_Q*cov_e_sqrt), cov_Q, transpose_b=True)
          ####
          ## ensure the xcovar is symetric
          xcov_sym = (xcov + tf.transpose(xcov))/2.
          ##### 
          #####
          with tf.device('/gpu:0'):
            B = tf.matmul(tf.matmul(cov_inv_sqrt, xcov_sym), cov_inv_sqrt)
          #####
          B_e, U = tf.self_adjoint_eig(B) 
          #####
          ##### 
          Uhat = tf.matmul(cov_inv_sqrt, U) 
          #####
          results.extend([B_e, Uhat]) 

        else:
          #### diagonal stats ########
          I = tf.ones([tf.shape(cov)[0]])
          cov += I*d_corr*tf.sqrt(damping_coeff)
          cov_e = cov
          cov_e_sqrt = tf.sqrt(cov_e)
          cov_inv_sqrt = 1./cov_e_sqrt
          cov_sqrt = cov_e_sqrt
          xcov_sym = xcov ## it is definitely symetric in the diagonal approx
          B = cov_inv_sqrt * xcov_sym * cov_inv_sqrt
          B_e = B
          U = None
          Uhat = cov_inv_sqrt
          results.extend([B_e, Uhat])

    
    B_e_f,U_f,B_e_b,U_b = results
    print('using option 1 factored RNN approximaition second stage')
    ######## DEBUG########
    #a = print_tracenorm(a, var.name+'_a')
    #b = print_tracenorm(b, var.name+'_b')
    #c = print_tracenorm(c, var.name+'_c')
    #d = print_tracenorm(d, var.name+'_d')
    use_KroneckerSum_dict[var] = 'option1'
    return B_e_f,U_f,B_e_b,U_b 
    #use_KroneckerSum_dict[var] = False
    #return kroneckerSumABminusCD_approx(a,b,c,d)
    #use_KroneckerSum_dict[var] = True
    #return kroneckerSumABminusCD(a,b,c,d)



def compute_rnn_whiten_tensor_option2_real(cov_f, xcov_f, cov_b, xcov_b, factor, stats, var, damping_coeff, use_KroneckerSum_dict):
    fstats_list = factor['fpropFactors']
    bstats_list = factor['bpropFactors']

    
    ## TO-DO: current assume the mini-batch size is the same across weight sharing 
    batch_size = (tf.shape(fstats_list[0])[0])
    H_size = int(fstats_list[0].get_shape()[1])
    seqLength = tf.to_float(len(fstats_list))
    ###########
    ## add in seq length later
    #cov_f /= tf.sqrt(seqLength)
    #xcov_f /= tf.sqrt(seqLength)
    #cov_b /= tf.sqrt(seqLength)
    #xcov_b /= tf.sqrt(seqLength)
    cov_b /= seqLength
    xcov_b /= seqLength
    ###########
        
    damping_correction = tf.sqrt(traceNormFunc(cov_f, True) / traceNormFunc(cov_b, True))

    cov_f_is_diagonal = len(cov_f.get_shape())==1
    cov_b_is_diagonal = len(cov_b.get_shape())==1
    results = []
    #with tf.device('/gpu:0'):
    for cov, xcov, d_corr, is_diagonal, is_act in [(cov_f, xcov_f, damping_correction, cov_f_is_diagonal, True),(cov_b, xcov_b, 1./damping_correction, cov_b_is_diagonal, False),]:
        ## linear time-series regression coefficient
        #cov = print_tracenorm(cov, print_name+'_cov')
        #xcov = print_tracenorm(xcov, print_name+'_xcov')
        if not is_diagonal: 
          I = tf.diag(tf.ones([tf.shape(cov)[0]]))
          #with tf.device('/gpu:0'):
          cov += I * d_corr * tf.sqrt(damping_coeff) ## factored damping for now and apply damping in eigen space later
          cov_e, cov_Q = tf.self_adjoint_eig(cov)
          cov_e_sqrt = tf.expand_dims(tf.sqrt(cov_e),0)
          with tf.device('/gpu:0'):
            cov_inv_sqrt = tf.matmul((cov_Q/cov_e_sqrt), cov_Q, transpose_b=True)
            cov_sqrt = tf.matmul((cov_Q*cov_e_sqrt), cov_Q, transpose_b=True)
            B = tf.matmul(tf.matmul(cov_inv_sqrt, xcov), cov_inv_sqrt)
          #####
          BB_e, BB_E = tf.self_adjoint_eig(tf.matmul(B,B, transpose_a=True)) 
          #####
          #####
          results.extend([BB_e, BB_E, B, cov_inv_sqrt]) 

        else:
          #### diagonal stats ########
          I = tf.ones([tf.shape(cov)[0]])
          cov += I*d_corr*tf.sqrt(damping_coeff)
          cov_e = cov
          cov_e_sqrt = tf.sqrt(cov_e)
          cov_inv_sqrt = 1./cov_e_sqrt
          #cov_sqrt = cov_e_sqrt
          ## xcovar is definitely symetric in the diagonal approx
          B = cov_inv_sqrt * xcov * cov_inv_sqrt
          BB_e = tf.square(B)
          BB_E = None
          results.extend([BB_e, BB_E, B, cov_inv_sqrt])

    
    BB_e_f,BB_E_f,B_f, cov_inv_sqrt_f, BB_e_b,BB_E_b,B_b, cov_inv_sqrt_b = results
    print('using real option 2 factored RNN approximaition second stage')
    ######## DEBUG########
    use_KroneckerSum_dict[var] = 'option2'
    return BB_e_f,BB_E_f,B_f, cov_inv_sqrt_f / tf.pow(seqLength, 1./4), BB_e_b,BB_E_b,B_b, cov_inv_sqrt_b / tf.pow(seqLength, 1./4)
    #use_KroneckerSum_dict[var] = False
    #return kroneckerSumABminusCD_approx(a,b,c,d)
    #use_KroneckerSum_dict[var] = True
    #return kroneckerSumABminusCD(a,b,c,d)






def kroneckerSumABminusCD_approx(A,B,C,D):
    pi = tf.sqrt(traceNormFunc(A)*traceNormFunc(D) / traceNormFunc(B) / traceNormFunc(C))
    factor1 = A + C*pi
    factor2 = B - D/pi
    if len(factor1.get_shape()) == 2:
      S1, E1 = tf.self_adjoint_eig(factor1)
    else:
      S1, E1 = factor1, None
    if len(factor2.get_shape()) == 2:
      S2, E2 = tf.self_adjoint_eig(factor2)
    else:
      S2, E2 = factor2, None
    #S1 = print_minimum(S1, 'S1')
    #S2 = print_minimum(S2, 'S2')
    return S1, E1, S2, E2


def kroneckerSumABminusCD(A,B,C,D):
    if len(A.get_shape()) == 1:
      #### diagonal
      A_e, A_Q = A, None
      A_inv_half = 1./tf.sqrt(A_e)
      ACA = C/A
      S1, E1 = ACA, None
      K1 = A_inv_half
    else:
      A_e, A_Q = tf.self_adjoint_eig(A)
      A_inv_half = tf.matmul(tf.matmul(A_Q, tf.diag(1./tf.sqrt(A_e))), A_Q, transpose_b=True)
      ACA = tf.matmul(tf.matmul(A_inv_half, C), A_inv_half)
      S1, E1 = tf.self_adjoint_eig(ACA)
      K1 = tf.matmul(A_inv_half, E1)

    if len(B.get_shape()) == 1:
      #### diagonal
      B_e, B_Q = B, None
      B_inv_half = 1./tf.sqrt(B_e)
      BDB = D/B
      S2, E2 = BDB, None
      K2 = B_inv_half
    else:
      B_e, B_Q = tf.self_adjoint_eig(B)
      B_inv_half = tf.matmul(tf.matmul(B_Q, tf.diag(1./tf.sqrt(B_e))), B_Q, transpose_b=True)
      BDB = tf.matmul(tf.matmul(B_inv_half, D), B_inv_half)
      S2, E2 = tf.self_adjoint_eig(BDB)
      K2 = tf.matmul(B_inv_half, E2)
    
    return S1, K1, S2, K2

def rnn_geometric_sum(coeffs, seqLength):
  term1 = 1- coeffs**2
  term2 = seqLength * term1 - 2 * coeffs * (1 - coeffs**seqLength)
  return term2/term1

def print_minimum(t, name):
  t_min = tf.reduce_min(t)
  t = tf.Print(t,[tf.convert_to_tensor('{} minimum val:'.format(name)),t_min])
  return t

def print_tracenorm(t, name):
  t_tracenorm_avg = traceNormFunc(t, True)
  t = tf.Print(t,[tf.convert_to_tensor('{} trace norm:'.format(name)),t_tracenorm_avg])
  return t
