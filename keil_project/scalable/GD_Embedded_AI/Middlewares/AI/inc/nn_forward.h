/*!
    \file    nn_forward.h
    \brief   definitions for the forward
*/

/*
    Copyright (c) 2023, GigaDevice Semiconductor Inc.

    Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/

#ifndef NN_FORWARD_H
#define NN_FORWARD_H

#include "gd_nn_interface.h"
#include "arm_nnfunctions.h"
#include "nn_operators_func_f32.h"

#ifdef __cplusplus
extern "C" 
{
#endif

/* conv func cb */
error_type nn_wrapper_conv_2d_func(struct nn_layer* layer);
/* depconv func cb */
error_type nn_wrapper_depthconv2d_func(struct nn_layer* layer);
/* fc func cb */
error_type nn_wrapper_fc_func(struct nn_layer* layer);
/* maxpool func cb */
error_type nn_wrapper_maxpool2d_func(struct nn_layer* layer);
/* avepool func cb */
error_type nn_wrapper_avgpool2d_func(struct nn_layer* layer);
/* padding func cb */
error_type nn_wrapper_pad_func(struct nn_layer* layer);
/* padv2 func cb */   
error_type nn_wrapper_padv2_func(struct nn_layer* layer);
/* transpose func cb */  
error_type nn_wrapper_transpose_func(struct nn_layer* layer);
/* softmax func cb */
error_type nn_wrapper_softmax_func(struct nn_layer* layer);
/* reshape func cb */
error_type nn_wrapper_reshape_func(struct nn_layer* layer);
/* op_quan func cb */
error_type nn_wrapper_op_quantization_func(struct nn_layer* layer);
/* op_dequan func cb */
error_type nn_wrapper_op_dequantization_func(struct nn_layer* layer);
/* expand_dims func cb */
error_type nn_wrapper_expand_dims_func(struct nn_layer* layer);
/* add func cb */
error_type nn_wrapper_add_func(struct nn_layer* layer);
/* sub func cb */
error_type nn_wrapper_sub_func(struct nn_layer* layer);
/* div func cb */
error_type nn_wrapper_div_func(struct nn_layer* layer);
/* concatenation func cb */
error_type nn_wrapper_concatenation_func(struct nn_layer* layer);
/* gather func cb */
error_type nn_wrapper_gather_func(struct nn_layer* layer);
/* pack func cb */
error_type nn_wrapper_pack_func(struct nn_layer* layer);
/* logistic func cb */
error_type nn_wrapper_logistic_func(struct nn_layer* layer);
/* leaky_relu func cb */
error_type nn_wrapper_leaky_relu_func(struct nn_layer* layer);
/* tanh func cb */
error_type nn_wrapper_tanh_func(struct nn_layer* layer);
/* mean func cb */
error_type nn_wrapper_mean_func(struct nn_layer* layer);
/* resize_nearest_neighbor func cb */
error_type nn_wrapper_resize_nearest_neighbor_func(struct nn_layer* layer);
/* resize_bilinear func cb */
error_type nn_wrapper_resize_bilinear_func(struct nn_layer* layer);
/* mul_func cb */
error_type nn_wrapper_mul_func(struct nn_layer* layer);
/* minimum_func cb */
error_type nn_wrapper_minimum_func(struct nn_layer* layer);
/* maximum_func cb */
error_type nn_wrapper_maximum_func(struct nn_layer* layer);
/* relu_func cb */
error_type nn_wrapper_relu_func(struct nn_layer* layer);
/* depth_to_space func cb */
error_type nn_wrapper_depth_to_space_func(struct nn_layer* layer);
/* space_to_depth func cb */
error_type nn_wrapper_space_to_depth_func(struct nn_layer* layer);
/* split func cb */
error_type nn_wrapper_split_func(struct nn_layer* layer);
/* splitv func cb */
error_type nn_wrapper_splitv_func(struct nn_layer* layer);
/* reduceMax func cb */
error_type nn_wrapper_reduceMax_func(struct nn_layer* layer);
/* UnidirectionalSequenceLSTM init */
error_type nn_UnidirectionalSequenceLSTM_init(struct nn_layer* layer, OpDataLSTM* params_ref, cmsis_nn_lstm_params* params_cmsis_nn);
/* BidirectionalSequenceLSTM init */
error_type nn_BidirectionalSequenceLSTM_init(struct nn_layer* layer, OpDataLSTM* params_ref1, OpDataLSTM* params_ref2,\
                        cmsis_nn_lstm_params* params_cmsis_nn1, cmsis_nn_lstm_params* params_cmsis_nn2);
/* UnidirectionalSequenceLSTM cb */
error_type nn_wrapper_UnidirectionalSequenceLSTM_func(struct nn_layer* layer);
/* BidirectionalSequenceLSTM cb */
error_type nn_wrapper_BidirectionalSequenceLSTM_func(struct nn_layer* layer);
/* abs func cb */
error_type nn_wrapper_abs_func(struct nn_layer* layer);
/* prelu func cb */
error_type nn_wrapper_prelu_func(struct nn_layer* layer);
/* log func cb */
error_type nn_wrapper_log_func(struct nn_layer* layer);
/* rsqrt func cb */
error_type nn_wrapper_rsqrt_func(struct nn_layer* layer);
/* pow func cb */
error_type nn_wrapper_pow_func(struct nn_layer* layer);
/* sqrt func cb */
error_type nn_wrapper_sqrt_func(struct nn_layer* layer);
/* gatherNd func cb */
error_type nn_wrapper_gatherNd_func(struct nn_layer* layer);
/* squareddifference func cb */
error_type nn_wrapper_squareddifference_func(struct nn_layer* layer);
/* sign func cb */
error_type nn_wrapper_sign_func(struct nn_layer* layer);
/* slice func cb */
error_type nn_wrapper_slice_func(struct nn_layer* layer);
/* Transpose_conv_2d func cb */
error_type nn_wrapper_Transpose_conv_2d_func(struct nn_layer* layer);
/* relu6 func cb */
error_type nn_wrapper_relu6_func(struct nn_layer* layer);
/* stridedslice func cb */
error_type nn_wrapper_stridedslice_func(struct nn_layer* layer);
/* exp func cb */
error_type nn_wrapper_exp_func(struct nn_layer* layer);
/* sum func cb */
error_type nn_wrapper_sum_func(struct nn_layer* layer);
/* batch_matmul func cb */
error_type nn_wrapper_batch_matmul_func(struct nn_layer* layer);
/* unpack func cb */
error_type nn_wrapper_unpack_func(struct nn_layer* layer);
/* log softmax func cb */
error_type nn_wrapper_log_softmax_func(struct nn_layer* layer);
/* mirror_pad func cb */
error_type nn_wrapper_mirror_pad_func(struct nn_layer* layer);
/* gelu func cb */
error_type nn_wrapper_gelu_func(struct nn_layer* layer);
/* sin func cb */
error_type nn_wrapper_sin_func(struct nn_layer* layer);
/* cos func cb */
error_type nn_wrapper_cos_func(struct nn_layer* layer);

#ifdef __cplusplus
}
#endif

#endif  /* NN_FORWARD_H */
