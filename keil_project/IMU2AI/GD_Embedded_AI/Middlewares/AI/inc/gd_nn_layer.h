/*!
    \file    gd_nn_layer.h
    \brief   definitions for the layer
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

#ifndef GD_NN_LAYER_H
#define GD_NN_LAYER_H
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" 
{
#endif

#include "gd_nn_types.h"
#include "gd_nn_basic_types.h"
#include "gd_nn_tensor.h"

/* constant var */

/* layer struct */
typedef struct nn_layer nn_layer;
/* operator_callback function pointer*/
typedef error_type (*operator_callback)(struct nn_layer*);

/* layer init function */
error_type nn_layer_init(nn_layer* layer,nn_uint8* paras);
/* layer struct data format init */
error_type nn_layer_struct_init(nn_layer* layer,nn_uint8* paras);
/* layer weight and bias data setting */
error_type nn_layer_set_weight_bias_address(nn_layer* layer);
/* layer special data setting */
error_type nn_layer_set_special_data_address(nn_layer* layer);
/* layer UnidirectionalSequenceLSTM_special_data setting */
error_type nn_layer_set_UnidirectionalSequenceLSTM_special_data_address(nn_layer* layer);
/* layer bidirectionalSequenceLSTM_special_data setting */
error_type nn_layer_set_BidirectionalSequenceLSTM_special_data_address(nn_layer* layer);
/* layer set integer lstm param by multiplier_shift_data setting */
error_type nn_layer_set_integer_lstm_param_by_multiplier_shift_data(nn_layer* layer);
/* layer set integer bidlstm param by multiplier_shift_data setting */
error_type nn_layer_set_integer_bidlstm_param_by_multiplier_shift_data(nn_layer* layer);
/* layer multi and shfit data setting */
error_type nn_layer_set_multiplier_shift_data_address(nn_layer* layer);
/* layer destory */
void nn_layer_destroy(nn_layer* layer);

#ifdef __cplusplus
}
#endif

#endif /* GD_NN_LAYER_H */
