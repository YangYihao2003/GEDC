/*!
    \file    gd_nn_interface.h
    \brief   definitions for the interface
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

#ifndef GD_NN_INTERFACE_H
#define GD_NN_INTERFACE_H
#pragma once

#include "gd_nn_tensor.h"
#include "gd_nn_layer.h"
#include "gd_nn_support.h"
#include "gd_nn_report.h"

#ifdef __cplusplus
extern "C" 
{
#endif

#undef DEBUG_LAYER
extern char* operator_name[];	
#define DEBUG_PRINT() \
	nn_uint8 layer_id = GET_VAL(nn_uint8, cur, 5); \
	printf("%d, %s start \n", layer_id, operator_name[layer_id]);
	
/* user struct define */
typedef struct {
    void* user_input;                                   /*!< user_input pointer */
    nn_uint32  user_input_size;                         /*!< user input data size */    
    void** user_output;                                  /*!< user_output pointer */
    nn_uint32  user_output_size;                        /*!< user output data size */
    const nn_uint8* model_paras_array;                  /*!< model paras array */
    operator_callback * operators_cb_array;             /*!< operators callback array */
    nn_handle  model_paras_array_dict;                  /*!< model_paras_array_dict */
    nn_model_report_struct * report_ptr;                /*!< report pointer */
} nn_model;

/* model init function */
error_type nn_model_init(nn_model* user_model);
/* model create function for every layer */
error_type nn_model_create(nn_uint8* model_paras,operator_callback nn_model_cb[], nn_layer** nn_layer_header);
/* model invoke function */
error_type nn_model_invoke(nn_model* user_model);
/* model invoke function for single model*/
error_type nn_model_single_invoke(nn_model* user_model);
/* model invoke function for muti branch model */
error_type nn_model_multi_invoke(nn_model* user_model);
/* model destory function */
void nn_model_destroy(nn_model* user_model);

#ifdef __cplusplus
}
#endif

#endif  /* GD_NN_INTERFACE_H */
