/*!
    \file    gd_nn_tensor.h
    \brief   definitions for the tensor
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

#ifndef GD_NN_TENSOR_H
#define GD_NN_TENSOR_H
#pragma once

#include "gd_nn_basic_types.h"
#include "gd_nn_types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" 
{
#endif

/* tensor structrue */           
typedef struct {                         
    nn_uint8	                    tensor_id;                      /*!< tensor index */
    nn_uint8	                    dims_count;                     /*!< dims_count */
    nn_uint8	                    tensor_format;                  /*!< format */   
    nn_uint8	                    buffer_pool_index;              /*!< buffer_pool_index */   
    nn_dims	                        tensor_dims;                    /*!< tensor shape */
    nn_uint32	                    offset;                         /*!< tensor data offset */   
    nn_uint32                       dim_size;                       /*!< dim_size offset */   
    nn_uint8*	                    data;                           /*!< tensor data pointer */    
}nn_tensor;

/* tensor create function */
nn_tensor* nn_tensor_struct_create(nn_uint8 tensor_num);
/* tensor struct init function */
error_type nn_tensor_struct_init(nn_tensor* tensor, nn_uint8* paras);

#ifdef __cplusplus
}
#endif

#endif /* GD_NN_TENSOR_H */
