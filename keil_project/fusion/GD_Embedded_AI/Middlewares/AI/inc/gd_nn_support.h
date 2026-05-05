/*!
    \file    gd_nn_support.h
    \brief   definitions for the support
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

#ifndef GD_NN_SUPPORT_H
#define GD_NN_SUPPORT_H
#pragma once

#include "gd_nn_basic_types.h"
#include "gd_nn_types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" 
{
#endif

typedef struct {
    
    nn_uint8* lstm_cell_state_ptr1;
    nn_uint8* lstm_out_state_ptr1;
    nn_uint8* lstm_cell_state_ptr2;
    nn_uint8* lstm_out_state_ptr2;
    
    nn_dims	cell_state_tensor_dims;
    nn_dims	out_state_tensor_dims;
    
    uint32_t lstm_flag; /* 0 is undirctional lstm, 1: bidirection lstm */

}lstm_status_info;

    
/* start_flag_check */
error_type nn_start_flag_check(nn_uint8* paras);
/* end_flag_check */
error_type nn_end_flag_check(nn_uint8* paras);
/* version_check */
error_type nn_version_check(nn_uint8* paras);
/* platform_check */
error_type nn_platform_check(nn_uint8* paras);
/* signature_check */
error_type nn_signature_check(nn_uint8* paras);

#if defined(USING_DMA_COPY_LARGE_WEIGHT) && (defined(GD32H7XX) || defined(GD32H77D) || defined(GD32H77E))

#include "gd32h7xx.h"

/* dma_config_memory to memory */
void dma_config_m2m(uint32_t source);
/* by block sent dma data memory to memory */
void dma_start_send_block_m2m(const uint8_t* src, uint8_t* txbuffer, uint32_t len);
/* dma_start_send memory to memory */
uint8_t dma_start_send_m2m(uint8_t* source, uint8_t* txbuffer, uint32_t len);
/* dma_stop_send memory to memory */
void dma_stop_send_m2m(void);

#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
}
#endif

#endif  /* GD_NN_SUPPORT_H */
