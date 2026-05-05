/*!
    \file    gd_nn_report.h
    \brief   report for the invoke
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

#ifndef _GD_NN_REPORT_H_
#define _GD_NN_REPORT_H_

#define GD_NN_MAX_STRING_NUMBER    48
#define GD_NN_OPER_NUMBER          20

#include <stdio.h>
#include <stdlib.h>
#include "systick.h"

/* constant var */
#define SysTick_CLKSource_HCLK                                   0x00000004
/* extern dec */
extern uint32_t sys_count;
extern uint32_t SystemCoreClock;

/* define layer report struct */ 
typedef struct {
    float  layer_invoke_time;                                   /*!< current layer run time */
} nn_layer_report_struct;

/* define model report struct */ 
typedef struct {
    uint32_t core_clock;
    char invoke_frame_version[4];                               /*!< invoke_frame_version */
    char* model_name;                                           /*!< model name */
    uint16_t model_quantization_type;                           /*!< model quantization type */
    uint16_t layer_number;                                      /*!< layer number */
    char** operator_name;                                       /*!< operator_name */
    float total_invoke_time;                                    /*!< total invoke time */
    nn_layer_report_struct* header;
} nn_model_report_struct;

/* printf model inference's report */
void gd_nn_report_printf(nn_model_report_struct* report);
/* start systick timing */
void  gd_nn_measure_time_start(void);
/* get systick timing */
float gd_nn_measure_time_get(float scale ,uint32_t clock);
/* stop systick timing */
void gd_nn_measure_time_stop(void);

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif
#endif  /* _GD_NN_REPORT_H_ */
