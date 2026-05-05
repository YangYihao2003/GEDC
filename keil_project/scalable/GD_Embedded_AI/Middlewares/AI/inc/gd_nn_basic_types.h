/*!
    \file    gd_nn_basic_types.h
    \brief   definitions for the basic_types
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

#ifndef GD_NN_BASIC_TYPES_H
#define GD_NN_BASIC_TYPES_H
#pragma once
/*****************************************************************************/
#ifdef __cplusplus
extern "C" 
{
#endif

#include <stdint.h>

/* constant value */
#define NN_LAYER_STRUCT_SIZE                12
#define NN_LAYER_STRUCT_ALL_SIZE            40   
#define NN_QUANTIZATION_STRUCT_SIZE         16
#define NN_TENSOR_STRUCT_SIZE               28
#define NN_RESET  0
#define NN_SET  1
    
#define     NOT_USE_RETURN

/* data type limit */
#define INT32_SIZE_MIN                      -2147483648
#define INT32_SIZE_MAX                      2147483647
#define INT16_SIZE_MIN                      -32768
#define INT16_SIZE_MAX                      32767
#define INT8_SIZE_MIN                       -128
#define INT8_SIZE_MAX                       127
#define UINT8_SIZE_MIN                      0
#define UINT8_SIZE_MAX                      255

typedef void*               nn_handle;
typedef void*               reserved_ptr;
typedef const void*         nn_handle_const;
    
typedef uint32_t            nn_size;
typedef uint16_t            nn_short_size;
typedef char                nn_char;
    
typedef int                 nn_int;
typedef int8_t              nn_int8;
typedef int16_t             nn_int16;
typedef int32_t             nn_int32;
typedef int64_t             nn_int64;

typedef unsigned int        nn_uint;
typedef uint8_t             nn_uint8;
typedef uint16_t            nn_uint16;
typedef uint32_t            nn_uint32;
typedef uint64_t            nn_uint64;

typedef float               nn_float;
typedef double              nn_double;

/* read array func with different type */
nn_uint8 read_nn_uint8(nn_uint8* paras);
nn_uint16 read_nn_uint16(nn_uint8* paras);
nn_uint32 read_nn_uint32(nn_uint8* paras);
nn_int8 read_nn_int8(nn_uint8* paras);
nn_int16 read_nn_int16(nn_uint8* paras);
nn_int32 read_nn_int32(nn_uint8* paras);
nn_float read_nn_f32(nn_uint8* paras);

#if defined(NOT_USE_RETURN)
void read_nn_d64(nn_uint8* paras, nn_double* out_data);
#else
nn_double read_nn_d64(nn_uint8* paras);
#endif

#ifdef __cplusplus
}
#endif

#endif  /* GD_NN_BASIC_TYPES_H */
