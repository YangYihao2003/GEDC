/*!
    \file    gd_nn_types.h
    \brief   definitions for the types
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

#ifndef GD_NN_TYPES_H
#define GD_NN_TYPES_H

#include "arm_nn_types.h"
#include "gd_nn_basic_types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" 
{
#endif

/* constant value */
#define NN_MAX(A, B) ((A) > (B) ? (A) : (B))
#define NN_MIN(A, B) ((A) < (B) ? (A) : (B))

#define PADDING_SIZE        5
#define TRANSPOSE_SIZE      6 
#define END_FLAG            0x98380704

/* get pointer var by offset */
#define GET_ADDRESS(type, ptr, offset)      (type*)(*(type**)((uint32_t)ptr + offset))
#define SET_ADDRESS(type, ptr, offset)      (*(type**)((uint32_t)ptr + offset))
/* get callback func pointer var by offset */
#define GET_CB_ADDRESS(type, ptr, offset)   (type)(*(type*)((uint32_t)ptr + offset))
#define SET_CB_ADDRESS(type, ptr, offset)   (*(type*)((uint32_t)ptr + offset))

/* get common var by offset */
#define GET_VAL(type, ptr, offset)  (type)(*(type*)((uint32_t)ptr + offset))    
#define SET_VAL(type, ptr, offset)  (*(type*)((uint32_t)ptr + offset))  

typedef cmsis_nn_dims nn_dims;

#define OP_SUPPORT_NUM 58

typedef enum {
    ok = 0,
    version_error = 1,
    platform_error = 2,
    signature_error = 3,
    param_error = 4,
    input_error = 5,
    handle_error = 6,
    init_error = 7,
    allocate_error = 8,
    create_error = 9,
    operator_error = 10,
    operator_algorim_error_f32 = 11,                    
    operator_algorim_error_int8 = 12,                   
    read_arr_end_error = 13,
    wei_bias_set_error = 14,                                 
    quan_shift_set_error = 15,
    arm_cmsis_nn_arg_error = 16,
    arm_cmsis_nn_no_impl_error = 17,
    arm_cmsis_nn_failure = 18,
    chip_error =19,
    special_data_address_set_error = 20,
    conv_reorder_version_error = 21,
    element_op_param_error = 22,
    set_integer_lstm_param_error = 23,
} error_type;

/* operators types enum*/
typedef enum {
    op_conv2d           = 0,
    op_depthconv2d      = 1,
    op_softmax          = 2,
    op_maxpool2d        = 3,
    op_avepool2d        = 4,
    op_fullconnected    = 5,
    op_leaky_relu       = 6,
    op_pad              = 7,
    op_transpose        = 8,
    op_logistic         = 9,
    op_reshape          = 10,
    op_quantization     = 11,
    op_dequantization   = 12,
    op_expand_dims      = 13, 
    op_add              = 14,
    op_concatenation    = 15, 
    op_gather           = 16,
    op_pack             = 17,
    op_tanh             = 18,
    op_mean             = 19,
    op_resize_nearest_neighbor = 20,
	op_mul				= 21,
	op_minimum			= 22,
	op_maximum			= 23,
	op_relu				= 24,
    op_resize_bilinear  = 25,
    op_sub              = 26,
    op_div              = 27,
    op_depth_to_space   = 28,
    op_space_to_depth   = 29,
    op_split            = 30,
    op_reduceMax        = 31,
    op_UnidirectionalSequenceLSTM   = 32,
    op_splitv           = 33,
    op_padv2            = 34,
    op_abs              = 35,
    op_prelu            = 36,
    op_log              = 37,
    op_rsqrt            = 38,
    op_sqrt             = 39,
    op_gatherNd         = 40,
    op_SquaredDifference= 41,
	op_BidirectionalSequenceLSTM   = 42,
    op_pow              = 43,
    op_sign             = 44,
    op_slice            = 45,
    op_transposeConv    = 46,
    op_relu6            = 47,
    op_stridedslice     = 48,
    op_exp              = 49,
    op_sum              = 50,
    op_batch_matmul     = 51,
    op_unpack           = 52,
    op_log_softmax      = 53,
    op_mirror_pad       = 54,
    op_gelu             = 55,
    op_sin              = 56,
    op_cos              = 57,
    
} nn_operators_type;

/* nn_quantization struct */           
typedef struct {   
    nn_uint8    format;                                 /*!< quantize format show per tensor or channel */
    nn_uint8    reserved;
    nn_uint16    multiplier_data_size;                  /*!< multiplier data number */
    nn_uint32    multiplier_array_offset;               /*!< multiplier data offset */
    cmsis_nn_per_channel_quant_params quant_params;     /*!< multiplier and shfit pointer */
    nn_uint16    shift_data_size;                       /*!< shift data number */
    nn_uint16    reserved_2;
    nn_uint32    shift_array_offset;                    /*!< shift data offset */
} nn_quantization_paras;

/* nn_quantization type */  
typedef enum {   
    quantization_none = 0,
    quantization_nn_i8_out_i8_wei,
    quantization_nn_f32_out_i8_wei,
   
} nn_quantization_type;

/* nn_activation type enum */           
typedef enum {
    tflite_none = 0,
    tflite_relu,
    tflite_relu_1,
    tflite_relu_6,
    tflite_tanh,
    tflite_sigmoid
} nn_activation_type;

/* float activation parameters structure*/ 
typedef struct{
    float float_activation_min;
    float float_activation_max;
}nn_activation_f32;
    
/* conv parameters structure */
typedef struct { 
    cmsis_nn_conv_params conv_params;
    nn_activation_f32 nn_activation;
} nn_conv2d_params;

/* dep conv parameters structure */
typedef struct { 
    cmsis_nn_dw_conv_params dw_conv_params;
    nn_activation_f32 nn_activation;
} nn_dw_conv_params;

/* transpose conv parameters structure */
typedef struct { 
    cmsis_nn_transpose_conv_params transpose_conv;
    nn_activation_f32 nn_activation;
} nn_transpose_conv_params;

/* pooling parameters structure */
typedef struct { 
    cmsis_nn_pool_params pool_params ;
    nn_int16 filter_height;
    nn_int16 filter_width;
    nn_activation_f32 nn_activation;
} nn_pooling_params;

/* fullconnected parameters structure */
typedef struct{ 
    cmsis_nn_fc_params fc_params;
    nn_activation_f32 nn_activation;
} nn_fc_params;

/* softmax parameters structure */
typedef struct {                                      
    float beta;                     /*!< float need */
    nn_uint16 outer_size_param;           
    nn_uint16 depth_param;                
    nn_int32 diff_min;              /*!< int8 need */  
    
    float input_scale;
    nn_int32 input_offset;
    float output_scale;
    nn_int32 output_offset;
    
} nn_softmax_params;

/* leakyrelu parameters structure */
typedef struct{ 
    float alpha;                    /*!< float need */ 
    nn_int32 output_multi_alpha;    /*!< int8 need */     
    nn_int32 output_shfit_alpha;
    nn_int32 output_multi_identity;
    nn_int32 output_shfit_identity;
    nn_int32 input_zero_point;
    nn_int32 output_zero_point;
} nn_leakyrelu_params;

/* logistic parameters structure */
typedef struct{ 
    nn_int32  input_zero_point;
    nn_int32  range_radius;
    nn_int32  input_multiplier;
    nn_int32  input_left_shift;  
} nn_logistic_params;

/* tanh parameters structure */
typedef struct{ 
    nn_int32  input_zero_point;
    nn_int32  range_radius;
    nn_int32  input_multiplier;
    nn_int32  input_left_shift;  
} nn_tanh_params;

/* transpose parameters structure */
typedef struct  {
    int32_t perm_count;
    int perm[TRANSPOSE_SIZE];
}nn_transpose_params;

/* pad parameters structure */
typedef struct {
    nn_uint8 left_padding_count;
    nn_uint8 right_padding_count;
    nn_uint8 resizing_category;
    nn_uint8 reserve;
    nn_uint32 left_padding[PADDING_SIZE];
    nn_uint32 right_padding[PADDING_SIZE];
    nn_int32 output_zero_point;
}nn_pad_params;

/* padv2 parameters structure */
typedef struct {
    nn_uint8 left_padding_count;
    nn_uint8 right_padding_count;
    nn_uint8 resizing_category;
    nn_int8 constant_values_i8;
    nn_uint32 left_padding[PADDING_SIZE];
    nn_uint32 right_padding[PADDING_SIZE];
    nn_int32 output_zero_point;
    float   constant_values_f32;
}nn_padv2_params;

/* operator quantize parameters structure */
typedef struct{ 
    float scale;
    int32_t zero_point;                     /*!< quantize params */
    int32_t quan_type;                      /*!< 0: f32_to_i8; 1: f32_to_i16 */
    int32_t requantize_output_multiplier;   /*!< requantize params */
    int32_t requantize_output_shift;
    int32_t input_zero_point;
} nn_op_quantization_params;

/* operator dequantize parameters structure */
typedef struct{ 
    float scale;
    int32_t zero_point;
    int32_t quan_type;                  /*!< 0: f32_to_i8; 1: f32_to_i16 */
} nn_op_dequantization_params;

/* operator expand parameters structure */
typedef struct{
    int32_t no_param;
}nn_expand_dims;

/* operator pack parameters structure */
typedef struct{
    uint16_t axis;
    uint16_t values_count;
    uint16_t packself_sign;
    uint16_t reserve;
}nn_pack_dims;

/* add parameters structure */
typedef struct{
    nn_int32 input_1_offset;                    /*!< arm_elementwise_add_s8 need */
    nn_int32 input_1_mult;
    nn_int32 input_1_shift;
    nn_int32 input_2_offset;
    nn_int32 input_2_mult;
    nn_int32 input_2_shift;
    nn_int32 out_offset;
    nn_int32 out_mult;
    nn_int32 out_shift;
    
    nn_int32 left_shift;
    
    cmsis_nn_activation nn_activation_int8;
    nn_activation_f32   nn_activation_float;    /*!< float need */
}nn_add_params;

/* add parameters structure */
typedef struct{
    nn_int32 input_1_offset;                    /*!< arm_elementwise_add_s8 need */
    nn_int32 input_1_mult;
    nn_int32 input_1_shift;
    nn_int32 input_2_offset;
    nn_int32 input_2_mult;
    nn_int32 input_2_shift;
    nn_int32 out_offset;
    nn_int32 out_mult;
    nn_int32 out_shift;
    
    nn_int32 left_shift;
    
    cmsis_nn_activation nn_activation_int8;
    nn_activation_f32   nn_activation_float;    /*!< float need */
    nn_int32 constant_value;
    
}nn_sub_params;

/* div parameters structure */
typedef struct{
   
    /* now version div only support two float data streams div */
   nn_activation_f32   nn_activation_float;    /*!< float need */ 
   
   nn_int32 constant_value;
    
}nn_div_params;

/* concatenation parameters structure */
typedef struct{
    nn_int16    axis;
    nn_uint16   input_count;
    nn_int32    input_zero_point;
    nn_float    in_scale;
    nn_int32    output_zero_point;
    nn_float    out_scale;
}nn_concatenation_params;

/* gather parameters structure */
typedef struct{
    nn_int16    coords_type;                /*!< 0: int32_t, 1: int64_t*/
    nn_int16    batch_dims;
    nn_int32    axis;    
}nn_gather_params;

/* mean parameters structure */
typedef struct{
    nn_uint16   axis_count;                
    nn_uint16   keep_dims;
    nn_int16    axis[4];
    nn_int32    input_zero_point;           /*!< int8 need */     
    nn_float    in_scale;
    nn_int32    output_zero_point;
    nn_float    out_scale;  
    nn_int32    multiplier;
    nn_int32    shfit;
}nn_mean_params;

/* resize_nearest parameters structure */
typedef struct{
    nn_uint16   align_corners;                
    nn_uint16   half_pixel_centers;
    nn_int32    output_size_height;
    nn_int32    output_size_width;          /*!< tflite only support resize width and height */ 
}nn_resize_nearest_neighbor;

/* resize_bilinear parameters structure */
typedef struct{
  /* int8_t inference params */
  nn_uint16   align_corners;
  nn_uint16   half_pixel_centers;
  nn_int32    output_size_height;
  nn_int32    output_size_width;          /*!< tflite only support resize width and height */ 
}nn_resize_bilinear_params;

/* mul broadcast structure */
typedef struct{
	nn_uint32 broadcast_category;
	int broadcast_shape[5];
}BroadcastParams;

/* mul parameters structure */
typedef struct{
    /* int8_t inference params */
    int32_t input1_offset;
    int32_t input2_offset;

    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;

    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    // float activation params.
    float float_activation_min;
    float float_activation_max;
    int32_t constant_value_int;
    float constant_value_f32;
}nn_mul_params;

/* minimum and maximum parameters structure */
typedef struct{
  /* int8_t inference params */
  int32_t constant_int;
  float constant_float;
}nn_mini_maxi_params;

/* relu parameters structure */
typedef struct{
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
}nn_relu_params;

/* depth_to_space parameters structure */
typedef struct { 
    int32_t block_size;
} nn_depth_to_space_params;

/* split parameters structure */
typedef struct { 
    int16_t num_splits;
    int16_t axis;
} nn_split_params;

/* splitv parameters structure */
typedef struct { 
    int32_t axis;
} nn_splitv_params;

/* reduceMax parameters structure */
typedef struct { 
    int16_t axis[4];
    int16_t keep_dims;
    int16_t num_axis;
} nn_reduceMax_params;

/* UnidirectionalSequenceLSTM float parameters structure */
typedef struct { 
    int32_t TfliteFuseActivation;
    float cell_clip;
    float proj_clip;
    int16_t kernal_type;
    int16_t asymmetric_quantize_inputs;
    int16_t time_major;
    int16_t diagonal_recurrent_tensors;
    /* every lstm layer buf need a special buf for memory */
    uint32_t buffer_start_index;        
    
} nn_UnidirectionalSequenceLSTM_params;

/* UnidirectionalSequenceLSTM int parameters structure need */
typedef struct {
    int time_major;
    int batch_size;
    int time_steps;
    int input_dimension;
    int state_dimension;
} LstmSizeInfo;

typedef struct {
    float cell_clip;
    int16_t quantized_cell_clip;
    int16_t cell_state_scale_power;
    
}CellStateInfo;

typedef struct {
    int input_offset;
    int weights_offset;
    int output_offset;
    int output_multiplier;
    int output_shift;
    int quantized_activation_min;
    int quantized_activation_max;
    int reserved;
} LSTM_FullyConnectedParams;

typedef struct {
    LSTM_FullyConnectedParams input_fc_params;
    LSTM_FullyConnectedParams recurrent_fc_params;
} GateParameters;

typedef struct {
    int input1_offset;
    int input2_offset;
    int output_offset;
    int output_multiplier;
    int output_shift;
    int quantized_activation_min;
    int quantized_activation_max;
} ArithmeticParams;

typedef struct {
    ArithmeticParams forget_cell_mul_params;
    ArithmeticParams input_mul_params;
    ArithmeticParams output_mul_params;
} InterGateParameters;

typedef struct {
    LstmSizeInfo size_info;
    CellStateInfo cell_state_info;
    int cell_gate_nonlinear_type;
    GateParameters forget_gate_parameters;
    GateParameters input_gate_parameters;
    GateParameters cell_gate_parameters;
    GateParameters output_gate_parameters;
    InterGateParameters inter_gate_parameters;
    int buffer_indices[4];  // TFLM only
} OpDataLSTM;

/* for cmsis UnidirectionalSequenceLSTM */
typedef struct {
  OpDataLSTM params_ref;                 // Used for fallback implementation
  cmsis_nn_lstm_params params_cmsis_nn;  // Used for  CMSIS-NN implementation
} OpData;

/* abs parameters structure */
typedef struct{
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
  int32_t needs_rescale;
}nn_abs_params;

/* prelu parameters structure */
typedef struct {
    int input_offset;
    int alpha_offset;
    int output_offset;
    int output_multiplier1;
    int output_shift1;
    int output_multiplier2;
    int output_shift2;
}nn_prelu_params;

/* rsqrt parameters structure */
typedef struct{
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
  int32_t needs_rescale;
}nn_rsqrt_params;

/* log parameters structure */
typedef struct {
    int reserved;
}nn_log_params;

/* sqrt parameters structure */
typedef struct {
    int reserved;
}nn_sqrt_params;

/* gatherNd parameters structure */
typedef struct {
    int reserved;
}nn_gatherNd_params;

/* SquaredDifference parameters structure */
typedef struct{
    nn_int32 input_1_offset;                    
    nn_int32 input_1_mult;
    nn_int32 input_1_shift;
    nn_int32 input_2_offset;
    nn_int32 input_2_mult;
    nn_int32 input_2_shift;
    nn_int32 out_offset;
    nn_int32 out_mult;
    nn_int32 out_shift;
    
    nn_int32 left_shift;
    
    cmsis_nn_activation nn_activation_int8;
    nn_activation_f32   nn_activation_float;    /*!< float need */
}nn_SquaredDifference_params;


/* pow parameters structure */
typedef struct {
    int pow_int;
    float pow_f32;
}nn_pow_params;

/* pow parameters structure */
typedef struct{
    int reserved;
}nn_sign_params;

/* slice parameters structure */
typedef struct {
    nn_int32 begin[5];
    nn_int32 size[5];
    nn_int16 begin_count;
    nn_int16 size_count;
}nn_slice_params;

/* relu6 parameters structure */
typedef struct {
    
    float input_scale;
    nn_int32 input_offset;
    float output_scale;
    nn_int32 output_offset;
    
    nn_int16 data_six;
    nn_int16 zero;
}nn_relu6_params;

/* strideslice parameters struct */           
typedef struct {
    nn_int32 start_indices_count;  
    nn_int32 start_indices[5];
    nn_int32 stop_indices_count;  
    nn_int32 stop_indices[5];
    nn_int32 stride_count;  
    nn_int32 stride[5];
    
    nn_int16 begin_mask;
    nn_int16 ellipsis_mask;
    nn_int16 end_mask;
    nn_int16 new_axis_mask;
    nn_int16 offset;
    nn_int16 shrink_axis_mask;
    
} nn_stridedslice_paras;

/* exp parameters structure */
typedef struct{
    int reserved;
}nn_exp_params;

/* batch_matmul parameters structure */
typedef struct{
    cmsis_nn_fc_params fc_params;
    float float_activation_min;
    float float_activation_max;
    
    nn_int32 out_mult;
    nn_int32 out_shift;
    
    nn_int16 adj_x;
    nn_int16 adj_y;
    nn_int16 lhs_is_constant_tensor;
    nn_int16 rhs_is_constant_tensor;

}nn_batch_matmul_params;

/* unpack parameters structure */
typedef struct{
    nn_int32 axis;
    nn_int32 num;
}op_unpack_params;

/* log_softmax parameters structure */
typedef struct {
//    nn_int32 input_multiplier;          
//    nn_int32 input_left_shift;
//    nn_int32 reverse_scaling_divisor;
//    nn_int32 reverse_scaling_right_shift;
    
    float input_scale;
    nn_int32 input_offset;
    float output_scale;
    nn_int32 output_offset;
    
    nn_int32 diff_min;
    
    nn_int32 outer_size;
    nn_int32 depth;
    
}nn_log_softmax_params;

/* mirror pad parameters structure */
typedef struct{
    nn_int32 padding_matrix[10];
    nn_int32 MirrorPadMode;     /* REFLECT is 1 */  
        
}op_mirror_pad_params;

/* gelu parameters structure */
typedef struct {                                      
           
    nn_int32 approximate;              
    
    float input_scale;
    nn_int32 input_offset;
    float output_scale;
    nn_int32 output_offset;
    
} nn_gelu_params;

/* sincos parameters structure */
typedef struct{
    int reserved;
}nn_sincos_params;

/* nn_malloc for layer struct */
void* nn_malloc(uint32_t size);
/* nn_quantization_struct_create */
nn_quantization_paras* nn_quantization_paras_struct_create(void);
/* nn_quantization_struct_init */
void nn_quantization_paras_struct_init(nn_quantization_paras* quantization, nn_uint8* paras);
/* nn_quantization_struct_destroy */
void nn_quantization_paras_struct_destroy(nn_quantization_paras* quantization);
/* nn_special_params_struct_create and init */
error_type nn_special_params_struct_create_init(nn_operators_type op_types, nn_handle* sp_ptr, nn_uint8* paras);
/* nn_special_params_struct_destroy*/
void nn_special_params_struct_destroy(nn_handle sp_ptr);
/* conv2d_init*/
static error_type nn_conv2d_params_struct_init(nn_conv2d_params* convp, nn_uint8* paras);
/* depthconv2d_init*/
static error_type nn_dw_conv_params_struct_init(nn_dw_conv_params* depthconvp, nn_uint8* paras);
/* nn_pooling_params_struct_init*/
static error_type nn_pooling_params_struct_init(nn_pooling_params* pooling, nn_uint8* paras);
/* softmax_init*/
static error_type nn_softmax_params_struct_init(nn_softmax_params* nn_softmax, nn_uint8* paras);
/* fullconnected_init*/
static error_type nn_fc_params_struct_init(nn_fc_params* fc_paras, nn_uint8* paras);
/* leaky_relu_init*/
static error_type nn_leakyrelu_params_struct_init(nn_leakyrelu_params* leakyrelup, nn_uint8* paras);
/* pad_init*/
static error_type nn_pad_params_struct_init(nn_pad_params* padp, nn_uint8* paras);
/* padv2_init*/
static error_type nn_padv2_params_struct_init(nn_padv2_params* padp, nn_uint8* paras);
/* transpose_init*/
static error_type nn_transpose_struct_init(nn_transpose_params* transposep, nn_uint8* paras);
/* quantization op init*/
static error_type nn_quantization_struct_init(nn_op_quantization_params* quantizationp, nn_uint8* paras);
/* dequantization op init*/
static error_type nn_dequantization_struct_init(nn_op_dequantization_params* dequantizationp, nn_uint8* paras);
/* expand_dims op init*/
static error_type nn_expand_dims_struct_init(nn_expand_dims* expand_dimsp, nn_uint8* paras);
/* add op init*/
static error_type nn_add_struct_init(nn_add_params* addp, nn_uint8* paras);
/* sub op init*/
static error_type nn_sub_struct_init(nn_sub_params* subp, nn_uint8* paras);
/* div op init*/
static error_type nn_div_struct_init(nn_div_params* divp, nn_uint8* paras);
/* concatenation op init*/
static error_type nn_concatenation_struct_init(nn_concatenation_params* concatp, nn_uint8* paras);
/* gather op init*/
static error_type nn_gather_struct_init(nn_gather_params* gatherp, nn_uint8* paras);
/* logistic op init*/
static error_type nn_logistic_struct_init(nn_logistic_params* logisticp, nn_uint8* paras);
/* pack op init*/
static error_type nn_pack_struct_init(nn_pack_dims* packp, nn_uint8* paras);
/* tanh op init*/
static error_type nn_tanh_struct_init(nn_tanh_params* tanhp, nn_uint8* paras);
/* mean op init*/
static error_type nn_mean_struct_init(nn_mean_params* meanp, nn_uint8* paras);
/* resize_nearest_neighbor op init*/
static error_type nn_resize_nearest_neighbor_struct_init(nn_resize_nearest_neighbor* resizep, nn_uint8* paras);
/* resize_bilinear op init*/
static error_type nn_resize_bilinear_struct_init(nn_resize_bilinear_params* resizep, nn_uint8* paras);
/* mul op init*/
static error_type nn_mul_struct_init(nn_mul_params* mulp, nn_uint8* paras);
/* mini_maxi op init*/
static error_type nn_mini_maxi_struct_init(nn_mini_maxi_params* mini_maxi_p, nn_uint8* paras);
/* relu op init*/
static error_type nn_relu_struct_init(nn_relu_params* relu_p, nn_uint8* paras);
/* depth_to_space init*/
static error_type nn_depth_to_space_struct_init(nn_depth_to_space_params* depth_to_space_p, nn_uint8* paras);
/* split init*/
static error_type nn_split_struct_init(nn_split_params* split_p, nn_uint8* paras);
/* splitv init*/
static error_type nn_splitv_struct_init(nn_splitv_params* split_p, nn_uint8* paras);
/* reduceMax init*/
static error_type nn_reduceMax_struct_init(nn_reduceMax_params* reduceMax_p, nn_uint8* paras);
/* UnidirectionalSequenceLSTM init*/
static error_type nn_UnidirectionalSequenceLSTM_struct_init(nn_UnidirectionalSequenceLSTM_params* UnidirectionalSequenceLSTM_p, nn_uint8* paras);
/* abs init*/
static error_type nn_abs_struct_init(nn_abs_params* abs_p, nn_uint8* paras);
/* prelu init*/
static error_type nn_prelu_struct_init(nn_prelu_params* prelu_p, nn_uint8* paras);
/* log init*/
static error_type nn_log_struct_init(nn_log_params* log_p, nn_uint8* paras);
/* rsqrt init*/
static error_type nn_rsqrt_struct_init(nn_rsqrt_params* rsqrt_p, nn_uint8* paras);
/* sqrt init*/
static error_type nn_sqrt_struct_init(nn_sqrt_params* sqrt_p, nn_uint8* paras);
/* gatherNd init*/
static error_type nn_gatherNd_struct_init(nn_gatherNd_params* gdtherNd_p, nn_uint8* paras);
/* nn_SquaredDifference_params init*/
static error_type nn_SquaredDifference_struct_init(nn_SquaredDifference_params* SquaredDifferencep, nn_uint8* paras);
/* pow struct init*/
static error_type nn_pow_struct_init(nn_pow_params* pow_p, nn_uint8* paras);
/* sign struct init*/
static error_type nn_sign_struct_init(nn_sign_params* sign_p, nn_uint8* paras);
/* slice struct init*/
static error_type nn_slice_struct_init(nn_slice_params* slicep, nn_uint8* paras);
/* relu6 struct init*/
static error_type nn_relu6_params_struct_init(nn_relu6_params* relu6p, nn_uint8* paras);
/* stridedslice struct init*/
static error_type nn_stridedslice_params_struct_init(nn_stridedslice_paras* stridedslicep, nn_uint8* paras);
/* exp struct init*/
static error_type nn_exp_struct_init(nn_exp_params* exp_p, nn_uint8* paras);
/* transpose_conv2d struct init*/
static error_type nn_transpose_conv2d_params_struct_init(nn_transpose_conv_params* convp, nn_uint8* paras);
/* batch_matmul struct init*/
static error_type nn_batch_matmul_struct_init(nn_batch_matmul_params* batch_matmul_p, nn_uint8* paras);
/* unpack struct init*/
static error_type nn_unpack_struct_init(op_unpack_params* unpack_p, nn_uint8* paras);
/* log_softmax struct init*/
static error_type nn_log_softmax_struct_init(nn_log_softmax_params* log_softmax_p, nn_uint8* paras);
/* mirror_pad struct init*/
static error_type nn_mirror_pad_struct_init(op_mirror_pad_params* mirror_pad_p, nn_uint8* paras);
/* gelu struct init*/
static error_type nn_gelu_struct_init(nn_gelu_params* gelu_p, nn_uint8* paras);
/* sin_cos struct init*/
static error_type nn_sincos_struct_init(nn_sincos_params* sin_cos_p, nn_uint8* paras);

#ifdef __cplusplus
}
#endif

#endif /* GD_NN_TYPES_H */
