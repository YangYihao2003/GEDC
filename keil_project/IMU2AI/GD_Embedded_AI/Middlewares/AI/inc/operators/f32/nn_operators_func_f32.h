/*!
    \file    nn_operators_func_f32.h
    \brief   definitions for the nn_operators_func
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

#ifndef _GD32_NN_WRAPPER_OPERATORS_H_
#define _GD32_NN_WRAPPER_OPERATORS_H_

#include "gd_nn_types.h"
#include "gd_nn_basic_types.h"
#include "gd_nn_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

    
/* conv2d fast */
error_type conv2d_fast_f32(const nn_conv2d_params* params, const nn_dims* input_shape,
                 const float* input_data, const nn_dims* filter_shape,
                 const float* filter_data, const nn_dims* bias_shape,
                 const float* bias_data, const nn_dims* output_shape,
                 float* output_data);
                 
/* conv2d_group_f32 fast */
error_type conv2d_fast_group_f32(const nn_conv2d_params* params, const nn_dims* input_shape,
                 const float* input_data, const nn_dims* filter_shape,
                 const float* filter_data, const nn_dims* bias_shape,
                 const float* bias_data, const nn_dims* output_shape,
                 float* output_data);
                 
/* normal conv2d function */
error_type conv2d_normal_f32(const nn_conv2d_params* params, const nn_dims* input_shape,
                 const float* input_data, const nn_dims* filter_shape,
                 const float* filter_data, const nn_dims* bias_shape,
                 const float* bias_data, const nn_dims* output_shape,
                 float* output_data);


/* activation_leaky_relu_f32 */
 error_type activation_leaky_relu_f32(const nn_leakyrelu_params* params,
                      const nn_dims* input_shape, const float* input_data,
                      const nn_dims* output_shape, float* output_data);
                      
 error_type activation_leaky_relu_int8(const nn_leakyrelu_params* params,
                              nn_dims* input_shape,
                              const int8_t* input_data,
                              const nn_dims* output_shape,
                              const int in_dims_size,
                              int8_t* output_data);

///* activation_sigmoid_f32 */
//error_type activation_sigmoid_f32(const nn_sigmoid_params* params,
//                      const nn_dims* input_shape, const float* input_data,
//                      const nn_dims* output_shape, float* output_data);

/* dwconv_fast_f32 */
error_type dwconv_fast_f32(const nn_dw_conv_params* params, const nn_dims* input_shape,
    const float* input_data, const nn_dims* filter_shape,
    const float* filter_data, const nn_dims* bias_shape,
    const float* bias_data, const nn_dims* output_shape,
    float* output_data);

/* dwconv_normal_f32 */
error_type dwconv_normal_f32(const nn_dw_conv_params *params, const nn_dims *input_shape,
                            const float *input_data, const nn_dims *filter_shape,
                            const float *filter_data, const nn_dims *bias_shape,
                            const float *bias_data, const nn_dims *output_shape,
                            float *output_data);

/* max_pool_f32 */
error_type max_pool_f32(nn_pooling_params* params,const float* input,float* output,
    const nn_dims* input_shape, const nn_dims* output_shape);

/* avg_pool_f32 */
error_type avg_pool_f32(const nn_pooling_params* params,
                        const nn_dims* input_shape,
                        const float* input_data,
                        const nn_dims* output_shape, float* output_data);

/* softmax_f32 */
error_type softmax_f32(const nn_softmax_params* params,
                    const int outer_size_param, const float* input_data,
                    const int depth_param, float* output_data);
                    
error_type softmax_int8(const nn_softmax_params *params,
                      const int outer_size_param, const int8_t *input_data,
                      const int depth_param, int8_t *output_data, float* temp);

/* conv_pad_i8 and f32 */
error_type PadImpl_int8(nn_pad_params* op_params,
                    const nn_dims* input_shape,  const int8_t* input_data,
                    const int32_t* pad_value_ptr, const nn_dims* output_shape, int out_dim_count,
                    int8_t* output_data) ;
                    
error_type PadImpl_float(nn_pad_params* op_params,
                    const nn_dims* input_shape, const float* input_data,
                    const int32_t* pad_value_ptr, const nn_dims* output_shape, int out_dim_count,
                    float* output_data) ;

error_type PadImplv2_int8(nn_padv2_params* op_params,
                    const nn_dims* input_shape, const int8_t* input_data,
                    const int8_t* pad_value_ptr, const nn_dims* output_shape, int out_dim_count,
                    int8_t* output_data) ;
                    
error_type PadImplv2_float(nn_padv2_params* op_params,
                    const nn_dims* input_shape, const float* input_data,
                    const float* pad_value_ptr, const nn_dims* output_shape, int out_dim_count,
                    float* output_data) ;
/* transpose_f32 */
error_type transpose_f32(const nn_transpose_params *params, const nn_uint32 *input_shape,
                        const float *input_data, const nn_uint8 input_dims_count, const nn_uint32 *output_shape,
                        float *output_data);
                        
error_type transpose_int8(const nn_transpose_params *params, const nn_uint32 *input_shape,
                        const int8_t *input_data, const nn_uint8 input_dims_count, const nn_uint32 *output_shape,
                        int8_t *output_data);

/* logistic_f32 */
error_type logistic_f32(const nn_dims* input_shape , const float* input_data,
                     const nn_dims* output_shape, float* output_data);

error_type logistic_int8(const nn_logistic_params* params,
                       const nn_dims *input_shape, const int8_t *input_data,
                       int8_t *output_data);

/* fullyconnected_normal_f32 */
error_type fullyconnected_normal_f32(const nn_fc_params *params, const nn_dims *input_shape,
                                    const float *input_data, const nn_dims *weights_shape,
                                    const float *weights_data, const nn_dims *bias_shape,
                                    const float *bias_data, const nn_dims *output_shape,
                                    float *output_data);

/* fullyconnected_fast_f32 */
error_type fullyconnected_fast_f32(const nn_fc_params* params, const nn_dims* input_shape,
    const float* input_data, const nn_dims* weights_shape,
    const float* weights_data, const nn_dims* bias_shape,
    const float* bias_data, const nn_dims* output_shape,
    float* output_data);

 /* f32_quan_to i8 */   
error_type quantize_f32_int8(const nn_op_quantization_params *params, const nn_dims *input_shape,
                        const float *input_data, const nn_dims *output_shape,
                        int8_t *output_data, int flat_size); 
 /* f32_quan_to i16 */   
error_type quantize_f32_int16(const nn_op_quantization_params *params, const nn_dims *input_shape,
                        const float *input_data, const nn_dims *output_shape,
                        int16_t *output_data, int flat_size);
 /* int8_dequan_to f32 */   
 error_type dequantize_int8_f32(const nn_op_dequantization_params *params, const nn_dims *input_shape,
                        const int8_t *input_data, const nn_dims *output_shape,
                        float *output_data, int flat_size);  
                                                
/* int16_dequan_to f32 */
error_type dequantize_int16_f32(const nn_op_dequantization_params *params, const nn_dims *input_shape,
                        const int16_t *input_data, const nn_dims *output_shape,
                        float *output_data, int flat_size); 
    
/* add operators f32 */                    
error_type add_f32(const nn_add_params *params, const float *input1_data, 
                   const float *input2_data, float *output_data, int32_t flat_size);

error_type add_f32_another_one_dim(const nn_add_params *params, const float *input1_data, 
                   const float input2_data, float *output_data, int32_t flat_size);

/* add broadcast i8 */  
error_type Add4DSlow_i8(
    const nn_add_params* params, const uint32_t* input1_shape,
    const int8_t* input1_data, const uint32_t* input2_shape,
    const int8_t* input2_data, const uint32_t* output_shape,
    int8_t* output_data);
	
error_type Add4DSlow_f32(const nn_add_params* params,
					const uint32_t* input1_shape,
					const float* input1_data, const uint32_t* input2_shape,
					const float* input2_data, const uint32_t* output_shape,
					float* output_data);

/* sub operators i8 for broadcast */
error_type BroadcastQuantSubSlow_i8(const nn_sub_params* params,
                            uint32_t* input1_shape,
                            const int8_t* input1_data,
                            uint32_t* input2_shape,
                            const int8_t* input2_data,
                            uint32_t* output_shape, int8_t* output_data, 
                            int in_dim_count1, int in_dim_count2, int out_dim_count);
                            
/* sub operators i8 for common */
error_type Sub_i8(const nn_sub_params* params,
                const int8_t* input1_data,
                const int8_t* input2_data,
                const int dim_size, int8_t* output_data);

/* sub operators f32 for common */                
error_type Sub_f32(const nn_sub_params* params,
                const float* input1_data,
                const float* input2_data, 
                const int dim_size, float* output_data);
                
                
 /* sub operators f32 for broadcast */               
error_type BroadcastQuantSubSlow_f32(const nn_sub_params* params,
                            uint32_t* input1_shape,
                            const float* input1_data,
                            uint32_t* input2_shape,
                            const float* input2_data,
                            uint32_t* output_shape, float* output_data, 
                            int in_dim_count1, int in_dim_count2, int out_dim_count);
                            
 /* div operators f32 */  
error_type Div_f32(const nn_div_params* params,
                const float* input1_data,
                const float* input2_data,
                const int dim_size, float* output_data);
  
error_type BroadcastDivSlow_f32(const nn_div_params* params,
                            uint32_t* input1_shape,
                            const float* input1_data,
                            uint32_t* input2_shape,
                            const float* input2_data,
                            uint32_t* output_shape, float* output_data, 
                            int in_dim_count1, int in_dim_count2, int out_dim_count) ;
                            
/* concatenation operators int8 */  
error_type concatenation_int8(const nn_concatenation_params* params,
                          int* input_shapes[],
                          nn_int8* input_data[],
                          nn_uint32* output_shape, int out_dims_count,
                          nn_int8* output_data);

/* concatenation operators f32 */  
error_type concatenation_float(const nn_concatenation_params* params,
                          int* input_shapes[],
                          float* input_data[],
                          nn_uint32* output_shape, int out_dims_count,
                          float* output_data);
                          
 /* gather operators int8 and f32 */                         
error_type Gather_int8(const nn_gather_params* params,
                    const nn_uint32* input_dims, int8_t* input_data, int in_dims_count,
                    const nn_dims* coords_dims, int32_t* coords_data, nn_uint32* output_dims, int8_t* output_data);

error_type Gather_float(const nn_gather_params* params,
                    const nn_uint32* input_dims, float* input_data, int in_dims_count,
                    const nn_dims* coords_dims, int32_t* coords_data, nn_uint32* output_dims, float* output_data);

/* Pack operators int8 and f32 */         
error_type Pack_int8( const nn_pack_dims* params,
                    const nn_uint32* input_dims, int in_dims_count, int8_t** inputs_data,
                    const nn_uint32* output_dims, int out_dims_count, int8_t* output_data, nn_uint8 in_buf_num);

error_type Pack_float( const nn_pack_dims* params,
                    const nn_uint32* input_dims, int in_dims_count, float** inputs_data,
                    const nn_uint32* output_dims, int out_dims_count, float* output_data, nn_uint8 in_buf_num);

/* Tanh operators int8 and f32 */                      
error_type Tanh_f32(const nn_dims* input_shape, const float* input_data, int in_dim_size,
                 const nn_dims* output_shape, float* output_data);

error_type Tanh_int8(const nn_tanh_params* params,
                 const nn_dims* input_shape, const int8_t* input_data, int in_dim_size,
                 const nn_dims* output_shape, int8_t* output_data);
                 
/* mean op int8 and int32 */         
error_type Mean_f32_no_keepdims(const float* input_data, const int* input_dims,
                 const int input_num_dims, float* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, uint16_t keep_dims,
                 int* temp_index, int* resolved_axis, float* temp_sum);
                
error_type Mean_f32_with_keepdims(const nn_mean_params* op_params,
                 const int* unextended_input_shape,
                 const float* input_data,
                 const int* unextended_output_shape, float* output_data);
                 
error_type Mean_i8(const int8_t* input_data, const int* input_dims,
                 const int input_num_dims, int8_t* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, uint16_t keep_dims,
                 int* temp_index, int* resolved_axis, int32_t* temp_sum);
   
error_type Mean_i8_with_keepdims(const nn_mean_params* op_params, int32_t multiplier,
                 int32_t shift, const int* unextended_input_shape,
                 const int8_t* input_data, int32_t input_zero_point,
                 const int* unextended_output_shape,
                 int8_t* output_data, int32_t output_zero_point);
                 
error_type QuantizedMeanOrSum(const int8_t* input_data, int32_t input_zero_point,
               float input_scale, const int* input_dims,
               const int input_num_dims, int8_t* output_data,
               int32_t output_zero_point, float output_scale,
               const int* output_dims,
               const int output_num_dims, const int* axis,
               const int num_axis_dimensions, uint16_t keep_dims,
               int* temp_index, int* resolved_axis, int32_t* temp_sum,
               uint16_t compute_sum);
               
  /* ResizeNearestNeighbor and ResizeBilinear op int8 and int32 */              
error_type ResizeNearestNeighbor_i8(
    const nn_resize_nearest_neighbor* op_params,
    const nn_dims* input_shape, const int8_t* input_data,
    const nn_dims* output_shape, int8_t* output_data); 
    
error_type ResizeNearestNeighbor_f32(
    const nn_resize_nearest_neighbor* op_params,
    const nn_dims* input_shape, const float* input_data,
    const nn_dims* output_shape, float* output_data) ;
    
error_type ResizeBilinear_i8(
    const nn_resize_bilinear_params* op_params,
    const nn_dims* input_shape, const int8_t* input_data,
    const int32_t output_height, const int32_t output_width,
    const nn_dims* output_shape, int8_t* output_data);
    
error_type ResizeBilinear_f32(
    const nn_resize_bilinear_params* op_params,
    const nn_dims* input_shape, const float* input_data,
    const int32_t output_height, const int32_t output_width,
    const nn_dims* output_shape, float* output_data);

  /* mul op int8 and int32 */ 
uint8_t Need_broadcast(const nn_uint32* shape0,
                                   const nn_uint32* shape1,
								   const nn_uint8 shape0_dimcount,
								   const nn_uint8 shape1_dimcount,
                                   BroadcastParams* params);
				   
error_type  Mul_broadcast_4dims_i8(
	const nn_mul_params* mul_params,
	const nn_uint32* input1_shape,
    const int8_t* input1_data, const nn_uint32* input2_shape,
    const int8_t* input2_data, const nn_uint32* output_shape, int8_t* output_data);
	
error_type  Mul_broadcast_4dims_f32(
	const nn_mul_params* mul_params,
	const nn_uint32* input1_shape,
    const float* input1_data, const nn_uint32* input2_shape,
    const float* input2_data, const nn_uint32* output_shape, float* output_data);
	
  /* mini op int8 and int32 donot test */ 	
error_type MinimumBroadcastSlow_i8( 	 uint32_t* unextended_input1_shape,
                                 const int8_t* input1_data,
                                 uint32_t* unextended_input2_shape,
                                 const int8_t* input2_data,
                                 uint32_t* unextended_output_shape,
                                 int8_t* output_data, int out_dim_size, int in_dim_count1, int in_dim_count2, int out_dim_count);
                                 
error_type MinimumBroadcastSlow_f32( 	 uint32_t* unextended_input1_shape,
                                 const float* input1_data,
                                 uint32_t* unextended_input2_shape,
                                 const float* input2_data,
                                 uint32_t* unextended_output_shape,
                                 float* output_data, int out_dim_size, int in_dim_count1, int in_dim_count2, int out_dim_count);
 
/* maxi op int8 and int32 donot test */ 
error_type MaximumBroadcastSlow_i8( 	 uint32_t* unextended_input1_shape,
                                 const int8_t* input1_data,
                                 uint32_t* unextended_input2_shape,
                                 const int8_t* input2_data,
                                 uint32_t* unextended_output_shape,
                                 int8_t* output_data, int out_dim_size, int in_dim_count1, int in_dim_count2, int out_dim_count);
                                 
error_type MaximumBroadcastSlow_f32( 	 uint32_t* unextended_input1_shape,
                                 const float* input1_data,
                                 uint32_t* unextended_input2_shape,
                                 const float* input2_data,
                                 uint32_t* unextended_output_shape,
                                 float* output_data, int out_dim_size, int in_dim_count1, int in_dim_count2, int out_dim_count); 
                                 
	
error_type Mul_f32(const nn_mul_params* params,
                const uint32_t* input1_shape, const float* input1_data,
                const uint32_t* input2_shape, const float* input2_data,
                const uint32_t* output_shape, float* output_data, int dim_size);
	
  /* relu op int8 and int32 donot test */ 				
error_type Relu_i8(const nn_relu_params* data, const int8_t* input_data,
                   int8_t* output_data, int dim_size);
error_type Relu_f32(const float* input_data, float* output_data, int dim_size);

  /* DepthToSpace f32 and i8*/ 
error_type DepthToSpace_f32(const nn_depth_to_space_params* op_params,
                         const nn_dims* input_shape,
                         const float* input_data,
                         const nn_dims* output_shape,
                         float* output_data);
                         
error_type DepthToSpace_i8(const nn_depth_to_space_params* op_params,
                         const nn_dims* input_shape,
                         const int8_t* input_data,
                         const nn_dims* output_shape,
                         int8_t* output_data);                        

  /* SpaceToDepth f32 and i8*/ 
error_type SpaceToDepth_f32(const nn_depth_to_space_params* op_params,
                         const nn_dims* input_shape,
                         const float* input_data,
                         const nn_dims* output_shape,
                         float* output_data) ;
                         
error_type SpaceToDepth_i8(const nn_depth_to_space_params* op_params,
                         const nn_dims* input_shape,
                         const int8_t* input_data,
                         const nn_dims* output_shape,
                         int8_t* output_data);
/* Split_f32 f32 and i8*/                         
error_type Split_f32(const float* input_data,
                nn_uint32 *input_shape, const int input_dim_count,
                float* output_data_arr[],
                nn_uint32 *output_shape,    
                int axis_value, const int output_count );
                
error_type Split_i8(const int8_t* input_data,
                nn_uint32 *input_shape, const int input_dim_count,
                int8_t* output_data_arr[],
                nn_uint32 *output_shape,    
                int axis_value, const int output_count );
                
error_type Splitv_f32(const float* input_data,
                nn_uint32 *input_shape, const int input_dim_count,
                nn_tensor* out_tensor,    
                int axis_value, const int output_count ) ;

error_type Splitv_i8(const int8_t* input_data,
                nn_uint32 *input_shape, const int input_dim_count,
                nn_tensor* out_tensor,    
                int axis_value, const int output_count );

/* ReduceMax f32 and i8*/ 

error_type ReduceMaxGeneric_f32(const float* input_data, const int* input_dims,
                          const int input_num_dims, float* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          int8_t keep_dims, int* temp_index, int* resolved_axis,
                          float init_value,
                          float reducer_f32(const float current, const float in));
                          
error_type ReduceMaxGeneric_i8(const int8_t* input_data, const int* input_dims,
                          const int input_num_dims, int8_t* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          int8_t keep_dims, int* temp_index, int* resolved_axis,
                          int8_t init_value,
                          int8_t reducer_i8(const int8_t current, const int8_t in));

#define TfLiteEvalTensor nn_tensor
/* UnidirectionalSequenceLSTM f32 */                           
error_type  EvalFloatLstm(
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* input_to_input_weights,
    const TfLiteEvalTensor* input_to_forget_weights,
    const TfLiteEvalTensor* input_to_cell_weights,
    const TfLiteEvalTensor* input_to_output_weights,
    const TfLiteEvalTensor* recurrent_to_input_weights,
    const TfLiteEvalTensor* recurrent_to_forget_weights,
    const TfLiteEvalTensor* recurrent_to_cell_weights,
    const TfLiteEvalTensor* recurrent_to_output_weights,
    const TfLiteEvalTensor* cell_to_input_weights,
    const TfLiteEvalTensor* cell_to_forget_weights,
    const TfLiteEvalTensor* cell_to_output_weights,
    const TfLiteEvalTensor* input_layer_norm_coefficients,
    const TfLiteEvalTensor* forget_layer_norm_coefficients,
    const TfLiteEvalTensor* cell_layer_norm_coefficients,
    const TfLiteEvalTensor* output_layer_norm_coefficients,
    const TfLiteEvalTensor* aux_input,
    const TfLiteEvalTensor* aux_input_to_input_weights,
    const TfLiteEvalTensor* aux_input_to_forget_weights,
    const TfLiteEvalTensor* aux_input_to_cell_weights,
    const TfLiteEvalTensor* aux_input_to_output_weights,
    const TfLiteEvalTensor* input_gate_bias,
    const TfLiteEvalTensor* forget_gate_bias,
    const TfLiteEvalTensor* cell_gate_bias,
    const TfLiteEvalTensor* output_gate_bias,
    const TfLiteEvalTensor* projection_weights,
    const TfLiteEvalTensor* projection_bias, const nn_UnidirectionalSequenceLSTM_params* _params_,/* initial TfLiteLSTMParams* params */
    int forward_sequence, int time_major, int output_offset,
    float* scratch_buffer, TfLiteEvalTensor* output_state,
    TfLiteEvalTensor* cell_state, TfLiteEvalTensor* output);

/* abs f32 and i8*/   
error_type abs_f32( const nn_abs_params* params,
                     int dim_size, float* input_data,
                     float* output_data);
    
error_type abs_int8( const nn_abs_params* params,
                     int dim_size, int8_t* input_data,
                     int8_t* output_data);
                     

/* Prelu4D f32 and i8*/   
error_type BroadcastPrelu4DSlowFloat(
    const nn_prelu_params* params, uint32_t* input_shape,
    const float* input_data, uint32_t* alpha_shape, const float* alpha_data,
     uint32_t* output_shape, float* output_data, int in_dim_count1, int in_dim_count2, int out_dim_count);
                     
error_type BroadcastPrelu4DSlow(
    const nn_prelu_params* params, uint32_t* input_shape,
    const int8_t* input_data, uint32_t* alpha_shape, const int8_t* alpha_data,
     uint32_t* output_shape, int8_t* output_data, int in_dim_count1, int in_dim_count2, int out_dim_count);

error_type CommonPrelu4DFloat(
    const nn_prelu_params* params, uint32_t* input_shape,
    const float* input_data, uint32_t* alpha_shape, const float* alpha_data,
    uint32_t* output_shape, float* output_data, int in_dim_count1, int in_dim_count2, int out_dim_count);
    
/* elementwise op */
error_type log_f32( int dim_size, float* input_data,
                     float* output_data);

error_type sqrt_f32( int dim_size, float* input_data,
                     float* output_data);

error_type pow_f32( float pow, int dim_size, float* input_data,
                     float* output_data);

error_type sign_f32( int dim_size, float* input_data,
                     float* output_data);

error_type sign_int8( int dim_size, int8_t* input_data,
                     int8_t* output_data);

/* rsqrt f32 and i8*/   
error_type rsqrt_f32( const nn_rsqrt_params* params,
                     int dim_size, float* input_data,
                     float* output_data) ;

error_type rsqrt_int8( const nn_rsqrt_params* params,
                     int dim_size, int8_t* input_data,
                     int8_t* output_data);
                     
/* GatherNd f32 and i8*/                  
error_type GatherNd_float(const nn_gatherNd_params* params,
                    const nn_uint32* input_dims_data, float* input_data, int in_dims_count, int in_dims_size,
                    const nn_uint32* indices_dims_data, int32_t* index_data, int indices_dims_count, 
                        nn_uint32* output_dims, float* output_data);
                    
error_type GatherNd_int8(const nn_gatherNd_params* params,
                    const nn_uint32* input_dims_data, int8_t* input_data, int in_dims_count, int in_dims_size,
                    const nn_uint32* indices_dims_data, int32_t* index_data, int indices_dims_count, 
                        nn_uint32* output_dims, int8_t* output_data);
                    
/* squareddifference f32 and i8*/   
error_type BroadcastBinaryFunction6DSlow(
    const nn_SquaredDifference_params* params, const nn_uint32* input1_shape,
    const int8_t* input1_data, int input1_dim_count, const nn_uint32* input2_shape, int input2_dim_count,
    const int8_t* input2_data, const nn_uint32* output_shape, int8_t* output_data,
    int8_t (*binary_func)(int8_t x, int8_t y, const nn_SquaredDifference_params* params));
    
int8_t SquaredDifference_i8(int8_t x, int8_t y, const nn_SquaredDifference_params* params);
    
void ElementWise_i8(int size, const nn_SquaredDifference_params* params, const int8_t* input1_data,
                 const int8_t* input2_data, int8_t* output_data,
                 int8_t (*binary_func)(int8_t x, int8_t y, const nn_SquaredDifference_params* params));
                 
float SquaredDifference_f32(float input1, float input2);
                 
error_type BinaryFunction_f32(const nn_uint32* input1_shape,
                           const float* input1_data, int input1_dim_size,
                           const nn_uint32* input2_shape,
                           const float* input2_data,
                           const nn_uint32* output_shape, float* output_data,
                           float (*func)(float input1, float input2));
                           
error_type BroadcastBinaryFunction4DSlow_f32(
     nn_uint32* unextended_input1_shape, const float* input1_data, int input1_dim_count,
     nn_uint32* unextended_input2_shape, const float* input2_data, int input2_dim_count,
     nn_uint32* unextended_output_shape, float* output_data, int output_dim_count,
    float (*func)(float input1, float input2));
     
 /* slice f32 and i8*/      
error_type Slice_f32(const nn_slice_params* op_params,
                     uint32_t* input_shape,  float* input_data, int input_dim_count,
                     uint32_t* output_shape, float* output_data);
                     
error_type Slice_f32_fast(const nn_slice_params* op_params,
                     uint32_t* input_shape,  float* input_data, int input_dim_count,
                     uint32_t* output_shape, float* output_data,float* temp_buffer);
                         
error_type Slice_int8(const nn_slice_params* op_params,
                     uint32_t* input_shape,  int8_t* input_data, int input_dim_count,
                     uint32_t* output_shape, int8_t* output_data);   
                     
error_type Slice_int8_fast(const nn_slice_params* op_params,
                     uint32_t* input_shape,  int8_t* input_data, int input_dim_count,
                     uint32_t* output_shape, int8_t* output_data,int8_t* temp_buffer);

/* Transposeconv2d f32 and int8 */  
error_type Transposeconv2d_f32(const nn_conv2d_params *params, const nn_dims *input_shape,
                            const float *input_data, const nn_dims *filter_shape,
                            const float *filter_data, const nn_dims *bias_shape,
                            const float *bias_data, const nn_dims *output_shape,
                            float *output_data); 
                            
error_type Transposeconv2d_f32_fast(const nn_transpose_conv_params *params, const nn_dims *input_shape,
                            const float *input_data, const nn_dims *filter_shape,
                            const float *filter_data, const nn_dims *bias_shape,
                            const float *bias_data, const nn_dims *output_shape,
                            float *output_data);
                            
error_type Transposeconv2d_i8(const nn_conv2d_params *params, const nn_dims *input_shape,
                            const int32_t* output_multiplier,
                            const int32_t* output_shift,
                            const int8_t *input_data, const nn_dims *filter_shape,
                            const int8_t *filter_data, const nn_dims *bias_shape,
                            const int32_t *bias_data, const nn_dims *output_shape,
                            int8_t *output_data, int32_t* scratch_buffer);   
                            
/* relu6 f32 and int8 */
error_type Relu6_i8(const nn_relu6_params* data, const int8_t* input_data,
                   int8_t* output_data, int dim_size);

error_type Relu6_f32(const nn_relu6_params* data, const float* input_data,
                   float* output_data, int dim_size);
 
/* StridedSlice f32 and int8 */
error_type StridedSlice_f32(nn_stridedslice_paras* op_params,
                          uint32_t* unextended_input_shape, int input_dim_count,
                          uint32_t* unextended_output_shape, int output_dim_count, 
                          float* input_data, float* output_data);

error_type StridedSlice_int8(nn_stridedslice_paras* op_params,
                          uint32_t* unextended_input_shape, int input_dim_count,
                          uint32_t* unextended_output_shape, int output_dim_count, 
                          int8_t* input_data, int8_t* output_data);
                          
/* exp f32 */                          
error_type exp_f32( int dim_size, float* input_data,
                     float* output_data);

/*  sum f32 and i8 */
error_type sum_f32(const float* input_data, const int* input_dims,
                          const int input_num_dims, float* output_data,
                          const int* output_dims, const int output_num_dims, const int output_dims_size,
                          const int* axis, const int64_t num_axis_dimensions,
                          uint16_t keep_dims, int* temp_index, int* resolved_axis
                          );
                          
error_type sum_f32_fast( float* input_data, const nn_uint32* input_dims,
                          const int input_num_dims, float* output_data,
                          const nn_uint32* output_dims, const int output_num_dims, const int output_dims_size,
                          const int* axis, const int64_t num_axis_dimensions,
                          uint16_t keep_dims, int* temp_index, int* resolved_axis,float* transpose_data);

error_type QuantizedSum(const int8_t* input_data, int32_t input_zero_point,
                               float input_scale, const int* input_dims,
                               const int input_num_dims, int8_t* output_data,
                               int32_t output_zero_point, int32_t output_multiplier,
                               int output_shift,
                               const int* output_dims,
                               const int output_num_dims, const int* axis,
                               const int num_axis_dimensions, uint16_t keep_dims,
                               int* temp_index, int* resolved_axis, int32_t* temp_sum,
                               uint16_t compute_sum);
                               
error_type QuantizedSum_fast(  int8_t* input_data, int32_t input_zero_point,
                               float input_scale, const uint32_t*input_dims,
                               const int input_num_dims, int8_t* output_data,
                               int32_t output_zero_point, int32_t output_multiplier,
                               int output_shift,
                               const int* output_dims,
                               const int output_num_dims, const int* axis,
                               const int num_axis_dimensions, uint16_t keep_dims,
                               int* temp_index, int* resolved_axis, int32_t* temp_sum,
                               uint16_t compute_sum);


/*  BatchMatMul f32 */
error_type BatchMatMul_f32(const nn_batch_matmul_params* op_params,  nn_uint32* lhs_shape, const float* lhs_data, int lhs_dim_count, 
                         nn_uint32* rhs_shape, const float* rhs_data, int rhs_dim_count, 
                        const nn_dims* output_shape, float* output_data);

/*  unpack f32 */                         
error_type UnpackImpl_f32(const nn_uint32* input_dims, float* input_data, int inputdim_count, \
                        const nn_uint32* output_dims, int output_count, int outdim_count, float** outputs_data, \
                        int axis) ;
                        
error_type UnpackImpl_int8(const nn_uint32* input_dims, int8_t* input_data, int inputdim_count, \
                        const nn_uint32* output_dims, int output_count, int outdim_count, int8_t** outputs_data, \
                        int axis);
/*  LogSoftmax f32 */ 
error_type  LogSoftmax_f32(const nn_log_softmax_params* params,
                       const float* input_data,
                       float* output_data) ;
                       
error_type  LogSoftmax_int8(const nn_log_softmax_params* params,
                       const int8_t* input_data,
                       int8_t* output_data, float* temp) ;

/*  MirrorPad_f32 */                        
error_type MirrorPad_f32(const int32_t* padding_matrix,
               const uint32_t* input_dims, int* output_dims_num_elements,
               int* input_dims_num_elements, const float* input_data,
               float* output_data, const int offset, const int num_dims,
               const int output_size);
               
 /*  MirrorPad_int8 */                
error_type MirrorPad_int8(const int32_t* padding_matrix,
               const uint32_t* input_dims, int* output_dims_num_elements,
               int* input_dims_num_elements, const int8_t* input_data,
               int8_t* output_data, const int offset, const int num_dims,
               const int output_size);

 /*  gelu_f32 int8 */                
error_type gelu_f32(const nn_gelu_params* params, int dim_size, float* input_data,
                     float* output_data);

error_type gelu_int8(const nn_gelu_params* params,  int dim_size, int8_t* input_data,
                     int8_t* output_data);
 /*  sin cos */  
error_type sin_f32( int dim_size, float* input_data,
                     float* output_data);
error_type cos_f32( int dim_size, float* input_data,
                     float* output_data); 
                        
__attribute__((weak,noreturn))
void __aeabi_assert (const char *expr, const char *file, int line);        
              

#ifdef __cplusplus
}
#endif

#endif  /* _GD32_NN_WRAPPER_OPERATORS_H_ */
