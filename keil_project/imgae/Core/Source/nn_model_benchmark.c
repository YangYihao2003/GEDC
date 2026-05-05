/*
	Copyright (c) 2023, GigaDevice Semiconductor Inc.

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

#include "nn_usart.h"
#include "nn_model_benchmark.h"
#include "nn_model_configure.h"
#include <string.h>

/* model invoke ret */
error_type ret_model_invoke_error = ok;
/* because of flash cache to get sram's data, header_array must 32byte align */
static __attribute__ ((aligned(32))) nn_uint8 header_array[START_HEADER_SIZE + 24];
/* define response str */
static __attribute__ ((aligned(32))) uint8_t response_arr[32] = "MCU_READY";
 
/*!
    \brief      nn_benchmark_init function£¬ get benchmark header, malloc in_buffer and out_buffer
    \param[in]  nn_benchmark benchmark
    \param[out] none
    \retval     benchmark state
*/
benchmark_state nn_benchmark_init(nn_benchmark* benchmark){
    
    /* store header from usart */
    benchmark_state ret = benchmark_ok;

    memset(header_array,0,START_HEADER_SIZE);
    /* init usart, dma, rcu */
    usart_configure();
    

    /* wait header from up software*/
    ret = (benchmark_state)usart_start_receive(header_array,8);
    if( header_array[0] == 'N' && header_array[1] == 'P' && header_array[2] == 'Z' &&\
        header_array[3] == 'N'){
        /* get npz number */
        memcpy(&(benchmark->npz_number), &header_array[4], 4);  
        
        /* return python message */
            
        usart_start_send((uint8_t*)"GET_NPZN", 8);  
    }
        
    return ret;
}

/*!
    \brief      nn_benchmark_start function£¬ start benchmark get data from pc and invoke by mcu
    \param[in]  nn_benchmark benchmark
                nn_model model_struct
    \param[out] none
    \retval     benchmark state
*/
benchmark_state nn_benchmark_start(nn_benchmark* benchmark, nn_model* model_struct)     
{
    
    benchmark_state ret = benchmark_ok;
    nn_uint8 tx_tag[4] ={'S','E','N','T'};
    /* invoke all npz number by benchmark->npz_number */
    for(nn_uint32 npz_num = 0; npz_num < benchmark->npz_number; npz_num ++) {
        /* wait header from up software*/
        ret = (benchmark_state)usart_start_receive(header_array,START_HEADER_SIZE);

        if( header_array[0] == 'H' && header_array[1] == 'E' && header_array[2] == 'A' &&\
            header_array[3] == 'D')
        {
            
            /* response to pc */
            ret = (benchmark_state)usart_start_send((uint8_t*)response_arr, 32);
            
            /* malloc in_buffer */
            benchmark->in_payload.byte_size = INPUT_SIZE * INPUT_TYPE_SIZE;                     
            benchmark->in_payload.buffer = (nn_uint8* )input_data;

            /* malloc out_buffer */
            benchmark->out_payload.byte_size =  OUTPUT_SIZE * OUTPUT_TYPE_SIZE;
            /* header buf can be use again */        
            benchmark->out_payload.buffer = (nn_uint8* )header_array; 
            
            memcpy(&benchmark->frame_number, &header_array[4], 4);  
            memcpy(&benchmark->out_payload.tag, tx_tag, 4); 
            
            benchmark->result.latency = 0.0;
            benchmark->result.run_times = 0;
            
            ret = benchmark_ok;

        } else {
            ret = benchmark_paras_error;
            /* response to pc */
            memcpy(response_arr, "MCU_ERROR", 10);
            usart_start_send((uint8_t*)response_arr, 32);
            
            return ret;
        }
        
        nn_uint32 frame_number = benchmark->frame_number, i = 0;

        benchmark->result.run_times = 0;

        for( i = 0; i < frame_number; i++)
        {

            /* 2.1 get input data from usart */
            ret = (benchmark_state)usart_start_receive(benchmark->in_payload.buffer, benchmark->in_payload.byte_size + PAYLOAD_HEADER);

            if( ret != benchmark_ok){
                break;
            } else {
                memcpy((uint8_t*)&(benchmark->in_payload),benchmark->in_payload.buffer,PAYLOAD_HEADER);
            }
            
            /* 2.2 verify checksum */        
            ret = nn_checksum_verify(&benchmark->in_payload);
            
            if( ret != benchmark_ok){
                break;
            }
            /* give input data to mcu invoke */        
            model_struct->user_input = (nn_uint8*)&benchmark->in_payload.buffer[PAYLOAD_HEADER];
            
            /* 2.3 run invoke code */
            ret_model_invoke_error = nn_model_invoke(model_struct);
            
            /* 2.4 generate checksum and idx */
            benchmark->out_payload.check_sum = 1;
            benchmark->out_payload.idx = i;
            
            /* 2.5 payload(tag,idx, bytesize, check_sum)memcpy to payload.buffer's head, send current frame's state, result */
            memcpy((nn_uint8*)benchmark->out_payload.buffer,(nn_uint8*)&(benchmark->out_payload), PAYLOAD_HEADER);   
            /* 2.6 send payload.buffer only */
            ret = (benchmark_state)usart_start_send(benchmark->out_payload.buffer, PAYLOAD_HEADER); 
            /* 2.7 send mcu invoke result */
            ret = (benchmark_state)usart_start_send((nn_uint8*)model_struct->user_output[0], benchmark->out_payload.byte_size);             
            
            if( ret != benchmark_ok){
                break;
            }        
            benchmark->result.run_times++;
            
        }
        benchmark->result.latency = model_struct->report_ptr->total_invoke_time;
        
        /* send 8 bytes to pc including latency: 4 bytes and tims: 4 bytes */
        ret = (benchmark_state)usart_start_send((uint8_t*)&benchmark->result, 8);
    }
    
    return ret;
}

/*!
    \brief      nn_checksum_verify function£¬ checksum check
    \param[in]  nn_in_out_payload payload
    \param[out] none
    \retval     benchmark state
*/
benchmark_state nn_checksum_verify(nn_in_out_payload* payload){
    
    benchmark_state ret;
    nn_uint32 s_check_sum = 0;
    
    if(payload->tag[0] == 'S' && payload->tag[1] == 'E' && \
       payload->tag[2] == 'N' && payload->tag[3] == 'T')
    {
        s_check_sum = nn_checksum_generate(payload);
        if( payload->check_sum == s_check_sum){
            ret = benchmark_ok;
        } else {
            ret = benchmark_checksum_error;
        }    
    } else {
        ret = benchmark_header_error;
    }
    
    return ret;
}

/*!
    \brief      nn_checksum_generate function£¬ checksum generate
    \param[in]  nn_in_out_payload payload
    \param[out] none
    \retval     s_check_sum
*/
uint32_t nn_checksum_generate(nn_in_out_payload* payload){
    nn_uint32 s_check_sum = 0, i;
    nn_uint32 len = payload->byte_size;
    nn_uint8 * in = &payload->buffer[PAYLOAD_HEADER];
    
    for (i = 0; i < len; i++) {
        s_check_sum += in[i];
    }
    return s_check_sum;

}
