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

#ifndef __GD32_NN_BENCHMARK_H
#define __GD32_NN_BENCHMARK_H

#include "gd_nn_types.h"
#include "gd_nn_interface.h"  

#define START_HEADER_SIZE       8  
#define PAYLOAD_HEADER          16 

typedef struct {
    nn_uint8 tag[4];
    nn_uint32 idx;
    nn_uint32 byte_size;
    nn_uint32 check_sum;
    nn_uint8* buffer;   /* tag + in_byte_size */
} nn_in_out_payload;

typedef struct {
    float latency;
    nn_uint32 run_times;
} nn_benchmark_result;

typedef struct {
    nn_in_out_payload in_payload;
    nn_in_out_payload out_payload;
    nn_uint32 frame_number;
    nn_uint32 npz_number;
    nn_benchmark_result result;
}nn_benchmark;

typedef enum  {
     benchmark_ok = 1,
     benchmark_time_out = 2,
     benchmark_paras_error = 3,
     benchmark_malloc_error = 4,
     benchmark_checksum_error = 5,
     benchmark_header_error = 6
}benchmark_state;

/* checksum check */
benchmark_state nn_checksum_verify(nn_in_out_payload* payload);
/* checksum generate */
uint32_t nn_checksum_generate(nn_in_out_payload* payload);
/* get benchmark header, malloc in_buffer and out_buffer */
benchmark_state nn_benchmark_init(nn_benchmark* benchmark);
/* start receive data, and invoke the model */
benchmark_state nn_benchmark_start(nn_benchmark* benchmark, nn_model* model_struct);
/* free buffer */
void nn_benchmark_finish(nn_benchmark* benchmark);

#endif
