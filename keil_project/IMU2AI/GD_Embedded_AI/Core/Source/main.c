#include "gd32h7xx.h"
#include "gd32h7xx_it.h"
#include "systick.h"
#include "FreeRTOS.h"
#include "task.h"
#include "stdio.h"
#include "gd32h759i_start.h"
#include "semphr.h"
#include "stdio.h"
#include "gd_nn_interface.h"
#include "nn_model_configure.h"
#include "gd_nn_interface.h"
#include "nn_model_configure.h"
#include "string.h"
// 在文件开头添加头文件
#include <stdarg.h>
#include "ringbuffer.h"
/*****************************个人宏定义**********************************/
#define DEBUG 1
#define USART0_RDATA_ADDRESS      ((uint32_t)&USART_RDATA(USART0))
#define USART0_DATA_ADDRESS       ((uint32_t)&USART_TDATA(USART0))
#define ARRAYNUM(arr_nanme)       (uint32_t)(sizeof(arr_nanme) / sizeof(*(arr_nanme)))

/******************************模型输入输出*********************************/
/* input and output code */
float* input_data_transform = (float*)static_buffer_peak;  
float* output_data_ptr_transform[1];  

float* input_data_mlp = (float*)static_buffer_peak;  
float* output_data_ptr_mlp[1];  
/*************************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
/********************************模型内部参数变量**********************************/
/* model parameters struct */
static nn_model imu_error_scalable; 
static nn_model imu_fusion_mlp;
/* gd model_paras_array addr and model_paras_data addr */
const nn_uint8* model_paras_array_and_data_transform[2] = {model_paras_arr_transform, model_paras_data_transform};
const nn_uint8* model_paras_array_and_data_mlp[2] = {model_paras_arr_mlp, model_paras_data_mlp};

/* model reporter instance, if need invoke two models, please create two nn_report*/
nn_model_report_struct nn_report_transform;
nn_model_report_struct nn_report_mlp;
/* gd model_paras_array_dict arr, if need invoke two models, please create two buf*/
nn_uint8 model_paras_array_info_buf_transform[8];
nn_uint8 model_paras_array_info_buf_mlp[8];
/*******************************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
/********************************模型初始化宏定义********************************/
#define nn_model_struct_init_transform(m_struct)\
    m_struct.user_input = input_data_transform;\
    m_struct.user_input_size = INPUT_SIZE_transform * INPUT_TYPE_SIZE_transform;\
    m_struct.user_output = (void**)output_data_ptr_transform;\
    m_struct.user_output_size = OUTPUT_SIZE_transform * OUTPUT_TYPE_SIZE_transform;\
    \
    m_struct.operators_cb_array = func_cb_arr_transform;\
    m_struct.model_paras_array = (const nn_uint8*)model_paras_array_and_data_transform;\
    m_struct.model_paras_array_dict = model_paras_array_info_buf_transform;\
    m_struct.report_ptr = &nn_report_transform; 

#define nn_model_struct_init_mlp(m_struct)\
    m_struct.user_input = input_data_mlp;\
    m_struct.user_input_size = INPUT_SIZE_mlp * INPUT_TYPE_SIZE_mlp;\
    m_struct.user_output = (void**)output_data_ptr_mlp;\
    m_struct.user_output_size = OUTPUT_SIZE_mlp * OUTPUT_TYPE_SIZE_mlp;\
    \
    m_struct.operators_cb_array = func_cb_arr_mlp;\
    m_struct.model_paras_array = (const nn_uint8*)model_paras_array_and_data_mlp;\
    m_struct.model_paras_array_dict = model_paras_array_info_buf_mlp;\
    m_struct.report_ptr = &nn_report_mlp; 
/* nn_report init macro */
#define nn_report_init(nn_report, clock, m_name, opera_name) \
    nn_report.core_clock = clock;\
    nn_report.model_name = m_name;\
    nn_report.operator_name = (char**)opera_name;\
/*******************************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
/********************************freertos变量声明********************************/
/*任务句柄*/
TaskHandle_t start_task_handle ; // 启动任务句柄
TaskHandle_t led_flash_task_handle ; // led任务句柄
TaskHandle_t imu_error_scalable_task_handle; //usartrecv任务句柄
TaskHandle_t imu_fusion_mlp_task_handle; //usartrecv任务句柄
TaskHandle_t fusion_filldata_task_handle; //数据填充任务句柄
TaskHandle_t scalable_filldata_task_handle;
TaskHandle_t fusion_process_output_task_handle; //结果处理任务句柄
TaskHandle_t scalable_process_output_task_handle;
TaskHandle_t send_messgae_task_handle;
TaskHandle_t rev_messgae_task_handle;
SemaphoreHandle_t model_mutex;    // 保护模型推理临界区（共享缓冲区和硬件资源）

SemaphoreHandle_t rx_mutex;  // 互斥锁
QueueHandle_t fusion_input_data_queue;  // 队列元素类型为 float[INPUT_SIZE_mlp]
QueueHandle_t fusion_output_data_queue;  // 队列元素类型为 float[OUTPUT_SIZE_mlp]
QueueHandle_t scalable_input_data_queue;  // 队列元素类型为 float[INPUT_SIZE_trans]
QueueHandle_t scalable_output_data_queue;  // 队列元素类型为 float[OUTPUT_SIZE_trans]
/********************************************************************************/
/*
---------------------------------------gap----------------------------------------
*/
/********************************freertos任务函数声明********************************/
/*任务函数声明*/
void start_task(void *pvParameters);
void led_flash_Task(void *pvParameters);
void imu_error_scalable_Task(void *pvParameters);
void imu_fusion_mlp_Task(void *pvParameters);
void fusion_filldata_Task(void *pvParameters);
void scalable_filldata_Task(void *pvParameters);
void fusion_process_output_Task(void *pvParameters);
void scalable_process_output_Task(void *pvParameters);
void send_messgae_Task(void *pvParameters);
void rev_messgae_Task(void *pvParameters);
/********************************************************************************/
/*
---------------------------------------gap----------------------------------------
*/
/********************************辅助变量定义********************************/
/*自定义变量*/
int trans_fps = 0;
int mlp_fps = 0;

RingBuffer tx_buffer;
__IO FlagStatus g_transfer_complete = RESET;
__attribute__((aligned(32))) char send_buffer[BUFFER_SIZE];
__attribute__((aligned(32))) char rxbuffer[BUFFER_SIZE];
const float fusion_data[12] = {-0.383158, -0.266366, 9.810000, 0.001714, 0.000084, 0.947500,
-0.440697, -0.241141, 9.810000, -0.030049, -0.001457, 1.047825};
const float trans_data[480] = {
    -0.383158f, -0.266366f, 9.810000f, 0.001714f, 0.000084f, 0.947500f,
    -0.440697f, -0.241141f, 9.810000f, -0.030049f, -0.001457f, 1.047825f,
    -0.495450f, -0.209828f, 9.810000f, 0.016379f, 0.027122f, 1.141525f,
    -0.547071f, -0.173217f, 9.810000f, -0.028735f, -0.006802f, 1.228008f,
    -0.595233f, -0.132232f, 9.810000f, 0.008909f, -0.018373f, 1.306726f,
    -0.639632f, -0.087908f, 9.810000f, 0.034795f, 0.038231f, 1.377184f,
    -0.679987f, -0.041365f, 9.810000f, -0.005013f, 0.037239f, 1.438934f,
    -0.716043f, 0.006223f, 9.810000f, -0.000615f, 0.017710f, 1.491587f,
    -0.747571f, 0.053654f, 9.810000f, -0.011658f, -0.023399f, 1.534810f,
    -0.774374f, 0.099730f, 9.810000f, 0.022022f, -0.001447f, 1.568329f,
    -0.796280f, 0.143287f, 9.810000f, -0.041027f, 0.012691f, 1.591933f,
    -0.813152f, 0.183227f, 9.810000f, 0.015217f, -0.003760f, 1.605472f,
    -0.824883f, 0.218541f, 9.810000f, -0.002861f, 0.005410f, 1.608861f,
    -0.831399f, 0.248336f, 9.810000f, 0.014138f, 0.000938f, 1.602077f,
    -0.832659f, 0.271862f, 9.810000f, -0.005036f, 0.064801f, 1.585166f,
    -0.828654f, 0.288522f, 9.810000f, -0.011893f, -0.000752f, 1.558232f,
    -0.819410f, 0.297898f, 9.810000f, -0.004786f, -0.031954f, 1.521446f,
    -0.804986f, 0.299752f, 9.810000f, -0.006909f, -0.029010f, 1.475042f,
    -0.785472f, 0.294038f, 9.810000f, 0.007764f, -0.007857f, 1.419311f,
    -0.760993f, 0.280899f, 9.810000f, 0.004365f, -0.033196f, 1.354608f,
    -0.731702f, 0.260668f, 9.810000f, -0.020531f, 0.002310f, 1.281340f,
    -0.697784f, 0.233855f, 9.810000f, 0.039231f, 0.003758f, 1.199971f,
    -0.659456f, 0.201138f, 9.810000f, -0.010085f, 0.016994f, 1.111016f,
    -0.616958f, 0.163342f, 9.810000f, -0.018453f, 0.000091f, 1.015036f,
    -0.570559f, 0.121421f, 9.810000f, -0.020325f, 0.036920f, 0.912639f,
    -0.520554f, 0.076435f, 9.810000f, 0.000874f, 0.039351f, 0.804472f,
    -0.467257f, 0.029519f, 9.810000f, 0.024496f, 0.022199f, 0.691218f,
    -0.411006f, -0.018143f, 9.810000f, 0.013610f, -0.009604f, 0.573595f,
    -0.352156f, -0.065346f, 9.810000f, -0.007649f, -0.013135f, 0.452345f,
    -0.291080f, -0.110899f, 9.810000f, -0.017168f, 0.025952f, 0.328236f,
    -0.228164f, -0.153653f, 9.810000f, 0.053913f, -0.003326f, 0.202051f,
    -0.163805f, -0.192526f, 9.810000f, 0.019309f, 0.008327f, 0.074589f,
    -0.098410f, -0.226539f, 9.810000f, 0.014287f, -0.006575f, -0.053345f,
    -0.032394f, -0.254832f, 9.810000f, -0.003984f, 0.026083f, -0.180942f,
    0.033828f, -0.276690f, 9.810000f, 0.047676f, -0.014213f, -0.307395f,
    0.099835f, -0.291562f, 9.810000f, -0.016114f, -0.017592f, -0.431904f,
    0.165212f, -0.299072f, 9.810000f, -0.003451f, 0.019985f, -0.553682f,
    0.229544f, -0.299031f, 9.810000f, 0.016991f, 0.040780f, -0.671960f,
    0.292424f, -0.291440f, 9.810000f, 0.019478f, 0.024490f, -0.785990f,
    0.353456f, -0.276490f, 9.810000f, 0.002621f, 0.014710f, -0.895050f,
    0.412253f, -0.254558f, 9.810000f, -0.023397f, 0.020507f, -0.998452f,
    0.468444f, -0.226200f, 9.810000f, 0.017384f, -0.004744f, -1.095541f,
    0.521673f, -0.192130f, 9.810000f, 0.024340f, 0.033123f, -1.185704f,
    0.571604f, -0.153208f, 9.810000f, -0.003107f, 0.013269f, -1.268370f,
    0.617921f, -0.110419f, 9.810000f, -0.028287f, -0.018415f, -1.343017f,
    0.660331f, -0.064841f, 9.810000f, -0.015008f, 0.014590f, -1.409173f,
    0.698567f, -0.017626f, 9.810000f, 0.001791f, -0.007062f, -1.466420f,
    0.732386f, 0.030033f, 9.810000f, 0.004320f, 0.025344f, -1.514396f,
    0.761575f, 0.076935f, 9.810000f, -0.003938f, 0.010221f, -1.552797f,
    0.785948f, 0.121894f, 9.810000f, -0.003226f, -0.002278f, -1.581381f,
    0.805353f, 0.163775f, 9.810000f, -0.011163f, 0.025956f, -1.599967f,
    0.819666f, 0.201521f, 9.810000f, 0.015060f, -0.010594f, -1.608438f,
    0.828797f, 0.234179f, 9.810000f, 0.022371f, -0.003779f, -1.606739f,
    0.832687f, 0.260924f, 9.810000f, -0.017109f, 0.019201f, -1.594882f,
    0.831314f, 0.281081f, 9.810000f, -0.006118f, -0.002287f, -1.572942f,
    0.824684f, 0.294140f, 9.810000f, 0.008643f, -0.006903f, -1.541057f,
    0.812841f, 0.299773f, 9.810000f, -0.026080f, 0.010194f, -1.499429f,
    0.795858f, 0.297837f, 9.810000f, 0.020369f, 0.018051f, -1.448322f,
    0.773844f, 0.288380f, 9.810000f, -0.007908f, -0.002537f, -1.388057f,
    0.746938f, 0.271643f, 9.810000f, -0.019086f, 0.049324f, -1.319017f,
    0.715309f, 0.248046f, 9.810000f, -0.038149f, -0.029204f, -1.241637f,
    0.679157f, 0.218186f, 9.810000f, 0.013885f, 0.020041f, -1.156408f,
    0.638712f, 0.182818f, 9.810000f, 0.012883f, -0.012494f, -1.063867f,
    0.594228f, 0.142833f, 9.810000f, -0.023017f, 0.030119f, -0.964600f,
    0.545988f, 0.099242f, 9.810000f, -0.025498f, -0.041521f, -0.859235f,
    0.494296f, 0.053145f, 9.810000f, 0.031988f, -0.026254f, -0.748437f,
    0.439479f, 0.005706f, 9.810000f, -0.016453f, -0.021697f, -0.632907f,
    0.381883f, -0.041877f, 9.810000f, 0.000738f, 0.021994f, -0.513376f,
    0.321872f, -0.088402f, 9.810000f, 0.029145f, 0.009420f, -0.390599f,
    0.259827f, -0.132696f, 9.810000f, 0.011051f, 0.009870f, -0.265353f,
    0.196139f, -0.173638f, 9.810000f, -0.001235f, 0.022259f, -0.138429f,
    0.131211f, -0.210197f, 9.810000f, 0.028444f, -0.041464f, -0.010630f,
    0.065454f, -0.241448f, 9.810000f, 0.000045f, -0.032024f, 0.117236f,
    -0.000718f, -0.266603f, 9.810000f, -0.002088f, -0.002983f, 0.244361f,
    -0.066884f, -0.285027f, 9.810000f, 0.016961f, 0.016642f, 0.369942f,
    -0.132628f, -0.296253f, 9.810000f, -0.012538f, -0.000262f, 0.493183f,
    -0.197534f, -0.300000f, 9.810000f, -0.002424f, -0.012955f, 0.613306f,
    -0.261190f, -0.296172f, 9.810000f, 0.008681f, 0.012313f, 0.729552f,
    -0.323196f, -0.284865f, 9.810000f, -0.017046f, 0.033899f, 0.841185f,
    -0.383158f, -0.266366f, 9.810000f, -0.012708f, -0.004550f, 0.947500f
};
uint8_t rx_count;

/********************************************************************************/
/*
---------------------------------------gap----------------------------------------
*/
/********************************自定义函数声明********************************/
/*自定义函数声明*/
void led_init(void);
void usart_init(void);
void timer1_1s_config(void);
void dma_config(void);
static void cache_enable(void);
void dma_rev_config(void);
/********************************************************************************/
/*
------------------------------gap-----------------------------------------------
*/			
/*********************************************************************************************************************/
/**
 * @brief  主函数：系统入口点 
 * @note   调试宏 `DEBUG` 控制模型初始化错误的打印，便于开发阶段排查问题
 * @warning 若模型初始化失败，在 DEBUG 模式下会直接返回 -1（但调度器尚未启动，通常需额外处理）
 */
int main(void)
{
		error_type ret_error;
    /* ------------------- 系统级配置 ------------------- */
    cache_enable();                     // 自定义函数：使能并配置 CPU 缓存（具体实现取决于 BSP）
    SCB_EnableICache();/* 使能 ARM Cortex-M 内核的指令缓存（I-Cache） */  
    SCB_EnableDCache(); /* 使能 ARM Cortex-M 内核的数据缓存（D-Cache） */
    mpu_config();
    /* ------------------- 外设初始化 ------------------- */
    led_init();                         // 初始化板载 LED（例如用于状态指示的 LED3/LED4）
                           
    
    timer1_1s_config();                 // 配置 TIMER1 产生 1 秒周期中断（例如用于翻转 LED4，作为心跳指示）
		
		    /* enable DMA1 clock */
    rcu_periph_clock_enable(RCU_DMA1);
    /* enable DMAMUX clock */
    rcu_periph_clock_enable(RCU_DMAMUX);
		nvic_priority_group_set(NVIC_PRIGROUP_PRE4_SUB0);
    nvic_irq_enable(DMA1_Channel7_IRQn, 0, 0);
		
		
		rx_mutex = xSemaphoreCreateBinary();//提前创建避免串口中断bug
		dma_rev_config();
		usart_init();
		usart_receive_config(USART0, USART_RECEIVE_ENABLE);
		usart_dma_transmit_config(USART0, USART_TRANSMIT_DMA_ENABLE);
		usart_dma_receive_config(USART0, USART_RECEIVE_DMA_ENABLE);
    /* ------------------- AI 模型初始化 ------------------- */
    /* 填充模型结构体中的默认参数（如输入/输出张量大小、量化参数等） */
    nn_model_struct_init_transform(imu_error_scalable);

    ret_error = nn_model_init(&imu_error_scalable);
    if(ret_error != ok) {
        rb_printf(&tx_buffer, "imu_fusion_mlp init error: %d\n", ret_error);
        return -1;
    }
		nn_model_struct_init_mlp(imu_fusion_mlp);

    ret_error = nn_model_init(&imu_fusion_mlp);
    if(ret_error != ok) {
        rb_printf(&tx_buffer, "imu_fusion_mlp init error: %d\n", ret_error);
        return -1;
    }
		rb_init(&tx_buffer);
		
    /* ------------------- FreeRTOS 任务创建 ------------------- */
    xTaskCreate((TaskFunction_t)start_task,            // 任务入口函数
                (char *)"start_task",                  // 任务名称（调试用）
                (configSTACK_DEPTH_TYPE)1024,           // 任务栈深度（单位：字）
                (void *)NULL,                          // 传递给任务的参数
                (UBaseType_t)5,                        // 任务优先级（数值越大优先级越高）
                (TaskHandle_t *)&start_task_handle);   // 保存任务句柄（用于后续管理）

    vTaskStartScheduler();

    /* 若调度器因某些错误（如内存不足）而返回，则在此处陷入死循环 */
    while (1) {
        /* 可选：添加错误指示（如点亮红色 LED） */
			printf("error: enter while ,rtos dead\r\n");
			NVIC_SystemReset(); // 添加这一行
    }
}
/******************* freertos任务函数*********************************/
/**
 * @brief 启动任务
 * 
 * @param pvParameters 
*/
void start_task(void *pvParameters)
{	

    // 创建保护模型推理的互斥量
	model_mutex = xSemaphoreCreateMutex();	

	rx_mutex = xSemaphoreCreateBinary();
	fusion_input_data_queue = xQueueCreate(10, INPUT_SIZE_mlp * sizeof(float));
	fusion_output_data_queue = xQueueCreate(10, OUTPUT_SIZE_mlp * sizeof(float));
	scalable_input_data_queue = xQueueCreate(10, INPUT_SIZE_transform * sizeof(float));
	scalable_output_data_queue = xQueueCreate(10, OUTPUT_SIZE_transform * sizeof(float));
#if DEBUG
	if(xSemaphoreTake(tx_buffer.rb_mutex, portMAX_DELAY) == pdTRUE){
		rb_printf(&tx_buffer, "GD32 : i hate bug\r\n");
		xSemaphoreGive(tx_buffer.rb_mutex);
	}
	if(xTaskCreate(led_flash_Task, "led_flash", 128, NULL, 8, &led_flash_task_handle)==-1){
		rb_printf(&tx_buffer, "Failed to create led_flash_Task !\r\n");
	}
	if(xTaskCreate(imu_error_scalable_Task, "imu_error_scalable", 1024 * 10, NULL, 5, &imu_error_scalable_task_handle)== -1){
		printf("Failed to create imu_correction_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(imu_fusion_mlp_Task, "imu_fusion_mlp", 1024 * 10, NULL, 5, &imu_fusion_mlp_task_handle)== -1){
		printf("Failed to create imu_correction_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(fusion_filldata_Task, "fusion_filldata", 1024 * 1, NULL, 5, &fusion_filldata_task_handle)== -1){
		printf("Failed to create fusion_filldata_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(scalable_filldata_Task, "scalable_filldata", 1024 * 1, NULL, 5, &scalable_filldata_task_handle)== -1){
		printf("Failed to create scalable_filldata_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(fusion_process_output_Task, "fusion_process_output", 1024 * 1, NULL, 5, &fusion_process_output_task_handle)== -1){
		printf("Failed to create fusion_filldata_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(scalable_process_output_Task, "scalable_process_output", 1024 * 1, NULL, 5, &scalable_process_output_task_handle)== -1){
		printf("Failed to create scalable_filldata_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(send_messgae_Task, "send_messgae", 1024 * 1, NULL, 4, &send_messgae_task_handle)== -1){
		printf("Failed to create send_messgae_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
	if(xTaskCreate(rev_messgae_Task, "rev_messgae", 1024 * 1, NULL, 4, &rev_messgae_task_handle)== -1){
		printf("Failed to create rev_messgae_Task because errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY!\r\n");
	}
#else
	xTaskCreate(led_flash_Task, "led_flash", 128, NULL, 5, &led_flash_task_handle);
	xTaskCreate(imu_fusion_mlp_Task, "imu_fusion_mlp", 1024 * 4, NULL, 5, &imu_fusion_mlp_task_handle);
	xTaskCreate(imu_error_scalable_Task, "imu_error_scalable", 1024 * 4, NULL, 5, &imu_error_scalable_task_handle);
	xTaskCreate(scalable_filldata_Task, "scalable_filldata", 1024 * 1, NULL, 5, &scalable_filldata_task_handle);
	xTaskCreate(fusion_filldata_Task, "fusion_filldata", 1024 * 1, NULL, 5, &fusion_filldata_task_handle);
	xTaskCreate(fusion_process_output_Task, "fusion_process_output", 1024 * 1, NULL, 5, &fusion_process_output_task_handle);
	xTaskCreate(scalable_process_output_Task, "scalable_process_output", 1024 * 1, NULL, 5, &scalable_process_output_task_handle);
#endif

	vTaskDelete(NULL);  
}
/*
------------------------------gap-----------------------------------------------
*/
/*
	定义led2闪烁任务
	默认1000ms
	指示freertos正常运行
*/
void led_flash_Task(void *pvParameters) {
	while (1){
		if(xSemaphoreTake(tx_buffer.rb_mutex, portMAX_DELAY) == pdTRUE){
			rb_printf(&tx_buffer, "mlp_fps  is %d \r\n trans_fps is %d \r\n", mlp_fps, trans_fps);
			xSemaphoreGive(tx_buffer.rb_mutex);
		}
		mlp_fps = 0;
		trans_fps = 0;
		gd_eval_led_toggle(LED2);
		vTaskDelay(1000);
	}
}
void rev_messgae_Task(void *pvParameters) {
	while (1){	
		if (xSemaphoreTake(rx_mutex, portMAX_DELAY) == pdTRUE){
			SCB_InvalidateDCache_by_Addr((uint32_t *)rxbuffer, BUFFER_SIZE);
			if(xSemaphoreTake(tx_buffer.rb_mutex, portMAX_DELAY) == pdTRUE){
				rb_printf(&tx_buffer, "\r\n rev num is %d  \r\n", rx_count);
				memset(rxbuffer, 0, BUFFER_SIZE);
				SCB_InvalidateDCache_by_Addr((uint32_t *)rxbuffer, BUFFER_SIZE);
				xSemaphoreGive(tx_buffer.rb_mutex);
			}	
		}
		vTaskDelay(100);
	}
}
void send_messgae_Task(void *pvParameters) {
	while (1){
		
		if(rb_used_space(&tx_buffer)){
			memset(send_buffer, 0, BUFFER_SIZE);          // 先清零
			rb_read_all(&tx_buffer, (uint8_t *)send_buffer, BUFFER_SIZE);
			dma_config();
			/* waiting for the transfer to complete */
			while(RESET == g_transfer_complete) {
				vTaskDelay(100);
			}
			
		}
		vTaskDelay(500);
	}
}
/*************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
void fusion_filldata_Task(void *pvParameters){
	while(1){
		if (xQueueSend(fusion_input_data_queue, fusion_data, portMAX_DELAY) != pdPASS) {
			rb_printf(&tx_buffer, "fusion_filldata_Task : data full\r\n");
		}
		vTaskDelay(100);
	}
}
/*************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
void scalable_filldata_Task(void *pvParameters){
	while(1){
		if (xQueueSend(scalable_input_data_queue, trans_data, portMAX_DELAY) != pdPASS) {
			rb_printf(&tx_buffer, "scalable_filldata_Task : data full\r\n");
		}
		vTaskDelay(100);
	}
}
/*************************************************************/
void fusion_process_output_Task(void *pvParameters){
	while(1){
		float output[OUTPUT_SIZE_mlp];
		if (xQueueReceive(fusion_output_data_queue, output, portMAX_DELAY) != pdPASS) {
					rb_printf(&tx_buffer, "fusion_process_output_Task : no data \r\n");
		}
		rb_printf(&tx_buffer, "fusion_process_output_Task : %f %f %f %f %f %f\r\n",output[0],output[1],output[2],output[3],output[4],output[5]);
		vTaskDelay(100);
	}
}
/*************************************************************/
void scalable_process_output_Task(void *pvParameters){
	while(1){
		float output[OUTPUT_SIZE_transform];
		if (xQueueReceive(scalable_output_data_queue, output, portMAX_DELAY) != pdPASS) {
					rb_printf(&tx_buffer, "scalable_process_output_Task : no data \r\n");
		}
		rb_printf(&tx_buffer, "%d : %f %f %f %f %f %f\r\n",80, output[0], output[1],output[2+6*79],output[3+6*79],output[4+6*79],output[5+6*79]);
		vTaskDelay(100);
	}
}
/*************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
/*误差预测任务*/
void imu_error_scalable_Task(void *pvParameters)
{
	while(1)
	{
    error_type ret_error = ok;
    UBaseType_t uxHighWaterMark;
    uint32_t lost_ticks;   // 推理期间丢失的系统滴答数（tick 个数）
    while(1)
    {
			/* 获取模型互斥量 —— 保证推理过程独占共享资源 */
			if (xSemaphoreTake(model_mutex, pdMS_TO_TICKS(1000)) != pdTRUE){
					// 超时处理：未能获取互斥量（可能死锁），可重启或跳过本轮
					rb_printf(&tx_buffer, "imu_error_scalable_Task: Failed get date \r\n");
					vTaskDelay(100);
					continue;
			}
#if DEBUG
			TickType_t total_start = xTaskGetTickCount();
			gd_eval_led_on(LED3);
#endif
			if(xQueueReceive(scalable_input_data_queue, input_data_transform, portMAX_DELAY) != pdPASS) {
					rb_printf(&tx_buffer, "imu_error_scalable_Task : no data \r\n");
			}
			ret_error = nn_model_invoke(&imu_error_scalable);  

/* 检查模型推理返回值 */
			if(ret_error != ok) {
					rb_printf(&tx_buffer, "Model invoke error: %d\r\n", ret_error);
					vTaskDelay(1000);
					continue;
			}

 /* 释放模型互斥量 */
			if (xQueueSend(scalable_output_data_queue, output_data_ptr_transform[0], portMAX_DELAY) != pdPASS) {
					rb_printf(&tx_buffer, "imu_error_scalable_Task : data full\r\n");
			}
			xSemaphoreGive(model_mutex);
#if DEBUG
			gd_eval_led_off(LED3);
			TickType_t total_end = xTaskGetTickCount();
			rb_printf(&tx_buffer, "\r\nimu_error_scalable_Task : Total task cycle time: %u ms\r\n", 
                   (unsigned int)((total_end - total_start) * portTICK_PERIOD_MS));
#endif
			trans_fps++;
/* 延时 然后开始下一轮推理 */
			vTaskDelay(100);
    }
	}
}
/*
------------------------------gap-----------------------------------------------
*/
/*IMU合成任务*/
void imu_fusion_mlp_Task(void *pvParameters)
{
	while(1)
	{
    error_type ret_error = ok;
    uint32_t lost_ticks;   // 推理期间丢失的系统滴答数（tick 个数）
    while(1)
    {
						/* 获取模型互斥量 —— 保证推理过程独占共享资源 */
			if (xSemaphoreTake(model_mutex, pdMS_TO_TICKS(1000)) != pdTRUE){
					// 超时处理：未能获取互斥量（可能死锁），可重启或跳过本轮
					rb_printf(&tx_buffer, "imu_fusion_mlp_Task: Failed get model_mutex \r\n");
					vTaskDelay(100);
					continue;
			}
#if DEBUG
			TickType_t total_start = xTaskGetTickCount();
			gd_eval_led_on(LED4);
#endif
			if (xQueueReceive(fusion_input_data_queue, input_data_mlp, portMAX_DELAY) != pdPASS) {
					rb_printf(&tx_buffer, "imu_fusion_mlp_Task : no data \r\n");
			}
			ret_error = nn_model_invoke(&imu_fusion_mlp);
/* 检查模型推理返回值 */
			if(ret_error != ok) {
					rb_printf(&tx_buffer, "Model invoke error: %d\r\n", ret_error);
					vTaskDelay(1000);
					continue;
			}
			if (xQueueSend(fusion_output_data_queue, output_data_ptr_mlp[0], portMAX_DELAY) != pdPASS) {
				rb_printf(&tx_buffer, "fusion_filldata_Task : data full\r\n");
			}
			xSemaphoreGive(model_mutex);
#if DEBUG
/* 熄灭 LED3，表示本轮任务结束 */
			gd_eval_led_off(LED4);
			TickType_t total_end = xTaskGetTickCount();
			rb_printf(&tx_buffer, "\r\nimu_fusion_mlp_Task : Total task cycle time: %u ms\r\n", (unsigned int)((total_end - total_start) * portTICK_PERIOD_MS));
#endif
			mlp_fps++;
			vTaskDelay(100);
    }
	}
}
/*****************************************************************************/
/*
------------------------------gap-----------------------------------------------
*/
/********************************************************************************/
/*!
 * \brief 配置TIMER1为1秒定时中断
 *        用于指示系统硬件是否正常运行
 */
void timer1_1s_config(void)
{
    timer_parameter_struct timer_initpara;

    /* 1. 使能TIMER1的时钟 */
    rcu_periph_clock_enable(RCU_TIMER1);

    /* 2. 复位TIMER1 */
    timer_deinit(TIMER1);

    /* 3. 初始化定时器参数结构体 */
    timer_struct_para_init(&timer_initpara);

    /* 4. 配置定时器参数：实现1秒定时中断 */
    // 假设 CK_TIMER = 300MHz
    // 设置预分频器值为 30000-1，将计数时钟降为 300MHz / 30000 = 10kHz
    timer_initpara.prescaler         = 29999U;
    // 设置自动重装载值为 10000-1，在10kHz的计数时钟下，产生10000次计数需要1秒
    timer_initpara.period            = 9999U;  // 此处改为 9999
    // 设置为向上计数模式
    timer_initpara.counterdirection  = TIMER_COUNTER_UP;
    // 设置为边沿对齐模式
    timer_initpara.alignedmode       = TIMER_COUNTER_EDGE;
    // 时钟分频，此处不使用，可设为TIMER_CKDIV_DIV1
    timer_initpara.clockdivision     = TIMER_CKDIV_DIV1;
    // 重复计数器，基本定时器用不到，设为0
    timer_initpara.repetitioncounter = 0U;

    /* 5. 根据配置初始化TIMER1 */
    timer_init(TIMER1, &timer_initpara);

    /* 6. 配置NVIC中断优先级并使能中断 */
    nvic_irq_enable(TIMER1_IRQn, 0, 0); // 抢占优先级和子优先级根据实际需求设置

    /* 7. 清除并使能TIMER1的更新中断 */
    timer_interrupt_flag_clear(TIMER1, TIMER_INT_FLAG_UP);
    timer_interrupt_enable(TIMER1, TIMER_INT_UP);

    /* 8. 使能TIMER1计数器 */
    timer_enable(TIMER1);
}



void dma_config(void)
{
    dma_single_data_parameter_struct dma_init_struct;
		g_transfer_complete = RESET;

    /* 刷新 D-Cache，保证 DMA 读到最新数据 */
    SCB_CleanDCache_by_Addr((uint32_t *)send_buffer, BUFFER_SIZE);
    /* deinitialize DMA1 channel7(USART0 tx) */
    dma_deinit(DMA1, DMA_CH7);
    dma_single_data_para_struct_init(&dma_init_struct);

    dma_init_struct.request             = DMA_REQUEST_USART0_TX;
    dma_init_struct.direction           = DMA_MEMORY_TO_PERIPH;
    dma_init_struct.memory0_addr        = (uint32_t)send_buffer;
    dma_init_struct.memory_inc          = DMA_MEMORY_INCREASE_ENABLE;
    dma_init_struct.periph_memory_width = DMA_PERIPH_WIDTH_8BIT;
    dma_init_struct.number              = BUFFER_SIZE;
    dma_init_struct.periph_addr         = USART0_DATA_ADDRESS;
    dma_init_struct.periph_inc          = DMA_PERIPH_INCREASE_DISABLE;
    dma_init_struct.priority            = DMA_PRIORITY_ULTRA_HIGH;
    dma_single_data_mode_init(DMA1, DMA_CH7, &dma_init_struct);
    /* configure DMA mode */
    dma_circulation_disable(DMA1, DMA_CH7);
    /* enable DMA1 channel7 transfer complete interrupt */
    dma_interrupt_enable(DMA1, DMA_CH7, DMA_CHXCTL_FTFIE);
    /* enable DMA1 channel7 */
    dma_channel_enable(DMA1, DMA_CH7);
}
void dma_rev_config(void)
{
    dma_single_data_parameter_struct dma_init_struct;
    /* enable DMA clock */
    rcu_periph_clock_enable(RCU_DMA0);
    rcu_periph_clock_enable(RCU_DMAMUX);

    /* initialize DMA channel 1 */
    dma_deinit(DMA0, DMA_CH1);
    dma_single_data_para_struct_init(&dma_init_struct);
    dma_init_struct.request      = DMA_REQUEST_USART0_RX;
    dma_init_struct.direction    = DMA_PERIPH_TO_MEMORY;
    dma_init_struct.memory0_addr  = (uint32_t)rxbuffer;
    dma_init_struct.memory_inc   = DMA_MEMORY_INCREASE_ENABLE;
    dma_init_struct.periph_memory_width = DMA_PERIPH_WIDTH_8BIT;
    dma_init_struct.number       = BUFFER_SIZE;
    dma_init_struct.periph_addr  = (uint32_t)USART0_RDATA_ADDRESS;
    dma_init_struct.periph_inc   = DMA_PERIPH_INCREASE_DISABLE;
    dma_init_struct.priority     = DMA_PRIORITY_ULTRA_HIGH;
    dma_single_data_mode_init(DMA0, DMA_CH1, &dma_init_struct);

    /* configure DMA mode */
    dma_circulation_disable(DMA0, DMA_CH1);

    /* disable the DMAMUX_MUXCH1 synchronization mode */
    dmamux_synchronization_disable(DMAMUX_MUXCH1);

    /* enable DMA channel 1 */
    dma_channel_enable(DMA0, DMA_CH1);
}


void DMA1_Channel7_IRQHandler(void)
{
    if(dma_interrupt_flag_get(DMA1, DMA_CH7, DMA_INT_FLAG_FTF)) {
        dma_interrupt_flag_clear(DMA1, DMA_CH7, DMA_INT_FLAG_FTF);
        g_transfer_complete = SET;
    }
}
 /*!
    \brief      enable the CPU Chache
    \param[in]  none
    \param[out] none
    \retval     none
*/
static void cache_enable(void)
{
    /* Enable I-Cache */
    SCB_EnableICache();
   
    //指令缓存（Instruction Cache，ICache）：用于存储CPU最近执行的指令，提高程序执行效率。
    //数据缓存（Data Cache，DCache）：用于存储CPU最近访问的数据，提高数据访问效率。
 
    /* Enable D-Cache */
    SCB_EnableDCache();
}
void USART0_IRQHandler(void){

	 BaseType_t xHigherPriorityTaskWoken = pdFALSE;
	if(RESET != usart_interrupt_flag_get(USART0, USART_INT_FLAG_IDLE)) {
		usart_flag_clear(USART0, USART_FLAG_IDLE);
		gd_eval_led_toggle(LED4);
		rx_count = BUFFER_SIZE - (dma_transfer_number_get(DMA0, DMA_CH1));
		rxbuffer[rx_count] = '\0';
		/* disable DMA and reconfigure */
		dma_channel_disable(DMA0, DMA_CH1);
		dma_transfer_number_config(DMA0, DMA_CH1, BUFFER_SIZE);
		dma_flag_clear(DMA0, DMA_CH1, DMA_FLAG_FTF);
		dma_channel_enable(DMA0, DMA_CH1);
		xSemaphoreGiveFromISR(rx_mutex, &xHigherPriorityTaskWoken);
	}
	 portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}