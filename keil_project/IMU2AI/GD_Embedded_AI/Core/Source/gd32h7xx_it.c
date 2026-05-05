#include "gd32h7xx_it.h"
#include "systick.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "gd32h759i_start.h"
#include <stdio.h>
#include "ringbuffer.h"
volatile uint32_t sys_count = 0;
extern void xPortSysTickHandler(void);
extern RingBuffer tx_buffer;
extern SemaphoreHandle_t rx_mutex;
/*!
    \brief      this function handles NMI exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void NMI_Handler(void)
{
    /* if NMI exception occurs, go to infinite loop */
    while(1) {
    }
}

/*!
    \brief      this function handles HardFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void HardFault_Handler(void)
{
		printf("\r\n!!! HardFault !!!\r\n");
		printf("HFSR: 0x%08X\n", SCB->HFSR);
    printf("CFSR: 0x%08X\n", SCB->CFSR);
    printf("MMFAR: 0x%08X\n", SCB->MMFAR);
    printf("BFAR: 0x%08X\n", SCB->BFAR);
    /* if Hard Fault exception occurs, go to infinite loop */
    while(1) {
    }
}

/*!
    \brief      this function handles MemManage exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void MemManage_Handler(void)
{
	printf("\r\n!!! MemManage Fault !!!\r\n");
    /* if Memory Manage exception occurs, go to infinite loop */
    while(1) {
    }
}

/*!
    \brief      this function handles BusFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void BusFault_Handler(void)
{
	printf("\r\n!!! BusFault !!!\r\n");
    /* if Bus Fault exception occurs, go to infinite loop */
    while(1) {
    }
}

/*!
    \brief      this function handles UsageFault exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void UsageFault_Handler(void)
{
	printf("\r\n!!! UsageFault !!!\r\n");
    /* if Usage Fault exception occurs, go to infinite loop */
    while(1) {
    }
}
/*!
    \brief      this function handles DebugMon exception
    \param[in]  none
    \param[out] none
    \retval     none
*/
void DebugMon_Handler(void)
{
    /* if DebugMon exception occurs, go to infinite loop */
    while(1) {
    }
}



/**
 * @brief 堆栈溢出钩子函数
 * 
 * @param xTask 发生堆栈溢出的任务句柄
 * @param pcTaskName 发生堆栈溢出的任务名称
*/
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    // 堆栈溢出时会到这里
#if DEBUG
	printf("Stack overflow in task: %s\r\n", pcTaskName);
#endif
	taskDISABLE_INTERRUPTS();
	while(1)
	{
	}   
}
/**
 * @brief malloc 失败钩子函数
 * @param  none
 * @retval none
 */
void vApplicationMallocFailedHook(void)
{
#if DEBUG
	printf("FATAL: Memory allocation failed! Check configTOTAL_HEAP_SIZE.\r\n");
#endif
	taskDISABLE_INTERRUPTS();
	while(1)
	{
	}  
}

/*!
 * \brief TIMER1的中断服务函数
 */
void TIMER1_IRQHandler(void)
{
    /* 检查是否为更新中断标志位 */
    if(SET == timer_interrupt_flag_get(TIMER1, TIMER_INT_FLAG_UP)) {
        /* 清除更新中断标志位 */
        timer_interrupt_flag_clear(TIMER1, TIMER_INT_FLAG_UP);

        /* 用户代码：在这里添加每秒需要执行的任务，例如翻转LED */
        gd_eval_led_toggle(LED1); // 以翻转LED为例
				
    }
}

/*!
 * \brief TIMER2的中断服务函数
 实现 vPortSetupTimerInterrupt 函数 */
void vPortSetupTimerInterrupt( void )
{
    /* --- 使能定时器时钟 --- */
    rcu_periph_clock_enable(RCU_TIMER4); // 为你的目标定时器使能时钟

    /* --- 配置定时器基本参数 --- */
    timer_parameter_struct timer_initpara;

    /* 复位定时器配置 */
    timer_deinit(TIMER4);
    timer_struct_para_init(&timer_initpara);

    /* 配置定时器参数 
     * 我们需要产生一个周期为 (1 / configTICK_RATE_HZ) 秒的中断
     * 假设定时器时钟频率 = 系统时钟频率 = SystemCoreClock (例如 120MHz) 
     * 分频系数: prescaler = (定时器时钟频率 / 所需中断频率) - 1
     * 周期值: period = ( (定时器时钟频率 / (prescaler + 1)) / configTICK_RATE_HZ ) - 1
     * 如果 configTICK_RATE_HZ = 1000, 系统时钟 120MHz
     * 我们可以设置 prescaler = 119, period = 999 
     */
    timer_initpara.prescaler         = 299; // 根据你的实际时钟计算
    timer_initpara.alignedmode       = TIMER_COUNTER_EDGE;
    timer_initpara.counterdirection  = TIMER_COUNTER_UP;
    timer_initpara.period            = 999; // 根据你的实际时钟计算
    timer_initpara.clockdivision     = TIMER_CKDIV_DIV1;
    timer_initpara.repetitioncounter = 0;
    timer_init(TIMER4, &timer_initpara);

    /* --- 使能定时器更新中断，并配置NVIC --- */
    timer_interrupt_enable(TIMER4, TIMER_INT_UP);
    timer_enable(TIMER4);

    /* 配置NVIC中断优先级 
     * 确保优先级低于 configMAX_SYSCALL_INTERRUPT_PRIORITY，以保证系统API能在中断中被安全调用
     */
    nvic_irq_enable(TIMER4_IRQn, 15, 0); // 优先级数值越高，实际优先级越低。15是一个安全的低优先级。
}
/* 在 gd32xxxx_it.c 中 */
void TIMER4_IRQHandler(void)
{
    /* 检查是否为更新中断 */
    if(SET == timer_interrupt_flag_get(TIMER4, TIMER_INT_FLAG_UP)) {
        /* 清除中断标志位，这一步很重要 */
        timer_interrupt_flag_clear(TIMER4, TIMER_INT_FLAG_UP);
        
        /* 调用 FreeRTOS 的系统时钟节拍处理函数 */
        if(xTaskGetSchedulerState() != taskSCHEDULER_NOT_STARTED) {
            xPortSysTickHandler();
        }
    }
}
/*************************************初始化函数***********************/


void led_init(void){
		gd_eval_led_init(LED2);
		gd_eval_led_on(LED2);
		gd_eval_led_init(LED1);
		gd_eval_led_on(LED1);
		gd_eval_led_init(LED3);
		gd_eval_led_on(LED3);
		gd_eval_led_init(LED4);
		gd_eval_led_on(LED4);
}
void usart_init(void){
	
	
	nvic_irq_enable(USART0_IRQn, 5, 0);
	
	/* configure EVAL_COM */
	gd_eval_com_init(EVAL_COM);

//	/* enable USART0 receive interrupt */
//	usart_interrupt_enable(EVAL_COM, USART_INT_RBNE);

//	/* enable USART0 transmit interrupt */
//	usart_interrupt_enable(EVAL_COM, USART_INT_TC);
	usart_flag_clear(USART0, USART_FLAG_IDLE);   // 先清除可能存在的悬浮空闲标志
	usart_interrupt_enable(USART0, USART_INT_IDLE);
}
/****************************系统配置函数*************************************************/
/*!
    \brief      mpu config function
    \param[in]  none
    \param[out] none
    \retval     none
*/
void mpu_config(void)
{
    mpu_region_init_struct mpu_init_struct;
    mpu_region_struct_para_init(&mpu_init_struct);

    /* disable the MPU */
    ARM_MPU_Disable();
    ARM_MPU_SetRegion(0, 0);

    /* configure the MPU attributes for the entire 4GB area, Reserved, no access */
    /* This configuration is highly recommended to prevent Speculative Prefetching of external memory, 
       which may cause CPU read locks and even system errors */
    mpu_init_struct.region_base_address  = 0x0;
    mpu_init_struct.region_size          = MPU_REGION_SIZE_4GB;
    mpu_init_struct.access_permission    = MPU_AP_NO_ACCESS;
    mpu_init_struct.access_bufferable    = MPU_ACCESS_NON_BUFFERABLE;
    mpu_init_struct.access_cacheable     = MPU_ACCESS_NON_CACHEABLE;
    mpu_init_struct.access_shareable     = MPU_ACCESS_SHAREABLE;
    mpu_init_struct.region_number        = MPU_REGION_NUMBER0;
    mpu_init_struct.subregion_disable    = 0x87;
    mpu_init_struct.instruction_exec     = MPU_INSTRUCTION_EXEC_NOT_PERMIT;
    mpu_init_struct.tex_type             = MPU_TEX_TYPE0;
    mpu_region_config(&mpu_init_struct);
    mpu_region_enable();

    /* enable the MPU */
    ARM_MPU_Enable(MPU_MODE_PRIV_DEFAULT);
}

/* retarget the C library printf function to the USART */
int fputc(int ch, FILE *f)
{
    usart_data_transmit(EVAL_COM, (uint8_t)ch);
    while(RESET == usart_flag_get(EVAL_COM, USART_FLAG_TBE));
    return ch;
}