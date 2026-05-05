 
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H
 
#include "gd32h7xx.h"

 
/*-----------------------------------------------------------
 * Application specific definitions.
 *
 * These definitions should be adjusted for your particular hardware and
 * application requirements.
 *
 * THESE PARAMETERS ARE DESCRIBED WITHIN THE 'CONFIGURATION' SECTION OF THE
 * FreeRTOS API DOCUMENTATION AVAILABLE ON THE FreeRTOS.org WEB SITE.
 *
 * See http://www.freertos.org/a00110.html
 *----------------------------------------------------------*/
//针对不同的编译器调用不同的stdint.h文件
/* Ensure stdint is only used by the compiler, and not the assembler. */
#if defined(__ICCARM__) || defined(__CC_ARM) || defined(__GNUC__)
	#include <stdint.h>
	extern uint32_t SystemCoreClock;
#endif
 
 
 
 
 
 
 
 
 #define INCLUDE_uxTaskGetStackHighWaterMark    1
 
#define configUSE_TICKLESS_IDLE    0
 
 
 
 
 
 
 
/* 置1：RTOS使用抢占式调度器；置0：RTOS使用协作式调度器（时间片）
 * 
 * 注：在多任务管理机制上，操作系统可以分为抢占式和协作式两种。
 * 协作式操作系统是任务主动释放CPU后，切换到下一个任务。
 * 任务切换的时机完全取决于正在运行的任务。
 */
#define configUSE_PREEMPTION					0
#define configUSE_TIME_SLICING        1  // 必须为1，开启时间片轮转
/*
 * 写入实际的CPU内核时钟频率，也就是CPU指令执行频率，通常称为Fclk
 * Fclk为供给CPU内核的时钟信号，我们所说的cpu主频为 XX MHz，
 * 就是指的这个时钟信号，相应的，1/Fclk即为cpu时钟周期；
 */
#define configCPU_CLOCK_HZ				( SystemCoreClock )
//RTOS系统节拍中断的频率。即一秒中断的次数，每次中断RTOS都会进行任务调度
#define configTICK_RATE_HZ				( ( TickType_t ) 1000 )
//可使用的最大优先级
#define configMAX_PRIORITIES			( 32 )
//空闲任务使用的堆栈大小
#define configMINIMAL_STACK_SIZE		( ( unsigned short ) 128 )
//任务名字字符串长度
#define configMAX_TASK_NAME_LEN			( 10 )
 
#define configUSE_16_BIT_TICKS			0
//空闲任务放弃CPU使用权给其他同优先级的用户任务
#define configIDLE_SHOULD_YIELD			1
//使用互斥信号量
#define configUSE_MUTEXES				1
/* 设置可以注册的信号量和消息队列个数 */
#define configQUEUE_REGISTRY_SIZE		8
//使用递归互斥信号量 
#define configUSE_RECURSIVE_MUTEXES		1
 
#define configUSE_APPLICATION_TASK_TAG	0
//为1时使用计数信号量
#define configUSE_COUNTING_SEMAPHORES	1
 
#define configGENERATE_RUN_TIME_STATS	0
 
/*****************************************************************
              FreeRTOS与内存申请有关配置选项                                               
*****************************************************************/
//支持动态内存申请
#define configSUPPORT_DYNAMIC_ALLOCATION        1    
//支持静态内存
#define configSUPPORT_STATIC_ALLOCATION			0					
//系统所有总的堆大小
#define configTOTAL_HEAP_SIZE					((size_t)(400*1024))   
/***************************************************************
             FreeRTOS与钩子函数有关的配置选项                                            
**************************************************************/
/* 置1：使用空闲钩子（Idle Hook类似于回调函数）；置0：忽略空闲钩子
 * 
 * 空闲任务钩子是一个函数，这个函数由用户来实现，
 * FreeRTOS规定了函数的名字和参数：void vApplicationIdleHook(void )，
 * 这个函数在每个空闲任务周期都会被调用
 * 对于已经删除的RTOS任务，空闲任务可以释放分配给它们的堆栈内存。
 * 因此必须保证空闲任务可以被CPU执行
 * 使用空闲钩子函数设置CPU进入省电模式是很常见的
 * 不可以调用会引起空闲任务阻塞的API函数
 */
 
#define configUSE_IDLE_HOOK				0
 
/* 置1：使用时间片钩子（Tick Hook）；置0：忽略时间片钩子
 * 
 * 
 * 时间片钩子是一个函数，这个函数由用户来实现，
 * FreeRTOS规定了函数的名字和参数：void vApplicationTickHook(void )
 * 时间片中断可以周期性的调用
 * 函数必须非常短小，不能大量使用堆栈，
 * 不能调用以”FromISR" 或 "FROM_ISR”结尾的API函数
 */
 /*xTaskIncrementTick函数是在xPortSysTickHandler中断函数中被调用的。因此，vApplicationTickHook()函数执行的时间必须很短才行*/
#define configUSE_TICK_HOOK				0
 
#define configUSE_MALLOC_FAILED_HOOK	1
/*
 * 大于0时启用堆栈溢出检测功能，如果使用此功能 
 * 用户必须提供一个栈溢出钩子函数，如果使用的话
 * 此值可以为1或者2，因为有两种栈溢出检测方法 */
#define configCHECK_FOR_STACK_OVERFLOW	1
/* 置1：启用堆栈溢出检测功能；置0：忽略堆栈溢出检测功能
 * 此值可以为1或者2，因为有两种栈溢出检测方法 */
//#define configCHECK_FOR_STACK_OVERFLOW  2
/* 置1：启用 malloc 失败钩子函数；置0：忽略 malloc 失败钩子函数
 * 当动态内存申请失败时，会调用此钩子函数
 * 此值可以为1或者2，因为有两种 malloc 失败检测方法 */
//#define configUSE_MALLOC_FAILED_HOOK    1
 
/***********************************************************************
                FreeRTOS与软件定时器有关的配置选项      
**********************************************************************/
/* Software timer definitions. */
#define configUSE_TIMERS				1
#define configTIMER_TASK_PRIORITY		 (configMAX_PRIORITIES-1)  
#define configTIMER_QUEUE_LENGTH		10
#define configTIMER_TASK_STACK_DEPTH	( configMINIMAL_STACK_SIZE * 2 )
 
/* Set the following definitions to 1 to include the API function, or zero
to exclude the API function. */
#define INCLUDE_vTaskPrioritySet		1
#define INCLUDE_uxTaskPriorityGet		1
#define INCLUDE_vTaskDelete				1
#define INCLUDE_vTaskCleanUpResources	1
#define INCLUDE_vTaskSuspend			1
#define INCLUDE_vTaskDelayUntil			1
#define INCLUDE_vTaskDelay				1
 
//#define INCLUDE_vTaskSuspend                          1
#define INCLUDE_xResumeFromISR                          1
#define configUSE_TRACE_FACILITY                        1
#define configUSE_STATS_FORMATTING_FUNCTIONS            1
// 硬件适配
//#define configSYSTICK_CLOCK_HZ      configCPU_CLOCK_HZ // SysTick 时钟源
//#define configSYSTICK_PERIPHERAL    SYSTICK_CLKSOURCE_HCLK // 使用内核时钟
 
// 中断优先级配置（Cortex-M4）
//#define configKERNEL_INTERRUPT_PRIORITY   0xF0 // 最低优先级（FreeRTOS 系统中断）
//#define configMAX_SYSCALL_INTERRUPT_PRIORITY 0x80 // 允许调用 FreeRTOS API 的最高中断优先级
 
/* 放在 FreeRTOSConfig.h 中 */
#define configENABLE_FPU    1
#define configENABLE_MPU    0
 
/* Cortex-M specific definitions. */
#ifdef __NVIC_PRIO_BITS
	/* __BVIC_PRIO_BITS will be specified when CMSIS is being used. */
	#define configPRIO_BITS       		__NVIC_PRIO_BITS
#else
	#define configPRIO_BITS       		4        /* 15 priority levels */
#endif
 
/* 在调用 “设置优先级” 函数时可以使用的最低中断优先级. */
#define configLIBRARY_LOWEST_INTERRUPT_PRIORITY			0xf
 
/* 任何调用中断安全的 FreeRTOS API 函数的中断服务程序所能使用的最高中断优先级。
不要从优先级高于此的任何中断中调用中断安全的 FreeRTOS API 函数！（优先级越高，数值越小） */
//系统可管理的最高中断优先级
#define configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY	5
 
/* 内核端口层自身使用的中断优先级。这些优先级对所有 Cortex-M 端口都是通用的，并且不依赖任何特定的库函数。. */
#define configKERNEL_INTERRUPT_PRIORITY 		( configLIBRARY_LOWEST_INTERRUPT_PRIORITY << (8 - configPRIO_BITS) )
/* !!!! configMAX_SYSCALL_INTERRUPT_PRIORITY must not be set to zero !!!!
See http://www.FreeRTOS.org/RTOS-Cortex-M3-M4.html. */
#define configMAX_SYSCALL_INTERRUPT_PRIORITY 	( configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY << (8 - configPRIO_BITS) )
 
/* Normal assert() semantics without relying on the provision of an assert.h
header file. */
#define configASSERT( x ) if( ( x ) == 0 ) { taskDISABLE_INTERRUPTS(); for( ;; ); }
 
/* Definitions that map the FreeRTOS port interrupt handlers to their CMSIS
standard names. */
/****************************************************************
            FreeRTOS与中断服务函数有关的配置选项                         
****************************************************************/
#define vPortSVCHandler     SVC_Handler
#define xPortPendSVHandler  PendSV_Handler
//#define xPortSysTickHandler SysTick_Handler
 
#endif /* FREERTOS_CONFIG_H */
 