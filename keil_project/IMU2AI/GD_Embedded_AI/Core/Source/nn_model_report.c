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

#include "gd_nn_report.h"
#include "gd32h759i_start.h"

/* global ver */
static char* quantization_type_name[5] = {"float","int8","uint8","int16","uint16"};
/* extern declare */
extern char* operator_name[];
extern char* platform_name;                                        
/*!
    \brief      printf model inference's report
    \param[in]  nn_model_report_struct* report
    \param[out] none
    \retval     none
*/ 
void gd_nn_report_printf(nn_model_report_struct* report){
    
    uint16_t id =0 ,i = 0;
    char* name = NULL;
    float time = 0.0;
    float percent = 0.0;
    char version[7] = {'v','0','.','0','.','0','\0'};
    
    nn_layer_report_struct* cur = report->header;
    if(report){
        printf("\n\n\n- - - - - - - - - - - - - - -  Report Summary Begining- - - - - - - - - - - - - - - - - - - - \n");
        printf("The platform mcu: %s\n",platform_name);
        printf("The system core clock: %d MHz\n",report->core_clock/1000000);
        
        version[1] = report->invoke_frame_version[0];
        version[3] = report->invoke_frame_version[1];
        version[5] = report->invoke_frame_version[2];
        
        printf("The invoke frame version: %s\n",version);
        printf("The model name: %s\n",report->model_name);
        printf("The quantization type: %s\n",quantization_type_name[report->model_quantization_type]);
        printf("The model's layer number: %d\n",report->layer_number);
        printf("The inference time: %.3f us (%.2f ms).\n",report->total_invoke_time, report->total_invoke_time/1000);
        printf("- - - - - - - - - - - - - - -  Report Summary Ending - - - - - - - - - - - - - - - - - - - - -\n");


        printf("- - - - - - - - - - - - - - -  Report Details Begining - - - - - - - - - - - - - - - - - - - - \n");
        printf("--------------------------------------------------------------------------------------\n");
        printf("%*s %*s %*s %*s",12,"id",14,"name",18,"time(us)",16,"percent\n");
        printf("--------------------------------------------------------------------------------------\n");
        for( i = 0; i < report->layer_number; i++){
            id = i;
            name = report->operator_name[i];
            time = cur[i].layer_invoke_time;
            percent = time / report-> total_invoke_time;
            printf("%*d %*s %*.1f %*.1f%%\n",11,id,17,name,14,time,14,percent*100);

        }
        printf("--------------------------------------------------------------------------------------\n");
        printf("- - - - - - - - - - - - - - -  Report Details Ending - - - - - - - - - - - - - - - - - - - - -\n");
    } else {
        printf("- - - - - - - - - - - - - - -  No Creater Reporter - - - - - - - - - - - - - - - - - - - - -\n");
    }
    
}

/*!
    \brief      start systick timing
    \param[in]  none
    \param[out] none
    \retval     none
*/ 
 void gd_nn_measure_time_start(void){
    
    SysTick->CTRL |= SysTick_CLKSource_HCLK; /*SettheSysTickclocksource.*/
    SysTick->LOAD = 0xFFFFFF; /*Timeload(SysTick->LOADis24bit).*/
    SysTick->VAL = 0; /*Emptythecountervalue.*/
    SysTick->CTRL |= SysTick_CTRL_ENABLE_Msk; /*Startthecountdown.*/
    __NOP();/*Waitingforamachinecycle.*/
    sys_count = 0;    
}

/*!
    \brief      get systick timing
    \param[in]  scale: (0xFFFFFF / SystemCoreClock)
    \param[in]  clock: SystemCoreClock
    \param[out] none
    \retval     return timing from start (us)
*/ 
 float gd_nn_measure_time_get(float scale , uint32_t clock){
    
    uint32_t count = SysTick->VAL; /*Readthecountervalue.*/
    //SysTick->CTRL &= ~SysTick_CTRL_ENABLE_Msk; /*Closecounter.*/

    float time = 0.0;
    
    if(clock > 0) {
        time = (float)(0xFFFFFF - count) / (float)clock  + scale * sys_count; /*Calculateprogramruntime.*/
    }
    return time * 1000000;
}

/*!
    \brief      stop systick timing
    \param[in]  none
    \param[out] none
    \retval     none
*/ 
 void gd_nn_measure_time_stop(void){
    
    SysTick->CTRL &= ~SysTick_CTRL_ENABLE_Msk; /*Closecounter.*/
    __NOP();/*Waitingforamachinecycle.*/
    sys_count = 0;
    
}
