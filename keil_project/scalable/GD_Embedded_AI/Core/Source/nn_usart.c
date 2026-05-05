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

#define RX_TX_STATE_OK          1
#define RX_TX_STATE_TIMEOUT     2

#define USER_DMA_TRANS_RCU		RCU_DMA0
#define USER_DMA_TRANS			DMA0
#define USER_DMA_TRANS_CH		DMA_CH0

#define USER_DMA_RECE_RCU		RCU_DMA0
#define USER_DMA_RECE			DMA0
#define USER_DMA_RECE_CH		DMA_CH1


/*!
    \brief      usart_configure function, include RCU, DMA, GPIO
    \param[in]  none
    \param[out] none
    \retval     none
*/
void usart_configure(void)
{
    dma_single_data_parameter_struct dma_init_struct;
  
    /* enable DMA clock */
    rcu_periph_clock_enable(USER_DMA_TRANS_RCU);
    /* enable DMA clock */
    rcu_periph_clock_enable(USER_DMA_RECE_RCU);
    /* enable DMAMUX clock */
    rcu_periph_clock_enable(RCU_DMAMUX); 

    /* initialize DMA channel, transmission channel*/
    dma_deinit(USER_DMA_TRANS, USER_DMA_TRANS_CH);
    dma_single_data_para_struct_init(&dma_init_struct);
    dma_init_struct.request      = DMA_REQUEST_USART0_TX;
    dma_init_struct.direction    = DMA_MEMORY_TO_PERIPH;
    // dma_init_struct.memory0_addr  = (uint32_t)txbuffer;
    dma_init_struct.memory_inc   = DMA_MEMORY_INCREASE_ENABLE;
    dma_init_struct.periph_memory_width = DMA_PERIPH_WIDTH_8BIT;
    // dma_init_struct.number       = ARRAYNUM(txbuffer);
    dma_init_struct.periph_addr  = (uint32_t)USART0_TDATA_ADDRESS;
    dma_init_struct.periph_inc   = DMA_PERIPH_INCREASE_DISABLE;
    dma_init_struct.priority     = DMA_PRIORITY_ULTRA_HIGH;
    dma_single_data_mode_init(USER_DMA_TRANS, USER_DMA_TRANS_CH, &dma_init_struct);

    /* configure DMA mode */
    dma_circulation_disable(USER_DMA_TRANS, USER_DMA_TRANS_CH);

    /* initialize DMA channel, reception channel*/
    dma_deinit(USER_DMA_RECE, USER_DMA_RECE_CH);
    dma_single_data_para_struct_init(&dma_init_struct);
    dma_init_struct.request      = DMA_REQUEST_USART0_RX;
    dma_init_struct.direction    = DMA_PERIPH_TO_MEMORY;

    dma_init_struct.memory_inc   = DMA_MEMORY_INCREASE_ENABLE;
    dma_init_struct.periph_memory_width = DMA_PERIPH_WIDTH_8BIT;

    dma_init_struct.periph_addr  = (uint32_t)USART0_RDATA_ADDRESS;
    dma_init_struct.periph_inc   = DMA_PERIPH_INCREASE_DISABLE;
    dma_init_struct.priority     = DMA_PRIORITY_ULTRA_HIGH;
    dma_single_data_mode_init(USER_DMA_RECE, USER_DMA_RECE_CH, &dma_init_struct);

    /* configure DMA mode */
    dma_circulation_disable(USER_DMA_RECE, USER_DMA_RECE_CH);
    
    /* enable COM GPIO clock */
    rcu_periph_clock_enable(EVAL_COM_GPIO_CLK);

    /* enable USART clock */
    rcu_periph_clock_enable(EVAL_COM_CLK);

    /* connect port to USARTx_Tx */
    gpio_af_set(EVAL_COM_GPIO_PORT, EVAL_COM_AF, EVAL_COM_TX_PIN);

    /* connect port to USARTx_Rx */
    gpio_af_set(EVAL_COM_GPIO_PORT, EVAL_COM_AF, EVAL_COM_RX_PIN);

    /* configure USART Tx as alternate function push-pull */
    gpio_mode_set(EVAL_COM_GPIO_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP,EVAL_COM_TX_PIN);
    gpio_output_options_set(EVAL_COM_GPIO_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ, EVAL_COM_TX_PIN);

    /* configure USART Rx as alternate function push-pull */
    gpio_mode_set(EVAL_COM_GPIO_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, EVAL_COM_RX_PIN);
    gpio_output_options_set(EVAL_COM_GPIO_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ, EVAL_COM_RX_PIN);

    /* USART configure */
    usart_deinit(EVAL_COM);
    usart_baudrate_set(EVAL_COM, 1152000U);
    usart_receive_config(EVAL_COM, USART_RECEIVE_ENABLE);
    usart_transmit_config(EVAL_COM, USART_TRANSMIT_ENABLE);

    usart_enable(EVAL_COM);
    
}

/*!
    \brief      usart receive with block
    \param[in]  rxbuffer: receive buffer
                len: length of bytes
    \param[out] none
    \retval     none
*/
uint8_t usart_start_receive_block(uint8_t* rxbuffer, uint32_t len)
{

    uint8_t ret = RX_TX_STATE_OK;

    /* clear full transfer finish flag */
    dma_flag_clear(USER_DMA_RECE, USER_DMA_RECE_CH, DMA_FLAG_FTF | DMA_FLAG_HTF);
    
    /* configure dma transfer number, buffer address */
    dma_memory_address_config(USER_DMA_RECE, USER_DMA_RECE_CH, DMA_MEMORY_0,(uint32_t)rxbuffer);
    dma_transfer_number_config(USER_DMA_RECE, USER_DMA_RECE_CH, len);
    
    /* enable DMA channel */
    dma_channel_enable(USER_DMA_RECE, USER_DMA_RECE_CH);
    /* USART DMA enable for reception */
    usart_dma_receive_config(USART0, USART_RECEIVE_DMA_ENABLE);    
    
    /* wait DMA channel transfer complete */
    while(RESET == dma_flag_get(USER_DMA_RECE, USER_DMA_RECE_CH, DMA_FLAG_FTF));
    SCB_InvalidateDCache_by_Addr((uint32_t*)rxbuffer, len); 

    return ret;
}

/*!
    \brief      usart receive wrapper 
    \param[in]  rxbuffer: receive buffer
                len: length of bytes
    \param[out] none
    \retval     none
*/
uint8_t usart_start_receive(uint8_t* rxbuffer, uint32_t len)
{
    uint8_t ret = 0;
    uint32_t expected_len = len;
    uint32_t received_len = 0;
    uint32_t cur_received_len = 0;

    while(received_len < expected_len)
    {
        cur_received_len = MIN(MAX_BLOCK_SIZE, expected_len - received_len);
        
        ret = usart_start_receive_block(rxbuffer + received_len, cur_received_len);
        
        if(ret == RX_TX_STATE_OK){
           received_len += cur_received_len;
        } else{
            break;
        }        
    }
    usart_stop_receive();
    return ret;
}

/*!
    \brief      usart stop receive 
    \param[in]  none
    \param[out] none
    \retval     none
*/
void usart_stop_receive(void)
{
    
    /* disable DMA channel */
    dma_channel_disable(USER_DMA_RECE, USER_DMA_RECE_CH);
    /* USART DMA disable for reception */
    usart_dma_receive_config(USART0, USART_RECEIVE_DMA_DISABLE);   
    
    dma_flag_clear(USER_DMA_RECE, USER_DMA_RECE_CH, DMA_FLAG_FTF | DMA_FLAG_HTF);
}

/*!
    \brief      usart send with block 
    \param[in]  txbuffer: send buffer
                len: length of bytes
    \param[out] none
    \retval     none
*/
uint8_t usart_start_send_block(uint8_t* txbuffer, uint32_t len)
{
    
    uint8_t ret = RX_TX_STATE_OK;
          
    /* invalidate d-cache by address */
    SCB_CleanDCache_by_Addr((uint32_t*)txbuffer, len);

    /* clear full transfer finish flag */
    dma_flag_clear(USER_DMA_TRANS, USER_DMA_TRANS_CH, DMA_FLAG_FTF | DMA_FLAG_HTF);
    
    /* configure dma transfer number, buffer address */
    dma_memory_address_config(USER_DMA_TRANS, USER_DMA_TRANS_CH, DMA_MEMORY_0,(uint32_t)txbuffer);
    dma_transfer_number_config(USER_DMA_TRANS, USER_DMA_TRANS_CH, len);

    /* enable DMA channel */
    dma_channel_enable(USER_DMA_TRANS, USER_DMA_TRANS_CH);
    /* USART DMA enable for transmission */
    usart_dma_transmit_config(USART0, USART_TRANSMIT_DMA_ENABLE);
    
    /* wait DMA channel transfer complete */
    while(RESET == dma_flag_get(USER_DMA_TRANS, USER_DMA_TRANS_CH, DMA_FLAG_FTF)); 

    return ret;
}

/*!
    \brief      usart send wrapper 
    \param[in]  txbuffer: send buffer
                len: length of bytes
    \param[out] none
    \retval     none
*/
uint8_t usart_start_send(uint8_t* txbuffer, uint32_t len)
{
    uint32_t expected_len = len;
    uint32_t sent_len = 0;
    uint32_t cur_sent_len = 0;
    uint8_t ret = 0;
    
    while(sent_len < expected_len)
    {
        cur_sent_len = MIN(MAX_BLOCK_SIZE, expected_len - sent_len);
        ret = usart_start_send_block(txbuffer + sent_len, cur_sent_len);
        
        if(ret == RX_TX_STATE_OK){
           sent_len += cur_sent_len;
        } else{
            break;
        }          
        
    }
    usart_stop_send();
    return 1;
}

/*!
    \brief      usart stop send 
    \param[in]  none
    \param[out] none
    \retval     none
*/
void usart_stop_send(void)
{
    /* disable DMA channel */
    dma_channel_disable(USER_DMA_TRANS, USER_DMA_TRANS_CH);
    /* USART DMA disable for transmission */
    usart_dma_transmit_config(USART0, USART_TRANSMIT_DMA_DISABLE);
    dma_flag_clear(USER_DMA_TRANS, USER_DMA_TRANS_CH, DMA_FLAG_FTF | DMA_FLAG_HTF);
    
}
