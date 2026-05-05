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

#ifndef GD_NN_USART_H
#define GD_NN_USART_H

#include "gd32h7xx.h"
#include "gd32h759i_start.h"
#include <stdio.h>


//#define MAX_BLOCK_SIZE  0xFFFF
#define MAX_BLOCK_SIZE              ((uint16_t)0x1000U)
#define RX_TX_TIMES_OUT             ((uint32_t)0x00FFFFFFU) 

#define USART0_TDATA_ADDRESS        (&USART_TDATA(USART0))
#define USART0_RDATA_ADDRESS        (&USART_RDATA(USART0))

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* usart configure function, include rcu, dma, gpio */
void usart_configure(void);
/* start receive block, max number:65535 */
uint8_t usart_start_receive_block(uint8_t* rxbuffer, uint32_t len);
/* start receive */
uint8_t usart_start_receive(uint8_t* rxbuffer, uint32_t len);
/* stop receive */
void usart_stop_receive(void);
/* start send block, max number:65535 */
uint8_t usart_start_send_block(uint8_t* txbuffer, uint32_t len);
/* start send */
uint8_t usart_start_send(uint8_t* txbuffer, uint32_t len);
/* stop send */
void usart_stop_send(void);

#endif /* GD_NN_USART_H */
