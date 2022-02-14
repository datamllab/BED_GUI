/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include "mxc_sys.h"
#include "mxc_delay.h"
#include "gcfr_regs.h"
#include "fcr_regs.h"
#include "icc.h"
#include "led.h"
#include "tmr.h"
///#include "tornadocnn.h"
//#include "sampledata.h"
#include "utils_faceid.h"
#include "cnn.h"

///Test
#include "sampleoutput.h"
volatile uint32_t cnn_time; // Stopwatch

#define SERIAL_INPUT


#ifdef SERIAL_INPUT
#define CON_BAUD 115200*8   //UART baudrate used for sending data to PC, use max 921600 for serial stream

extern uint32_t cnt;
int ConsoleUARTInit(uint32_t baud)
{

	mxc_uart_regs_t* ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);
	int err;

    NVIC_ClearPendingIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_DisableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
	NVIC_SetPriority(MXC_UART_GET_IRQ(CONSOLE_UART), 1);
    NVIC_EnableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));

	if ((err = MXC_UART_Init(ConsoleUart, baud, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
		return err;
	}
	return 0;
}

#endif

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

void cnn_wait(void)
{
//  while ((*((volatile uint32_t *) 0x50100000) & (1<<12)) != 1<<12) ;
  while (cnn_time == 0)
    __WFI(); // Wait for CNN
//  CNN_COMPLETE; // Signal that processing is complete
//  cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);
}

int cnn_load(int8_t mode)
{
  cnn_init(); // Bring state machine into consistent state

  cnn_load_weights(); // Load kernels
  cnn_load_bias();

  cnn_configure(); // Configure state machine

  cnn_start(); // Start CNN processing
  // load_input(mode); // Load data input
#ifndef SERIAL_INPUT
  load_input(mode); // Load data input via FIFO
#else
  load_input_serial(mode); // Load data input via FIFO
#endif


  return 1;
}

int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t sample_output[] = SAMPLE_OUTPUT;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) return CNN_FAIL;
  }

  return CNN_OK;
}


static int32_t ml_data[CNN_NUM_OUTPUTS];
static q31_t max_box[5 + NUM_CLASSES + 2] = {0};

int main(void)
{

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  // printf("Waiting...\n");
#ifdef SERIAL_INPUT
  // initialize UART
  ConsoleUARTInit(CON_BAUD);
#endif
  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);


  // intial time
  // uint32_t pass_time = utils_get_time_ms();


  // initi and loading weights can be done once
  cnn_init(); // Bring state machine into consistent state

  cnn_load_weights(); // Load kernels
  cnn_load_bias();

  cnn_configure(); // Configure state machine


  volatile uint8_t read_byte = 0;

  while(1){
    const char *startSync = "Start_Sequence";
#if 1
    while (1){
      if (MXC_UART_GetRXFIFOAvailable(MXC_UARTn) > 0) {
        read_byte = MXC_UART_ReadCharacter(MXC_UARTn);
        if (read_byte == 38 || read_byte == 48 || read_byte == 58) {
          break;
        }
      }
      
      uart_write((uint8_t*)startSync, 14);
      MXC_TMR_Delay(MXC_TMR0, MSEC(200));
    }
#else
    read_byte = 58;
#endif
    switch (read_byte) {
      case 38: // Test Image
        uart_read(rxBuffer, sizeof(rxBuffer));
        uart_write(rxBuffer, sizeof(rxBuffer));
        break;

      case 48: // Test Embedding
        uart_read(rxBuffer, sizeof(rxBuffer));

        const char *cnn_receiveDataPass = "Pass_cnn_receiveData";
        uart_write((uint8_t*)cnn_receiveDataPass, 20);
        MXC_TMR_Delay(MXC_TMR0, MSEC(200));
        if (wait_for_feedback() == 0)
          return -1;

        if (!cnn_load(1)) {
          fail();
          return 0;
        }

        const char *cnn_loadPass = "Pass_cnn_load";
        uart_write((uint8_t*)cnn_loadPass, 13);
        MXC_TMR_Delay(MXC_TMR0, MSEC(200));
        if (wait_for_feedback() == 0)
          return -1;
        
        cnn_wait();
        const char *cnn_waitPass = "Pass_cnn_wait";
        uart_write((uint8_t*)cnn_waitPass, 13);
        MXC_TMR_Delay(MXC_TMR0, MSEC(200));
        if (wait_for_feedback() == 0)
          return -1;

        int success_check;
       // success_check = cnn_check();
        const char *cnn_checkPass = "Pass_cnn_check";
        uart_write((uint8_t*)cnn_checkPass, 14);
        MXC_TMR_Delay(MXC_TMR0, MSEC(200));
        if (wait_for_feedback() == 0)
          return -1;

        uart_write((uint8_t*)(&success_check), sizeof(int));

        cnn_unload((uint32_t *) ml_data);
        const char *cnn_unloadPass = "Pass_cnn_unload";
        uart_write((uint8_t*)cnn_unloadPass, 15);
        MXC_TMR_Delay(MXC_TMR0, MSEC(200));
        if (wait_for_feedback() == 0)
          return -1;

        // send embedding to host device
        uart_write((uint8_t *)ml_data, sizeof(ml_data));

        break;

      case 58:
          LED_Off(LED1);
          LED_Off(LED2);

      #ifdef SERIAL_INPUT
        LED_On(LED1);
        LED_Off(LED2);
        cnt = 0;

        cnn_start(); // Start CNN processing

        for (int j=0; j<16; j++){
          uart_read(rxBuffer, sizeof(rxBuffer));

          load_input_serial(0);

        }

        cnn_wait();
        LED_Off(LED1);

#ifdef PATTERN_INPUT
        /// to test against sampleoutput.h
        if (check_output() != CNN_OK) fail();
#endif
      #else
        LED_Off(LED1);
        LED_Off(LED2);
        uart_read(rxBuffer, sizeof(rxBuffer));
        LED_On(LED1);
        if (!cnn_load(0)) {
          fail();
          return 0;
        }

        // cnn_wait();
      #endif



        // cnn_unload((uint32_t *) ml_data);
        cus_cnn_unload((uint32_t *) ml_data);

        // uint32_t time_taken = utils_get_time_ms() - pass_time;


        NMS_max(ml_data, CNN_NUM_OUTPUTS, max_box, cnn_time);

        // send embedding to host device
        // uart_write((uint8_t *)ml_data, sizeof(ml_data)); //sizeof(ml_data)
        uart_write((uint8_t *)max_box, 12 * 4); //sizeof(max_box)

        break;
      
      default:
        break;
    }

  //  printf("cnt: %d\n",cnt);
    cnt = 0;
    const char *endSync = "End_Sequence";
    uart_write((uint8_t*)endSync, 12);
    MXC_TMR_Delay(MXC_TMR0, MSEC(200));
    if (wait_for_feedback() == 0)
      return -1;

  }

  return 0;
}

