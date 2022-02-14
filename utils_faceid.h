#ifndef UTILS_FACEID_H_
#define UTILS_FACEID_H_

#include "uart.h"
#include "sampledata.h"

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;


//-----------

// #define PATTERN_INPUT    // if defined, overwrites samples with 0x101112 and checks against sample_output
//-----------
#define CONSOLE_UART 0
#define MXC_UARTn   MXC_UART_GET_UART(CONSOLE_UART)
#define IMG_SIZE 224  // need to change
#define NUM_CLASSES 5  // need to change
#define NUM_GRIDS 7
#define NUM_BOXES 2
#define BOX_DIMENSION (5 * NUM_BOXES)
#define DATA_SIZE_IN (IMG_SIZE * IMG_SIZE * 3 / 16) // for serial load / 16

#define GRID_SIZE (IMG_SIZE / NUM_GRIDS)
#define NUM_CHANNELS (BOX_DIMENSION + NUM_CLASSES)
#define NUM_LUT_BITS 8

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// Data input: HWC (little data): 3x160x120
//static const uint32_t input_0[] = INPUT_0;
extern uint8_t rxBuffer[DATA_SIZE_IN];

int uart_write(uint8_t* data, unsigned int len);

int uart_read(uint8_t* buffer, unsigned int len);

int wait_for_feedback();

void load_input(int8_t mode);

void load_input_serial(int8_t mode);

q31_t q_div(q31_t a, q31_t b);

q31_t q_mul(q31_t a, q31_t b);

q31_t sigmoid(q31_t in);

void inline_softmax_q17p14_q15(q31_t * vec_in, const uint16_t start, const uint16_t end);

void NMS_max(q31_t * vec_in, const uint16_t dim_vec, q31_t* max_box, uint32_t time_taken);

int cus_cnn_unload(uint32_t *out_buf);
uint32_t utils_get_time_ms(void);

#endif
