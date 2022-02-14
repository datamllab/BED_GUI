/**************************************************************************************************
* Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically generated for the yolov1 network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 428,989,904 ops (420,950,768 macc; 8,039,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 89,915,392 ops (86,704,128 macc; 3,211,264 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 176,920,576 ops (173,408,256 macc; 3,512,320 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 11,189,248 ops (10,838,016 macc; 351,232 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 14,551,040 ops (14,450,688 macc; 100,352 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 3,311,616 ops (3,211,264 macc; 100,352 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 58,003,456 ops (57,802,752 macc; 200,704 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 1,831,424 ops (1,605,632 macc; 225,792 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 1,630,720 ops (1,605,632 macc; 25,088 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 1,630,720 ops (1,605,632 macc; 25,088 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 14,500,864 ops (14,450,688 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 457,856 ops (401,408 macc; 56,448 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 3,625,216 ops (3,612,672 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 407,680 ops (401,408 macc; 6,272 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 3,625,216 ops (3,612,672 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16: 7,237,888 ops (7,225,344 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17: 7,237,888 ops (7,225,344 macc; 12,544 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18: 1,822,016 ops (1,806,336 macc; 15,680 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19: 1,809,472 ops (1,806,336 macc; 3,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 20: 203,840 ops (200,704 macc; 3,136 comp; 0 add; 0 mul; 0 bitwise)
    Layer 21: 50,960 ops (50,176 macc; 784 comp; 0 add; 0 mul; 0 bitwise)
    Layer 22: 13,328 ops (12,544 macc; 784 comp; 0 add; 0 mul; 0 bitwise)
    Layer 23: 11,760 ops (11,760 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 298,544 bytes out of 442,368 bytes total (67%)
  Bias memory:   975 bytes out of 2,048 bytes total (48%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 735

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

/* Port pin actions used to signal that processing is active */
#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
