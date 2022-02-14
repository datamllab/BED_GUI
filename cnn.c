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

// yolov1
// Created using ./ai8xize.py --verbose --log --overwrite --fifo --test-dir test_sdk --prefix yolov1 --checkpoint-file ../ai8x-training/zaid/yolo/Yolov1_checkpoint-q.pth.tar --config-file networks/yolo-224-hwc-ai85_MXIM.yaml --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint

// DO NOT EDIT - regenerate this file instead!

// Configuring 24 layers:
// Layer 0: 3x224x224 (streaming HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x224x224 output
// Layer 1: 64x224x224 (streaming HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 24x112x112 output
// Layer 2: 24x112x112 (streaming HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 16x56x56 output
// Layer 3: 16x56x56 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 32x56x56 output
// Layer 4: 32x56x56 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x56x56 output
// Layer 5: 32x56x56 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x56x56 output
// Layer 6: 64x56x56 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x28x28 output
// Layer 7: 32x28x28 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x28x28 output
// Layer 8: 64x28x28 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x28x28 output
// Layer 9: 32x28x28 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x28x28 output
// Layer 10: 64x28x28 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x28x28 output
// Layer 11: 32x28x28 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x28x28 output
// Layer 12: 64x28x28 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x14x14 output
// Layer 13: 32x14x14 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x14x14 output
// Layer 14: 64x14x14 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 32x14x14 output
// Layer 15: 32x14x14 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x14x14 output
// Layer 16: 64x14x14 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x14x14 output
// Layer 17: 64x14x14 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x14x14 output
// Layer 18: 64x14x14 (HWC data), 2x2 max pool with stride 2/2, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x7x7 output
// Layer 19: 64x7x7 (HWC data), no pooling, conv2d with kernel size 3x3, stride 1/1, pad 1/1, ReLU, 64x7x7 output
// Layer 20: 64x7x7 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 64x7x7 output
// Layer 21: 64x7x7 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 16x7x7 output
// Layer 22: 16x7x7 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, ReLU, 16x7x7 output
// Layer 23: 16x7x7 (HWC data), no pooling, conv2d with kernel size 1x1, stride 1/1, pad 0/0, no activation, 15x7x7 output

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "gcfr_regs.h"
#include "cnn.h"
#include "weights.h"

void CNN_ISR(void)
{
  // Acknowledge interrupt to all groups
  *((volatile uint32_t *) 0x50100000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50500000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50900000) &= ~((1<<12) | 1);
  *((volatile uint32_t *) 0x50d00000) &= ~((1<<12) | 1);

  CNN_COMPLETE; // Signal that processing is complete
#ifdef CNN_INFERENCE_TIMER
  cnn_time = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);
#else
  cnn_time = 1;
#endif
}

int cnn_continue(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) |= 1; // Re-enable group 0

  return CNN_OK;
}

int cnn_stop(void)
{
  *((volatile uint32_t *) 0x50100000) &= ~1; // Disable group 0

  return CNN_OK;
}

void memcpy32(uint32_t *dst, const uint32_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

// Kernels:
static const uint32_t kernels_0[] = KERNELS_0;
static const uint32_t kernels_1[] = KERNELS_1;
static const uint32_t kernels_2[] = KERNELS_2;
static const uint32_t kernels_3[] = KERNELS_3;
static const uint32_t kernels_4[] = KERNELS_4;
static const uint32_t kernels_5[] = KERNELS_5;
static const uint32_t kernels_6[] = KERNELS_6;
static const uint32_t kernels_7[] = KERNELS_7;
static const uint32_t kernels_8[] = KERNELS_8;
static const uint32_t kernels_9[] = KERNELS_9;
static const uint32_t kernels_10[] = KERNELS_10;
static const uint32_t kernels_11[] = KERNELS_11;
static const uint32_t kernels_12[] = KERNELS_12;
static const uint32_t kernels_13[] = KERNELS_13;
static const uint32_t kernels_14[] = KERNELS_14;
static const uint32_t kernels_15[] = KERNELS_15;
static const uint32_t kernels_16[] = KERNELS_16;
static const uint32_t kernels_17[] = KERNELS_17;
static const uint32_t kernels_18[] = KERNELS_18;
static const uint32_t kernels_19[] = KERNELS_19;
static const uint32_t kernels_20[] = KERNELS_20;
static const uint32_t kernels_21[] = KERNELS_21;
static const uint32_t kernels_22[] = KERNELS_22;
static const uint32_t kernels_23[] = KERNELS_23;
static const uint32_t kernels_24[] = KERNELS_24;
static const uint32_t kernels_25[] = KERNELS_25;
static const uint32_t kernels_26[] = KERNELS_26;
static const uint32_t kernels_27[] = KERNELS_27;
static const uint32_t kernels_28[] = KERNELS_28;
static const uint32_t kernels_29[] = KERNELS_29;
static const uint32_t kernels_30[] = KERNELS_30;
static const uint32_t kernels_31[] = KERNELS_31;
static const uint32_t kernels_32[] = KERNELS_32;
static const uint32_t kernels_33[] = KERNELS_33;
static const uint32_t kernels_34[] = KERNELS_34;
static const uint32_t kernels_35[] = KERNELS_35;
static const uint32_t kernels_36[] = KERNELS_36;
static const uint32_t kernels_37[] = KERNELS_37;
static const uint32_t kernels_38[] = KERNELS_38;
static const uint32_t kernels_39[] = KERNELS_39;
static const uint32_t kernels_40[] = KERNELS_40;
static const uint32_t kernels_41[] = KERNELS_41;
static const uint32_t kernels_42[] = KERNELS_42;
static const uint32_t kernels_43[] = KERNELS_43;
static const uint32_t kernels_44[] = KERNELS_44;
static const uint32_t kernels_45[] = KERNELS_45;
static const uint32_t kernels_46[] = KERNELS_46;
static const uint32_t kernels_47[] = KERNELS_47;
static const uint32_t kernels_48[] = KERNELS_48;
static const uint32_t kernels_49[] = KERNELS_49;
static const uint32_t kernels_50[] = KERNELS_50;
static const uint32_t kernels_51[] = KERNELS_51;
static const uint32_t kernels_52[] = KERNELS_52;
static const uint32_t kernels_53[] = KERNELS_53;
static const uint32_t kernels_54[] = KERNELS_54;
static const uint32_t kernels_55[] = KERNELS_55;
static const uint32_t kernels_56[] = KERNELS_56;
static const uint32_t kernels_57[] = KERNELS_57;
static const uint32_t kernels_58[] = KERNELS_58;
static const uint32_t kernels_59[] = KERNELS_59;
static const uint32_t kernels_60[] = KERNELS_60;
static const uint32_t kernels_61[] = KERNELS_61;
static const uint32_t kernels_62[] = KERNELS_62;
static const uint32_t kernels_63[] = KERNELS_63;

int cnn_load_weights(void)
{
  *((volatile uint8_t *) 0x50180001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50180000, kernels_0, 1292);
  *((volatile uint8_t *) 0x50184001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50184000, kernels_1, 1292);
  *((volatile uint8_t *) 0x50188001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50188000, kernels_2, 1292);
  *((volatile uint8_t *) 0x5018c101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5018c000, kernels_3, 1148);
  *((volatile uint8_t *) 0x50190001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50190000, kernels_4, 1292);
  *((volatile uint8_t *) 0x50194001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50194000, kernels_5, 1292);
  *((volatile uint8_t *) 0x50198001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50198000, kernels_6, 1292);
  *((volatile uint8_t *) 0x5019c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5019c000, kernels_7, 1292);
  *((volatile uint8_t *) 0x501a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a0000, kernels_8, 1292);
  *((volatile uint8_t *) 0x501a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a4000, kernels_9, 1292);
  *((volatile uint8_t *) 0x501a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501a8000, kernels_10, 1292);
  *((volatile uint8_t *) 0x501ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501ac000, kernels_11, 1292);
  *((volatile uint8_t *) 0x501b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b0000, kernels_12, 1292);
  *((volatile uint8_t *) 0x501b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b4000, kernels_13, 1292);
  *((volatile uint8_t *) 0x501b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501b8000, kernels_14, 1292);
  *((volatile uint8_t *) 0x501bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x501bc000, kernels_15, 1292);
  *((volatile uint8_t *) 0x50580001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50580000, kernels_16, 1274);
  *((volatile uint8_t *) 0x50584001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50584000, kernels_17, 1274);
  *((volatile uint8_t *) 0x50588001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50588000, kernels_18, 1274);
  *((volatile uint8_t *) 0x5058c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5058c000, kernels_19, 1274);
  *((volatile uint8_t *) 0x50590101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50590000, kernels_20, 1130);
  *((volatile uint8_t *) 0x50594101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50594000, kernels_21, 1130);
  *((volatile uint8_t *) 0x50598101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50598000, kernels_22, 1130);
  *((volatile uint8_t *) 0x5059c101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5059c000, kernels_23, 1130);
  *((volatile uint8_t *) 0x505a0101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a0000, kernels_24, 1130);
  *((volatile uint8_t *) 0x505a4101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a4000, kernels_25, 1130);
  *((volatile uint8_t *) 0x505a8101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505a8000, kernels_26, 1130);
  *((volatile uint8_t *) 0x505ac101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505ac000, kernels_27, 1130);
  *((volatile uint8_t *) 0x505b0101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b0000, kernels_28, 1130);
  *((volatile uint8_t *) 0x505b4101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b4000, kernels_29, 1130);
  *((volatile uint8_t *) 0x505b8101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505b8000, kernels_30, 1130);
  *((volatile uint8_t *) 0x505bc101) = 0x01; // Set address
  memcpy32((uint32_t *) 0x505bc000, kernels_31, 1130);
  *((volatile uint8_t *) 0x50980041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50980000, kernels_32, 1238);
  *((volatile uint8_t *) 0x50984041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50984000, kernels_33, 1238);
  *((volatile uint8_t *) 0x50988041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50988000, kernels_34, 1238);
  *((volatile uint8_t *) 0x5098c041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5098c000, kernels_35, 1238);
  *((volatile uint8_t *) 0x50990041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50990000, kernels_36, 1238);
  *((volatile uint8_t *) 0x50994041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50994000, kernels_37, 1238);
  *((volatile uint8_t *) 0x50998041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50998000, kernels_38, 1238);
  *((volatile uint8_t *) 0x5099c041) = 0x01; // Set address
  memcpy32((uint32_t *) 0x5099c000, kernels_39, 1238);
  *((volatile uint8_t *) 0x509a0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a0000, kernels_40, 1274);
  *((volatile uint8_t *) 0x509a4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a4000, kernels_41, 1274);
  *((volatile uint8_t *) 0x509a8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509a8000, kernels_42, 1274);
  *((volatile uint8_t *) 0x509ac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509ac000, kernels_43, 1274);
  *((volatile uint8_t *) 0x509b0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b0000, kernels_44, 1274);
  *((volatile uint8_t *) 0x509b4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b4000, kernels_45, 1274);
  *((volatile uint8_t *) 0x509b8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509b8000, kernels_46, 1274);
  *((volatile uint8_t *) 0x509bc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x509bc000, kernels_47, 1274);
  *((volatile uint8_t *) 0x50d80001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d80000, kernels_48, 1274);
  *((volatile uint8_t *) 0x50d84001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d84000, kernels_49, 1274);
  *((volatile uint8_t *) 0x50d88001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d88000, kernels_50, 1274);
  *((volatile uint8_t *) 0x50d8c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d8c000, kernels_51, 1274);
  *((volatile uint8_t *) 0x50d90001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d90000, kernels_52, 1274);
  *((volatile uint8_t *) 0x50d94001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d94000, kernels_53, 1274);
  *((volatile uint8_t *) 0x50d98001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d98000, kernels_54, 1274);
  *((volatile uint8_t *) 0x50d9c001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50d9c000, kernels_55, 1274);
  *((volatile uint8_t *) 0x50da0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da0000, kernels_56, 1274);
  *((volatile uint8_t *) 0x50da4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da4000, kernels_57, 1274);
  *((volatile uint8_t *) 0x50da8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50da8000, kernels_58, 1274);
  *((volatile uint8_t *) 0x50dac001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50dac000, kernels_59, 1274);
  *((volatile uint8_t *) 0x50db0001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db0000, kernels_60, 1274);
  *((volatile uint8_t *) 0x50db4001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db4000, kernels_61, 1274);
  *((volatile uint8_t *) 0x50db8001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50db8000, kernels_62, 1274);
  *((volatile uint8_t *) 0x50dbc001) = 0x01; // Set address
  memcpy32((uint32_t *) 0x50dbc000, kernels_63, 1274);

  return CNN_OK;
}

static const uint8_t bias_0[] = BIAS_0;
static const uint8_t bias_1[] = BIAS_1;
static const uint8_t bias_2[] = BIAS_2;
static const uint8_t bias_3[] = BIAS_3;

static void memcpy_8to32(uint32_t *dst, const uint8_t *src, int n)
{
  while (n-- > 0) {
    *dst++ = *src++;
  }
}

int cnn_load_bias(void)
{
  memcpy_8to32((uint32_t *) 0x50108000, bias_0, sizeof(uint8_t) * 256);
  memcpy_8to32((uint32_t *) 0x50508000, bias_1, sizeof(uint8_t) * 240);
  memcpy_8to32((uint32_t *) 0x50908000, bias_2, sizeof(uint8_t) * 240);
  memcpy_8to32((uint32_t *) 0x50d08000, bias_3, sizeof(uint8_t) * 239);

  return CNN_OK;
}

int cnn_init(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x50100000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50100004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50100008) = 0x00000017; // Layer count
  *((volatile uint32_t *) 0x50500000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50500004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50500008) = 0x00000017; // Layer count
  *((volatile uint32_t *) 0x50900000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50900004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50900008) = 0x00000017; // Layer count
  *((volatile uint32_t *) 0x50d00000) = 0x00108008; // Stop SM
  *((volatile uint32_t *) 0x50d00004) = 0x0000040e; // SRAM control
  *((volatile uint32_t *) 0x50d00008) = 0x00000017; // Layer count

  return CNN_OK;
}

int cnn_configure(void)
{
  // Layer 0 group 0
  *((volatile uint32_t *) 0x50100010) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50100090) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50100310) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50100410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50100a10) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100610) = 0x000001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50100690) = 0x000000df; // TRAM ptr max
  *((volatile uint32_t *) 0x50100790) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100710) = 0x00070007; // Mask and processor enables
  *((volatile uint32_t *) 0x50100810) = 0x00000001; // Stream processing start
  *((volatile uint32_t *) 0x50100910) = 0x00000002; // Rollover
  *((volatile uint32_t *) 0x50100990) = 0x0000c400; // Input frame size

  // Layer 0 group 1
  *((volatile uint32_t *) 0x50500010) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50500090) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50500310) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50500410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a10) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500610) = 0x000001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50500690) = 0x000000df; // TRAM ptr max
  *((volatile uint32_t *) 0x50500790) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500810) = 0x00000001; // Stream processing start
  *((volatile uint32_t *) 0x50500910) = 0x00000002; // Rollover
  *((volatile uint32_t *) 0x50500990) = 0x0000c400; // Input frame size

  // Layer 0 group 2
  *((volatile uint32_t *) 0x50900010) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50900090) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50900310) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50900410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a10) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900610) = 0x000001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50900690) = 0x000000df; // TRAM ptr max
  *((volatile uint32_t *) 0x50900790) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900810) = 0x00000001; // Stream processing start
  *((volatile uint32_t *) 0x50900910) = 0x00000002; // Rollover
  *((volatile uint32_t *) 0x50900990) = 0x0000c400; // Input frame size

  // Layer 0 group 3
  *((volatile uint32_t *) 0x50d00010) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50d00090) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50d00310) = 0x00000800; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00410) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00590) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a10) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00610) = 0x000001f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00690) = 0x000000df; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00790) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d00810) = 0x00000001; // Stream processing start
  *((volatile uint32_t *) 0x50d00910) = 0x00000002; // Rollover
  *((volatile uint32_t *) 0x50d00990) = 0x0000c400; // Input frame size

  // Layer 1 group 0
  *((volatile uint32_t *) 0x50100014) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50100094) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50100194) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100214) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50100294) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100314) = 0x00014c00; // SRAM write ptr
  *((volatile uint32_t *) 0x50100414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50100594) = 0x0000eba0; // Layer control
  *((volatile uint32_t *) 0x50100a14) = 0x0000b800; // Layer control 2
  *((volatile uint32_t *) 0x50100614) = 0x020002b8; // Mask offset and count
  *((volatile uint32_t *) 0x50100694) = 0x00e0014f; // TRAM ptr max
  *((volatile uint32_t *) 0x50100794) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x50100714) = 0xffffffff; // Mask and processor enables
  *((volatile uint32_t *) 0x50100814) = 0x000002ac; // Stream processing start
  *((volatile uint32_t *) 0x50100894) = 0x00e40021; // Stream processing delta
  *((volatile uint32_t *) 0x50100914) = 0x0000038e; // Rollover

  // Layer 1 group 1
  *((volatile uint32_t *) 0x50500014) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50500094) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50500194) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500214) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50500294) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500314) = 0x00014c00; // SRAM write ptr
  *((volatile uint32_t *) 0x50500414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50500594) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a14) = 0x0000b800; // Layer control 2
  *((volatile uint32_t *) 0x50500614) = 0x020002b8; // Mask offset and count
  *((volatile uint32_t *) 0x50500694) = 0x00e0014f; // TRAM ptr max
  *((volatile uint32_t *) 0x50500794) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x50500714) = 0xffffffff; // Mask and processor enables
  *((volatile uint32_t *) 0x50500814) = 0x000002ac; // Stream processing start
  *((volatile uint32_t *) 0x50500894) = 0x00e40021; // Stream processing delta
  *((volatile uint32_t *) 0x50500914) = 0x0000038e; // Rollover

  // Layer 1 group 2
  *((volatile uint32_t *) 0x50900014) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50900094) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50900194) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900214) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50900294) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900314) = 0x00014c00; // SRAM write ptr
  *((volatile uint32_t *) 0x50900414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50900594) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a14) = 0x0000b800; // Layer control 2
  *((volatile uint32_t *) 0x50900614) = 0x020002b8; // Mask offset and count
  *((volatile uint32_t *) 0x50900694) = 0x00e0014f; // TRAM ptr max
  *((volatile uint32_t *) 0x50900794) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x50900714) = 0xffffffff; // Mask and processor enables
  *((volatile uint32_t *) 0x50900814) = 0x000002ac; // Stream processing start
  *((volatile uint32_t *) 0x50900894) = 0x00e40021; // Stream processing delta
  *((volatile uint32_t *) 0x50900914) = 0x0000038e; // Rollover

  // Layer 1 group 3
  *((volatile uint32_t *) 0x50d00014) = 0x000100e1; // Rows
  *((volatile uint32_t *) 0x50d00094) = 0x000100e1; // Columns
  *((volatile uint32_t *) 0x50d00194) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00214) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d00294) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00314) = 0x00014c00; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00414) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00514) = 0x00000800; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00594) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a14) = 0x0000b800; // Layer control 2
  *((volatile uint32_t *) 0x50d00614) = 0x020002b8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00694) = 0x00e0014f; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00794) = 0x00026000; // Post processing register
  *((volatile uint32_t *) 0x50d00714) = 0xffffffff; // Mask and processor enables
  *((volatile uint32_t *) 0x50d00814) = 0x000002ac; // Stream processing start
  *((volatile uint32_t *) 0x50d00894) = 0x00e40021; // Stream processing delta
  *((volatile uint32_t *) 0x50d00914) = 0x0000038e; // Rollover

  // Layer 2 group 0
  *((volatile uint32_t *) 0x50100018) = 0x00010071; // Rows
  *((volatile uint32_t *) 0x50100098) = 0x00010071; // Columns
  *((volatile uint32_t *) 0x50100198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50100298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100318) = 0x00003000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100518) = 0x00000c00; // SRAM read ptr
  *((volatile uint32_t *) 0x50100598) = 0x0000cba0; // Layer control
  *((volatile uint32_t *) 0x50100a18) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50100618) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50100698) = 0x01500187; // TRAM ptr max
  *((volatile uint32_t *) 0x50100798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100818) = 0x0000015c; // Stream processing start
  *((volatile uint32_t *) 0x50100898) = 0x00740022; // Stream processing delta
  *((volatile uint32_t *) 0x50100918) = 0x000001ce; // Rollover

  // Layer 2 group 1
  *((volatile uint32_t *) 0x50500018) = 0x00010071; // Rows
  *((volatile uint32_t *) 0x50500098) = 0x00010071; // Columns
  *((volatile uint32_t *) 0x50500198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50500298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500318) = 0x00003000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500518) = 0x00000c00; // SRAM read ptr
  *((volatile uint32_t *) 0x50500598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a18) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50500618) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50500698) = 0x01500187; // TRAM ptr max
  *((volatile uint32_t *) 0x50500798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500818) = 0x0000015c; // Stream processing start
  *((volatile uint32_t *) 0x50500898) = 0x00740022; // Stream processing delta
  *((volatile uint32_t *) 0x50500918) = 0x000001ce; // Rollover

  // Layer 2 group 2
  *((volatile uint32_t *) 0x50900018) = 0x00010071; // Rows
  *((volatile uint32_t *) 0x50900098) = 0x00010071; // Columns
  *((volatile uint32_t *) 0x50900198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50900298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900318) = 0x00003000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900518) = 0x00000c00; // SRAM read ptr
  *((volatile uint32_t *) 0x50900598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a18) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50900618) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50900698) = 0x01500187; // TRAM ptr max
  *((volatile uint32_t *) 0x50900798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900718) = 0xff00ff00; // Mask and processor enables
  *((volatile uint32_t *) 0x50900818) = 0x0000015c; // Stream processing start
  *((volatile uint32_t *) 0x50900898) = 0x00740022; // Stream processing delta
  *((volatile uint32_t *) 0x50900918) = 0x000001ce; // Rollover

  // Layer 2 group 3
  *((volatile uint32_t *) 0x50d00018) = 0x00010071; // Rows
  *((volatile uint32_t *) 0x50d00098) = 0x00010071; // Columns
  *((volatile uint32_t *) 0x50d00198) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00218) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d00298) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00318) = 0x00003000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00418) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00518) = 0x00000c00; // SRAM read ptr
  *((volatile uint32_t *) 0x50d00598) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a18) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50d00618) = 0x00000078; // Mask offset and count
  *((volatile uint32_t *) 0x50d00698) = 0x01500187; // TRAM ptr max
  *((volatile uint32_t *) 0x50d00798) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00718) = 0xffffffff; // Mask and processor enables
  *((volatile uint32_t *) 0x50d00818) = 0x0000015c; // Stream processing start
  *((volatile uint32_t *) 0x50d00898) = 0x00740022; // Stream processing delta
  *((volatile uint32_t *) 0x50d00918) = 0x000001ce; // Rollover

  // Layer 3 group 0
  *((volatile uint32_t *) 0x5010001c) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x5010009c) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x5010031c) = 0x00010000; // SRAM write ptr
  *((volatile uint32_t *) 0x5010041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5010059c) = 0x0000ab20; // Layer control
  *((volatile uint32_t *) 0x50100a1c) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x5010061c) = 0x000000f8; // Mask offset and count
  *((volatile uint32_t *) 0x5010069c) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x5010079c) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5010071c) = 0xfff0fff0; // Mask and processor enables

  // Layer 3 group 1
  *((volatile uint32_t *) 0x5050001c) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x5050009c) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x5050031c) = 0x00010000; // SRAM write ptr
  *((volatile uint32_t *) 0x5050041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5050059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a1c) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x5050061c) = 0x000000f8; // Mask offset and count
  *((volatile uint32_t *) 0x5050069c) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x5050079c) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5050071c) = 0x000f000f; // Mask and processor enables

  // Layer 3 group 2
  *((volatile uint32_t *) 0x5090001c) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x5090009c) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x5090031c) = 0x00010000; // SRAM write ptr
  *((volatile uint32_t *) 0x5090041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x5090059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a1c) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x5090061c) = 0x000000f8; // Mask offset and count
  *((volatile uint32_t *) 0x5090069c) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x5090079c) = 0x00024000; // Post processing register

  // Layer 3 group 3
  *((volatile uint32_t *) 0x50d0001c) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x50d0009c) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x50d0031c) = 0x00010000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d0041c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0051c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d0059c) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a1c) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d0061c) = 0x000000f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d0069c) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x50d0079c) = 0x00025080; // Post processing register

  // Layer 4 group 0
  *((volatile uint32_t *) 0x50100020) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x501000a0) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x50100320) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003a0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005a0) = 0x0000cb20; // Layer control
  *((volatile uint32_t *) 0x50100a20) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100620) = 0x04800578; // Mask offset and count
  *((volatile uint32_t *) 0x50100120) = 0x00000100; // 1D

  // Layer 4 group 1
  *((volatile uint32_t *) 0x50500020) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x505000a0) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x50500320) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003a0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a20) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500620) = 0x04800578; // Mask offset and count
  *((volatile uint32_t *) 0x50500120) = 0x00000100; // 1D

  // Layer 4 group 2
  *((volatile uint32_t *) 0x50900020) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x509000a0) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x50900320) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003a0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a20) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900620) = 0x04800578; // Mask offset and count
  *((volatile uint32_t *) 0x50900120) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900720) = 0xffffffff; // Mask and processor enables

  // Layer 4 group 3
  *((volatile uint32_t *) 0x50d00020) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x50d000a0) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x50d00320) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003a0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00420) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005a0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a20) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00620) = 0x04800578; // Mask offset and count
  *((volatile uint32_t *) 0x50d00120) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007a0) = 0x000010a0; // Post processing register
  *((volatile uint32_t *) 0x50d00720) = 0xffffffff; // Mask and processor enables

  // Layer 5 group 0
  *((volatile uint32_t *) 0x50100024) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x501000a4) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x50100424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100524) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005a4) = 0x00002b20; // Layer control
  *((volatile uint32_t *) 0x50100a24) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100624) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x501006a4) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x501007a4) = 0x00025000; // Post processing register
  *((volatile uint32_t *) 0x50100724) = 0xffffffff; // Mask and processor enables

  // Layer 5 group 1
  *((volatile uint32_t *) 0x50500024) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x505000a4) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x50500424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500524) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005a4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a24) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500624) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x505006a4) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x505007a4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500724) = 0xffffffff; // Mask and processor enables

  // Layer 5 group 2
  *((volatile uint32_t *) 0x50900024) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x509000a4) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x50900424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900524) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005a4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a24) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900624) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x509006a4) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x509007a4) = 0x00024000; // Post processing register

  // Layer 5 group 3
  *((volatile uint32_t *) 0x50d00024) = 0x00010039; // Rows
  *((volatile uint32_t *) 0x50d000a4) = 0x00010039; // Columns
  *((volatile uint32_t *) 0x50d00424) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00524) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005a4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a24) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00624) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006a4) = 0x00000037; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007a4) = 0x00024000; // Post processing register

  // Layer 6 group 0
  *((volatile uint32_t *) 0x50100028) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x501000a8) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x501001a8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100228) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002a8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100328) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003a8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005a8) = 0x0000eba0; // Layer control
  *((volatile uint32_t *) 0x50100a28) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100628) = 0x2ac02bb8; // Mask offset and count
  *((volatile uint32_t *) 0x50100128) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007a8) = 0x000230c0; // Post processing register
  *((volatile uint32_t *) 0x50100728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 1
  *((volatile uint32_t *) 0x50500028) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x505000a8) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x505001a8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500228) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002a8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500328) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003a8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005a8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a28) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500628) = 0x2ac02bb8; // Mask offset and count
  *((volatile uint32_t *) 0x50500128) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007a8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 2
  *((volatile uint32_t *) 0x50900028) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x509000a8) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x509001a8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900228) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002a8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900328) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003a8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005a8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a28) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900628) = 0x2ac02bb8; // Mask offset and count
  *((volatile uint32_t *) 0x50900128) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007a8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900728) = 0xffffffff; // Mask and processor enables

  // Layer 6 group 3
  *((volatile uint32_t *) 0x50d00028) = 0x00000037; // Rows
  *((volatile uint32_t *) 0x50d000a8) = 0x00000037; // Columns
  *((volatile uint32_t *) 0x50d001a8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00228) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002a8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00328) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003a8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00428) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005a8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a28) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00628) = 0x2ac02bb8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00128) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007a8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00728) = 0xffffffff; // Mask and processor enables

  // Layer 7 group 0
  *((volatile uint32_t *) 0x5010002c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x501000ac) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5010042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010052c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005ac) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a2c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5010062c) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x501006ac) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x501007ac) = 0x00024000; // Post processing register

  // Layer 7 group 1
  *((volatile uint32_t *) 0x5050002c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x505000ac) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5050042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050052c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a2c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5050062c) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x505006ac) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x505007ac) = 0x00025000; // Post processing register

  // Layer 7 group 2
  *((volatile uint32_t *) 0x5090002c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x509000ac) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5090042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090052c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a2c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5090062c) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x509006ac) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x509007ac) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090072c) = 0xffffffff; // Mask and processor enables

  // Layer 7 group 3
  *((volatile uint32_t *) 0x50d0002c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x50d000ac) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50d0042c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0052c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005ac) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a2c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d0062c) = 0x02c004b8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006ac) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007ac) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d0072c) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 0
  *((volatile uint32_t *) 0x50100030) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x501000b0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50100330) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005b0) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a30) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100630) = 0x2be02cd8; // Mask offset and count
  *((volatile uint32_t *) 0x50100130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007b0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 1
  *((volatile uint32_t *) 0x50500030) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x505000b0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50500330) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a30) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500630) = 0x2be02cd8; // Mask offset and count
  *((volatile uint32_t *) 0x50500130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007b0) = 0x000230c0; // Post processing register
  *((volatile uint32_t *) 0x50500730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 2
  *((volatile uint32_t *) 0x50900030) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x509000b0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50900330) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a30) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900630) = 0x2be02cd8; // Mask offset and count
  *((volatile uint32_t *) 0x50900130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007b0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900730) = 0xffffffff; // Mask and processor enables

  // Layer 8 group 3
  *((volatile uint32_t *) 0x50d00030) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x50d000b0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50d00330) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003b0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00430) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005b0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a30) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00630) = 0x2be02cd8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00130) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007b0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00730) = 0xffffffff; // Mask and processor enables

  // Layer 9 group 0
  *((volatile uint32_t *) 0x50100034) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x501000b4) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50100434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100534) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005b4) = 0x00006b20; // Layer control
  *((volatile uint32_t *) 0x50100a34) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100634) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006b4) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x501007b4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100734) = 0xffffffff; // Mask and processor enables

  // Layer 9 group 1
  *((volatile uint32_t *) 0x50500034) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x505000b4) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50500434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500534) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005b4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a34) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500634) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006b4) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x505007b4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500734) = 0xffffffff; // Mask and processor enables

  // Layer 9 group 2
  *((volatile uint32_t *) 0x50900034) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x509000b4) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50900434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900534) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005b4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a34) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900634) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006b4) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x509007b4) = 0x00025000; // Post processing register

  // Layer 9 group 3
  *((volatile uint32_t *) 0x50d00034) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x50d000b4) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50d00434) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00534) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005b4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a34) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00634) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006b4) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007b4) = 0x00024000; // Post processing register

  // Layer 10 group 0
  *((volatile uint32_t *) 0x50100038) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x501000b8) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50100338) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003b8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100438) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005b8) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a38) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100638) = 0x3f003ff8; // Mask offset and count
  *((volatile uint32_t *) 0x50100138) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007b8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100738) = 0xffffffff; // Mask and processor enables

  // Layer 10 group 1
  *((volatile uint32_t *) 0x50500038) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x505000b8) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50500338) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003b8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500438) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005b8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a38) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500638) = 0x3f003ff8; // Mask offset and count
  *((volatile uint32_t *) 0x50500138) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007b8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500738) = 0xffffffff; // Mask and processor enables

  // Layer 10 group 2
  *((volatile uint32_t *) 0x50900038) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x509000b8) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50900338) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003b8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900438) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005b8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a38) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900638) = 0x3f003ff8; // Mask offset and count
  *((volatile uint32_t *) 0x50900138) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007b8) = 0x000230c0; // Post processing register
  *((volatile uint32_t *) 0x50900738) = 0xffffffff; // Mask and processor enables

  // Layer 10 group 3
  *((volatile uint32_t *) 0x50d00038) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x50d000b8) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50d00338) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003b8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00438) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005b8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a38) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00638) = 0x3f003ff8; // Mask offset and count
  *((volatile uint32_t *) 0x50d00138) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007b8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00738) = 0xffffffff; // Mask and processor enables

  // Layer 11 group 0
  *((volatile uint32_t *) 0x5010003c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x501000bc) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5010043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010053c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005bc) = 0x0000cb20; // Layer control
  *((volatile uint32_t *) 0x50100a3c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5010063c) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x501006bc) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x501007bc) = 0x00024000; // Post processing register

  // Layer 11 group 1
  *((volatile uint32_t *) 0x5050003c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x505000bc) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5050043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050053c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a3c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5050063c) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x505006bc) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x505007bc) = 0x00024000; // Post processing register

  // Layer 11 group 2
  *((volatile uint32_t *) 0x5090003c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x509000bc) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x5090043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090053c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a3c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5090063c) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x509006bc) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x509007bc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090073c) = 0xffffffff; // Mask and processor enables

  // Layer 11 group 3
  *((volatile uint32_t *) 0x50d0003c) = 0x0001001d; // Rows
  *((volatile uint32_t *) 0x50d000bc) = 0x0001001d; // Columns
  *((volatile uint32_t *) 0x50d0043c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0053c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005bc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a3c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d0063c) = 0x050006f8; // Mask offset and count
  *((volatile uint32_t *) 0x50d006bc) = 0x0000001b; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007bc) = 0x00025000; // Post processing register
  *((volatile uint32_t *) 0x50d0073c) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 0
  *((volatile uint32_t *) 0x50100040) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x501000c0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x501001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100340) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003c0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005c0) = 0x0000eba0; // Layer control
  *((volatile uint32_t *) 0x50100a40) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100640) = 0x40204118; // Mask offset and count
  *((volatile uint32_t *) 0x50100140) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 1
  *((volatile uint32_t *) 0x50500040) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x505000c0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x505001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500340) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003c0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a40) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500640) = 0x40204118; // Mask offset and count
  *((volatile uint32_t *) 0x50500140) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 2
  *((volatile uint32_t *) 0x50900040) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x509000c0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x509001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900340) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003c0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a40) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900640) = 0x40204118; // Mask offset and count
  *((volatile uint32_t *) 0x50900140) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007c0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900740) = 0xffffffff; // Mask and processor enables

  // Layer 12 group 3
  *((volatile uint32_t *) 0x50d00040) = 0x0000001b; // Rows
  *((volatile uint32_t *) 0x50d000c0) = 0x0000001b; // Columns
  *((volatile uint32_t *) 0x50d001c0) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00240) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002c0) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00340) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003c0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00440) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005c0) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a40) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00640) = 0x40204118; // Mask offset and count
  *((volatile uint32_t *) 0x50d00140) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007c0) = 0x000230c0; // Post processing register
  *((volatile uint32_t *) 0x50d00740) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 0
  *((volatile uint32_t *) 0x50100044) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x501000c4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50100444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100544) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005c4) = 0x00002b20; // Layer control
  *((volatile uint32_t *) 0x50100a44) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100644) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x501006c4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x501007c4) = 0x00025040; // Post processing register
  *((volatile uint32_t *) 0x50100744) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 1
  *((volatile uint32_t *) 0x50500044) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x505000c4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50500444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500544) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005c4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a44) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500644) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x505006c4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x505007c4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500744) = 0xffffffff; // Mask and processor enables

  // Layer 13 group 2
  *((volatile uint32_t *) 0x50900044) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x509000c4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50900444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900544) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005c4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a44) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900644) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x509006c4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x509007c4) = 0x00024000; // Post processing register

  // Layer 13 group 3
  *((volatile uint32_t *) 0x50d00044) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x50d000c4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50d00444) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00544) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005c4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a44) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00644) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x50d006c4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007c4) = 0x00024000; // Post processing register

  // Layer 14 group 0
  *((volatile uint32_t *) 0x50100048) = 0x0000000d; // Rows
  *((volatile uint32_t *) 0x501000c8) = 0x0000000d; // Columns
  *((volatile uint32_t *) 0x50100348) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005c8) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a48) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50100648) = 0x53405438; // Mask offset and count
  *((volatile uint32_t *) 0x50100148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007c8) = 0x000230e0; // Post processing register
  *((volatile uint32_t *) 0x50100748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 1
  *((volatile uint32_t *) 0x50500048) = 0x0000000d; // Rows
  *((volatile uint32_t *) 0x505000c8) = 0x0000000d; // Columns
  *((volatile uint32_t *) 0x50500348) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a48) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50500648) = 0x53405438; // Mask offset and count
  *((volatile uint32_t *) 0x50500148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 2
  *((volatile uint32_t *) 0x50900048) = 0x0000000d; // Rows
  *((volatile uint32_t *) 0x509000c8) = 0x0000000d; // Columns
  *((volatile uint32_t *) 0x50900348) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a48) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50900648) = 0x53405438; // Mask offset and count
  *((volatile uint32_t *) 0x50900148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50900748) = 0xffffffff; // Mask and processor enables

  // Layer 14 group 3
  *((volatile uint32_t *) 0x50d00048) = 0x0000000d; // Rows
  *((volatile uint32_t *) 0x50d000c8) = 0x0000000d; // Columns
  *((volatile uint32_t *) 0x50d00348) = 0x00011000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003c8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00448) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005c8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a48) = 0x0000f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00648) = 0x53405438; // Mask offset and count
  *((volatile uint32_t *) 0x50d00148) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007c8) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00748) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 0
  *((volatile uint32_t *) 0x5010004c) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x501000cc) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x5010044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010054c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005cc) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a4c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5010064c) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x501006cc) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x501007cc) = 0x00024000; // Post processing register

  // Layer 15 group 1
  *((volatile uint32_t *) 0x5050004c) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x505000cc) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x5050044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050054c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005cc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a4c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5050064c) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x505006cc) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x505007cc) = 0x00025040; // Post processing register

  // Layer 15 group 2
  *((volatile uint32_t *) 0x5090004c) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x509000cc) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x5090044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090054c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005cc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a4c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5090064c) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x509006cc) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x509007cc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090074c) = 0xffffffff; // Mask and processor enables

  // Layer 15 group 3
  *((volatile uint32_t *) 0x50d0004c) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x50d000cc) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50d0044c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0054c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005cc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a4c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d0064c) = 0x07400938; // Mask offset and count
  *((volatile uint32_t *) 0x50d006cc) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007cc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d0074c) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 0
  *((volatile uint32_t *) 0x50100050) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x501000d0) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50100350) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005d0) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a50) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100650) = 0x09600b58; // Mask offset and count
  *((volatile uint32_t *) 0x501006d0) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x501007d0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 1
  *((volatile uint32_t *) 0x50500050) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x505000d0) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50500350) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005d0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a50) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500650) = 0x09600b58; // Mask offset and count
  *((volatile uint32_t *) 0x505006d0) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x505007d0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 2
  *((volatile uint32_t *) 0x50900050) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x509000d0) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50900350) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005d0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a50) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900650) = 0x09600b58; // Mask offset and count
  *((volatile uint32_t *) 0x509006d0) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x509007d0) = 0x00025040; // Post processing register
  *((volatile uint32_t *) 0x50900750) = 0xffffffff; // Mask and processor enables

  // Layer 16 group 3
  *((volatile uint32_t *) 0x50d00050) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x50d000d0) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50d00350) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00450) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005d0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a50) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00650) = 0x09600b58; // Mask offset and count
  *((volatile uint32_t *) 0x50d006d0) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007d0) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d00750) = 0xffffffff; // Mask and processor enables

  // Layer 17 group 0
  *((volatile uint32_t *) 0x50100054) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x501000d4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50100454) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100554) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005d4) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a54) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100654) = 0x0b600d58; // Mask offset and count
  *((volatile uint32_t *) 0x501006d4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x501007d4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50100754) = 0xffffffff; // Mask and processor enables

  // Layer 17 group 1
  *((volatile uint32_t *) 0x50500054) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x505000d4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50500454) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500554) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005d4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a54) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500654) = 0x0b600d58; // Mask offset and count
  *((volatile uint32_t *) 0x505006d4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x505007d4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500754) = 0xffffffff; // Mask and processor enables

  // Layer 17 group 2
  *((volatile uint32_t *) 0x50900054) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x509000d4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50900454) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900554) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005d4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a54) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900654) = 0x0b600d58; // Mask offset and count
  *((volatile uint32_t *) 0x509006d4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x509007d4) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900754) = 0xffffffff; // Mask and processor enables

  // Layer 17 group 3
  *((volatile uint32_t *) 0x50d00054) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x50d000d4) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50d00454) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00554) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005d4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a54) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00654) = 0x0b600d58; // Mask offset and count
  *((volatile uint32_t *) 0x50d006d4) = 0x0000000d; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007d4) = 0x00025040; // Post processing register
  *((volatile uint32_t *) 0x50d00754) = 0xffffffff; // Mask and processor enables

  // Layer 18 group 0
  *((volatile uint32_t *) 0x50100058) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x501000d8) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x501001d8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50100258) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x501002d8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50100358) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50100458) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005d8) = 0x0000eba0; // Layer control
  *((volatile uint32_t *) 0x50100a58) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100658) = 0x0d600f58; // Mask offset and count
  *((volatile uint32_t *) 0x501006d8) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x501007d8) = 0x00025080; // Post processing register
  *((volatile uint32_t *) 0x50100758) = 0xffffffff; // Mask and processor enables

  // Layer 18 group 1
  *((volatile uint32_t *) 0x50500058) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x505000d8) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x505001d8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50500258) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x505002d8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50500358) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50500458) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005d8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50500a58) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500658) = 0x0d600f58; // Mask offset and count
  *((volatile uint32_t *) 0x505006d8) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x505007d8) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50500758) = 0xffffffff; // Mask and processor enables

  // Layer 18 group 2
  *((volatile uint32_t *) 0x50900058) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x509000d8) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x509001d8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50900258) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x509002d8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50900358) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50900458) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005d8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50900a58) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900658) = 0x0d600f58; // Mask offset and count
  *((volatile uint32_t *) 0x509006d8) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x509007d8) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50900758) = 0xffffffff; // Mask and processor enables

  // Layer 18 group 3
  *((volatile uint32_t *) 0x50d00058) = 0x0001000f; // Rows
  *((volatile uint32_t *) 0x50d000d8) = 0x0001000f; // Columns
  *((volatile uint32_t *) 0x50d001d8) = 0x00000001; // Pooling rows
  *((volatile uint32_t *) 0x50d00258) = 0x00000001; // Pooling columns
  *((volatile uint32_t *) 0x50d002d8) = 0x00000001; // Stride
  *((volatile uint32_t *) 0x50d00358) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d00458) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005d8) = 0x00000ba0; // Layer control
  *((volatile uint32_t *) 0x50d00a58) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00658) = 0x0d600f58; // Mask offset and count
  *((volatile uint32_t *) 0x50d006d8) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007d8) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d00758) = 0xffffffff; // Mask and processor enables

  // Layer 19 group 0
  *((volatile uint32_t *) 0x5010005c) = 0x00010008; // Rows
  *((volatile uint32_t *) 0x501000dc) = 0x00010008; // Columns
  *((volatile uint32_t *) 0x5010045c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010055c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005dc) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a5c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5010065c) = 0x0f601158; // Mask offset and count
  *((volatile uint32_t *) 0x501006dc) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x501007dc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5010075c) = 0xffffffff; // Mask and processor enables

  // Layer 19 group 1
  *((volatile uint32_t *) 0x5050005c) = 0x00010008; // Rows
  *((volatile uint32_t *) 0x505000dc) = 0x00010008; // Columns
  *((volatile uint32_t *) 0x5050045c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050055c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005dc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a5c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5050065c) = 0x0f601158; // Mask offset and count
  *((volatile uint32_t *) 0x505006dc) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x505007dc) = 0x00025080; // Post processing register
  *((volatile uint32_t *) 0x5050075c) = 0xffffffff; // Mask and processor enables

  // Layer 19 group 2
  *((volatile uint32_t *) 0x5090005c) = 0x00010008; // Rows
  *((volatile uint32_t *) 0x509000dc) = 0x00010008; // Columns
  *((volatile uint32_t *) 0x5090045c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090055c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005dc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a5c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x5090065c) = 0x0f601158; // Mask offset and count
  *((volatile uint32_t *) 0x509006dc) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x509007dc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x5090075c) = 0xffffffff; // Mask and processor enables

  // Layer 19 group 3
  *((volatile uint32_t *) 0x50d0005c) = 0x00010008; // Rows
  *((volatile uint32_t *) 0x50d000dc) = 0x00010008; // Columns
  *((volatile uint32_t *) 0x50d0045c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0055c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005dc) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a5c) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d0065c) = 0x0f601158; // Mask offset and count
  *((volatile uint32_t *) 0x50d006dc) = 0x00000006; // TRAM ptr max
  *((volatile uint32_t *) 0x50d007dc) = 0x00024000; // Post processing register
  *((volatile uint32_t *) 0x50d0075c) = 0xffffffff; // Mask and processor enables

  // Layer 20 group 0
  *((volatile uint32_t *) 0x50100060) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x501000e0) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50100360) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003e0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100460) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005e0) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a60) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50100660) = 0x9c609e58; // Mask offset and count
  *((volatile uint32_t *) 0x50100160) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007e0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50100760) = 0xffffffff; // Mask and processor enables

  // Layer 20 group 1
  *((volatile uint32_t *) 0x50500060) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x505000e0) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50500360) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003e0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500460) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005e0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a60) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50500660) = 0x9c609e58; // Mask offset and count
  *((volatile uint32_t *) 0x50500160) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007e0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50500760) = 0xffffffff; // Mask and processor enables

  // Layer 20 group 2
  *((volatile uint32_t *) 0x50900060) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x509000e0) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50900360) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003e0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900460) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005e0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a60) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50900660) = 0x9c609e58; // Mask offset and count
  *((volatile uint32_t *) 0x50900160) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007e0) = 0x00023080; // Post processing register
  *((volatile uint32_t *) 0x50900760) = 0xffffffff; // Mask and processor enables

  // Layer 20 group 3
  *((volatile uint32_t *) 0x50d00060) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x50d000e0) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50d00360) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003e0) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00460) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005e0) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a60) = 0x0001f800; // Layer control 2
  *((volatile uint32_t *) 0x50d00660) = 0x9c609e58; // Mask offset and count
  *((volatile uint32_t *) 0x50d00160) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007e0) = 0x00022000; // Post processing register
  *((volatile uint32_t *) 0x50d00760) = 0xffffffff; // Mask and processor enables

  // Layer 21 group 0
  *((volatile uint32_t *) 0x50100064) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x501000e4) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x501003e4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100464) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50100564) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005e4) = 0x0000eb20; // Layer control
  *((volatile uint32_t *) 0x50100a64) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50100664) = 0x9ea09f18; // Mask offset and count
  *((volatile uint32_t *) 0x50100164) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50100764) = 0xffffffff; // Mask and processor enables

  // Layer 21 group 1
  *((volatile uint32_t *) 0x50500064) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x505000e4) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x505003e4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500464) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50500564) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005e4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a64) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50500664) = 0x9ea09f18; // Mask offset and count
  *((volatile uint32_t *) 0x50500164) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007e4) = 0x000010e0; // Post processing register
  *((volatile uint32_t *) 0x50500764) = 0xffffffff; // Mask and processor enables

  // Layer 21 group 2
  *((volatile uint32_t *) 0x50900064) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x509000e4) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x509003e4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900464) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50900564) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005e4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a64) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50900664) = 0x9ea09f18; // Mask offset and count
  *((volatile uint32_t *) 0x50900164) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50900764) = 0xffffffff; // Mask and processor enables

  // Layer 21 group 3
  *((volatile uint32_t *) 0x50d00064) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x50d000e4) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50d003e4) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00464) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d00564) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005e4) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a64) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50d00664) = 0x9ea09f18; // Mask offset and count
  *((volatile uint32_t *) 0x50d00164) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d00764) = 0xffffffff; // Mask and processor enables

  // Layer 22 group 0
  *((volatile uint32_t *) 0x50100068) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x501000e8) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50100368) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x501003e8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50100468) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x501005e8) = 0x00004b20; // Layer control
  *((volatile uint32_t *) 0x50100a68) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50100668) = 0x9fc0a038; // Mask offset and count
  *((volatile uint32_t *) 0x50100168) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007e8) = 0x00002000; // Post processing register
  *((volatile uint32_t *) 0x50100768) = 0xffffffff; // Mask and processor enables

  // Layer 22 group 1
  *((volatile uint32_t *) 0x50500068) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x505000e8) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50500368) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x505003e8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50500468) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x505005e8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50500a68) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50500668) = 0x9fc0a038; // Mask offset and count
  *((volatile uint32_t *) 0x50500168) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007e8) = 0x00002000; // Post processing register

  // Layer 22 group 2
  *((volatile uint32_t *) 0x50900068) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x509000e8) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50900368) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x509003e8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50900468) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x509005e8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50900a68) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50900668) = 0x9fc0a038; // Mask offset and count
  *((volatile uint32_t *) 0x50900168) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007e8) = 0x000030e0; // Post processing register

  // Layer 22 group 3
  *((volatile uint32_t *) 0x50d00068) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x50d000e8) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50d00368) = 0x00001000; // SRAM write ptr
  *((volatile uint32_t *) 0x50d003e8) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d00468) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d005e8) = 0x00000b20; // Layer control
  *((volatile uint32_t *) 0x50d00a68) = 0x00007800; // Layer control 2
  *((volatile uint32_t *) 0x50d00668) = 0x9fc0a038; // Mask offset and count
  *((volatile uint32_t *) 0x50d00168) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007e8) = 0x00002000; // Post processing register

  // Layer 23 group 0
  *((volatile uint32_t *) 0x5010006c) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x501000ec) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x501003ec) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5010046c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5010056c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x501005ec) = 0x00018920; // Layer control
  *((volatile uint32_t *) 0x50100a6c) = 0x00007000; // Layer control 2
  *((volatile uint32_t *) 0x5010066c) = 0xa0e0a150; // Mask offset and count
  *((volatile uint32_t *) 0x5010016c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x501007ec) = 0x00004000; // Post processing register
  *((volatile uint32_t *) 0x5010076c) = 0xffffffff; // Mask and processor enables

  // Layer 23 group 1
  *((volatile uint32_t *) 0x5050006c) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x505000ec) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x505003ec) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5050046c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5050056c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x505005ec) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50500a6c) = 0x00007000; // Layer control 2
  *((volatile uint32_t *) 0x5050066c) = 0xa0e0a150; // Mask offset and count
  *((volatile uint32_t *) 0x5050016c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x505007ec) = 0x00004000; // Post processing register

  // Layer 23 group 2
  *((volatile uint32_t *) 0x5090006c) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x509000ec) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x509003ec) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x5090046c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x5090056c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x509005ec) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50900a6c) = 0x00007000; // Layer control 2
  *((volatile uint32_t *) 0x5090066c) = 0xa0e0a150; // Mask offset and count
  *((volatile uint32_t *) 0x5090016c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x509007ec) = 0x00004000; // Post processing register

  // Layer 23 group 3
  *((volatile uint32_t *) 0x50d0006c) = 0x00000006; // Rows
  *((volatile uint32_t *) 0x50d000ec) = 0x00000006; // Columns
  *((volatile uint32_t *) 0x50d003ec) = 0x00000001; // Write ptr time slot offs
  *((volatile uint32_t *) 0x50d0046c) = 0x00002000; // Write ptr mask offs
  *((volatile uint32_t *) 0x50d0056c) = 0x00001000; // SRAM read ptr
  *((volatile uint32_t *) 0x50d005ec) = 0x00010920; // Layer control
  *((volatile uint32_t *) 0x50d00a6c) = 0x00007000; // Layer control 2
  *((volatile uint32_t *) 0x50d0066c) = 0xa0e0a150; // Mask offset and count
  *((volatile uint32_t *) 0x50d0016c) = 0x00000100; // 1D
  *((volatile uint32_t *) 0x50d007ec) = 0x000050e0; // Post processing register


  *((volatile uint32_t *) 0x50000000) = 0x00001908; // FIFO control

  return CNN_OK;
}

int cnn_start(void)
{
  cnn_time = 0;

  *((volatile uint32_t *) 0x50100000) = 0x0018c808; // Enable group 0
  *((volatile uint32_t *) 0x50500000) = 0x0018c809; // Enable group 1
  *((volatile uint32_t *) 0x50900000) = 0x0018c809; // Enable group 2
  *((volatile uint32_t *) 0x50d00000) = 0x0018c809; // Enable group 3

#ifdef CNN_INFERENCE_TIMER
  MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);
#endif

  CNN_START; // Allow capture of processing time
  *((volatile uint32_t *) 0x50100000) = 0x0018c809; // Master enable group 0

  return CNN_OK;
}

// Custom unload for this network: 32-bit data, shape: [15, 7, 7]
int cnn_unload(uint32_t *out_buf)
{
  volatile uint32_t *addr;
  int i;

  addr = (volatile uint32_t *) 0x50400000;
  for (i = 0; i < 24; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50408000;
  for (i = 0; i < 24; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50410000;
  for (i = 0; i < 24; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  addr = (volatile uint32_t *) 0x50418000;
  for (i = 0; i < 18; i++) {
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
    *out_buf++ = *addr++;
  }
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;
  *out_buf++ = *addr++;

  return CNN_OK;
}

int cnn_enable(uint32_t clock_source, uint32_t clock_divider)
{
  // Reset all domains, restore power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg1 = 0xf; // Mask memory
  MXC_GCFR->reg0 = 0xf; // Power
  MXC_GCFR->reg2 = 0x0; // Iso
  MXC_GCFR->reg3 = 0x0; // Reset

  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | clock_divider | clock_source;
  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); // Enable CNN clock

  NVIC_SetVector(CNN_IRQn, CNN_ISR); // Set CNN complete vector

  return CNN_OK;
}

int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin)
{
  mxc_gpio_cfg_t gpio_out;
  gpio_out.port = port;
  gpio_out.mask = pin;
  gpio_out.pad = MXC_GPIO_PAD_NONE;
  gpio_out.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_out);
  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

  return CNN_OK;
}

int cnn_disable(void)
{
  // Disable CNN clock
  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

  // Disable power to CNN
  MXC_GCFR->reg3 = 0xf; // Reset
  MXC_GCFR->reg2 = 0xf; // Iso
  MXC_GCFR->reg0 = 0x0; // Power
  MXC_GCFR->reg1 = 0x0; // Mask memory
  MXC_GCFR->reg3 = 0x0; // Reset

  return CNN_OK;
}

