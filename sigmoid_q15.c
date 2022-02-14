//
// Created by Yi-Wei Chen on 5/12/21.
//

#include "mxc.h"
#include "utils_faceid.h"
#include "sigmoid_lut.h"

static const q15_t lut[] = SIGMOID_LUT;

q15_t q_div(q15_t a, q15_t b)
{
    /* pre-multiply by the base (Upscale to Q32 so that the result will be in Q16 format) */
    q31_t temp = (q31_t)a << 15;
    /* Rounding: mid values are rounded up (down for negative values). */
    /* OR compare most significant bits */
    if ((temp >= 0 && b >= 0) || (temp < 0 && b < 0)) {
        temp += (b >> 1);    /* OR shift 1 bit i.e. temp += (b >> 1); */
    } else {
        temp -= (b >> 1);    /* OR shift 1 bit i.e. temp -= (b >> 1); */
    }
    return (q15_t)(temp / b);
}

// saturate to range of int16_t
q15_t sat16(q31_t x)
{
    if (x > 32767) return 32767;
    else if (x < -32768) return -32768;
    else return (q15_t)x;
}


q15_t q_mul(q15_t a, q15_t b)
{
    q15_t result;
    q31_t temp;
    q31_t K = (1 << (15 - 1));
    temp = (q31_t)a * (q31_t)b; // result type is operand's type
    // Rounding; mid values are rounded up
    temp += K;
    // Correct by dividing by base and saturate result
    result = sat16(temp >> 15);

    return result;
}

/**
 * @addtogroup Sigmoid
 * @{
 */

/**
 * @brief Q17.14 fixed point sigmoid function, returns Q17.14
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimension
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 * @details
 *
 *  Here, we use lookup table and interpolation to implement sigmoid function
 *  min_x = -8 (Q17: -131072), max_x = 8 - unit (Q17: 131072 - unit)
 *  unit = 8 / (2 ** (NUM_LUT_BITS - 1))
 *  min_y = 0.000335350137902423 (Q17: 5), max_y = 0.999642968177795 (Q17: 16378)
 *
 */

q15_t sigmoid(q31_t in)
{
    q15_t out;
    q15_t y1, y2, slope, diff;
    uint16_t idx;
    uint16_t num_entries = 0x1 << NUM_LUT_BITS;
    uint16_t offset = num_entries >> 1;
    uint8_t shift = 18 - NUM_LUT_BITS;
    uint16_t unit = 0x1 << shift;
    q31_t upper = 131072 - unit;
    q31_t lower = -131072;
    if (in >= upper)
        return lut[num_entries - 1];
    else if (in <= lower)
        return lut[0];
    else {
        idx = (in >> shift) + offset;
        y1 = lut[idx];
        y2 = lut[idx + 1];
        slope = q_div(y2 - y1, unit * 2);
        diff = in & (unit - 1);
        out = y1 + q_mul(slope, diff * 2);
//        out = y1 + ((y2 - y1) >> (shift - 1)) * (in & unit - 1);
        return out;
    }
}
