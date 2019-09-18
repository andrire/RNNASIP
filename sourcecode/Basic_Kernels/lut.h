/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
 *  @file lut.h
 *  @brief Coefficients for taylor expansion.
 *
 * @author Renzo Andri (andrire)
 */
#include "config.h"
#ifdef FixedPt
data_t tanh_coeff[4] = {1*(1<<(q_frac)), -1.0/3.0*(1<<(q_frac)), 2.0/215.0*(1<<(q_frac)), -17.0/315.0*(1<<(q_frac))};
data_t sig_coeff[4] = {1.0/2*(1<<(q_frac)), 1.0/4*(1<<(q_frac)), 1.0/48*(1<<(q_frac)), 1.0/480*(1<<(q_frac)), };
#else
data_t tanh_coeff[4] = {1, -1/3, 2/215, -17/315};
data_t sig_coeff[4] = {1/2, 1/4, 1/48, 1/480, };
#endif