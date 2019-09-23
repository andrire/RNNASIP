/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
 *  @file basicKernel.c
 *  @brief C Implementation for basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP
 * 
 *  C implemenentation implementing several levels of opitmizations for the RISC-Y extension and the tzscale extension
 * 
 * @author Renzo Andri (andrire)
 */

#ifndef INCLHEADER
#define INCLHEADER
#include <stdio.h>
// #include <stdlib.h>
#include <config.h>
//#include <math.h>
#include "basicKernel.h"
#include "lut.h" // coefficients for taylor expansion

#endif

/** \brief Length (in time) of RNN Sequence */
int rnn_seqSize=1; 
/** \brief Length (in time) of LSTM Sequence */
int lstm_seqSize=1;


/** \brief Piecewise Linear Approximation of tangent hyperbolic and sigmoid */
const int lut_numelements = 16;
const int lb_lut_numelements = 4;

/** \brief Coefficients of piecewise Linear Approximation of tangent hyperbolic and sigmoid */
const short lut_Tanh_m[16] = {4021, 3563, 2835, 2070, 1418, 929, 592, 370, 228, 140, 86, 52, 32, 19, 12, 7};
const int lut_Tanh_q[16] = {17060, 512067, 2012407, 4361003, 7021506, 9510743, 11575189, 13158594, 14311861, 15123015, 15679911, 16055709, 16306104, 16471340, 16579558, 16650000};
const short lut_sig_m[16] = {1019, 988, 930, 850, 758, 660, 563, 472, 391, 319, 258, 207, 165, 131, 104, 82};
const int lut_sig_q[16] = {8389671, 8423495, 8544906, 8789991, 9169470, 9670607, 10264318, 10914030, 11583389, 12241371, 12864661, 13437943, 13952921, 14406803, 14800713, 15138308};


#ifndef ASIP
/** @brief Tells the PULP-SDK to initialize or reset performance counters
 */
void startPerf () {
  // rt_perf_reset(&perf);
  // numFunctionCalls++;
  // rt_perf_start(&perf);

}
/** @brief Tells the PULP-SDK to stop the performance counters
 */
void endPerf () {
  // rt_perf_stop(&perf);
}
#endif

/** @brief Sigmoid Activation Function
 *
 *  @param value input varialbe
 *  @return sigmoid of the input variable
 */
inline data_t sig(data_t value) {
  data_t a = value;
  unsigned int lutsize = 16;
  unsigned int value1 = 4096;
  unsigned int value0p999 = 4095;
  int m;
  int q;
  int q_signed;
  unsigned short sign = a<0 ? 1 : 0;
  int tmp;
  int mac_result;
  int mac_result_signed;
  data_t abs_a;
  if(sign==0x1) {
    abs_a = -a;
  } else {
    abs_a = (a);
  }

  tmp = abs_a>>(13-3); 
  if(tmp>=lutsize) {
    return (sign)?(data_t)0: (data_t)value1;
  } else {
    m = lut_sig_m[tmp];
    q =lut_sig_q[tmp];
    mac_result = (m*abs_a+q)>>12;
    mac_result_signed = (sign==1)? ~mac_result : mac_result;
    if(sign==1) {
       return  (value0p999+(mac_result_signed)); // 1-(mx+q)=4096+(~mac_result+1)=4095+(~mac_result)
    } else {
      return mac_result_signed;
    }
  }
}

/** @brief Tangent Hyperbolic
 *
 *  @param value input variable
 *  @return tangent hypberbolic of the input variable
 */
  inline data_t Tanh(data_t value) {
    data_t x = value;
    unsigned int lutsize = 16;
    unsigned int value1 = 4096;
    unsigned int value0p999 = 4095;
    int m;
    int q;
    int q_signed;
    unsigned short sign = (x>>31) & 0x1;
    int id;
    int mac_result;
    int mac_result_signed;
    int abs_x;
    if(sign==0x1) {
      abs_x = -x;
    } else {
      abs_x = x;
    }
    id = abs_x>>(13-3); // get index of LUT
    if(id>=lutsize) {
       return (sign==0x1)?(int)-value1: (int)value1;
    } else {
      m = lut_Tanh_m[id];
      q =lut_Tanh_q[id];
      mac_result = (m*abs_x+q)>>12;
      mac_result_signed = (sign==1)? ~mac_result : mac_result;
      return mac_result_signed;
    }
}


/** @brief Runs a neural network
 *
 *  Iterates through all the layers while passing the intermediate FM with a double 
 *  buffering approach
 *
 *  @param network Array of concecutive layers of the current neural network
 *  @param depth Number of Layers (aka array size)
 *  @param inFeatures Input Feature Map
 *  @param buffer Buffer to store intermediate results
 *  @return Output Feature Map
 */
data_t * NOINLINE inferNetwork(
  struct layer * network, 
  int depth, 
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ buffer)
{
 //printf("delete, just for test1");
  data_t * in = inFeatures;
  data_t * out = &buffer[BUFFER_SIZE2];

  short toFIRST = False;
  for(int i = 0; i < depth; i++)
  {
      //printf("delete, just for test 2");
    struct layer lay = network[i];
    if(lay.type == LINEAR)
    {

#ifdef DEBUG_LSTM
      printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
      printf("Inputs in: ");
      PrintTensor(lay.attributes[LAY_LIN_IN], in);
#endif
      // startPerf();
      LinearLayer(lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT], True, 
        lay.parameters[LAY_LIN_WEIGHTS],
        lay.parameters[LAY_LIN_BIAS], 
        // Input and Output Features
        in, //inFeatures,
        out); // outFeatures 
      // endPerf();
#ifdef DEBUG_LSTM
      printf("Results in: ");
      PrintTensor(lay.attributes[LAY_LIN_OUT], out);
#endif
      toFIRST ^= 1; 

      // switch buffer (double buffering)
      if(toFIRST) 
      {
        in  = &buffer[BUFFER_SIZE2];
        out = &buffer[0];
      }
      else 
      {
        in  = &buffer[0];
        out = &buffer[BUFFER_SIZE2];
      }
    }
    else if (lay.type == LSTM)
    {
#ifdef DEBUG_LSTM
      printf("LSTM (%i, %i)\n", lay.attributes[LAY_LSTM_IN], lay.attributes[LAY_LSTM_HID]);
      printf("Inputs in: ");
      PrintTensor(lay.attributes[LAY_LSTM_IN], in);
#endif
      
      // startPerf();
      int numHidden = lay.attributes[LAY_LSTM_HID];
      LSTMLayer (
        // Layer Attributes
        lay.attributes[LAY_LSTM_IN], numHidden,
        // Layer Parameters
        lay.parameters[LSTM_WGHT_IH],
        lay.parameters[LSTM_WGHT_HH],
        lay.parameters[LSTM_BIAS_IH],
        lay.parameters[LSTM_BIAS_HH],
        // Input and Output Features
        in,
        lay.parameters[LSTM_H],
        // Hidden Features
        lay.parameters[LSTM_C],
        // intermediate nodes
        out + 1*numHidden*1, //f
        out + 2*numHidden*1, //i
        out + 3*numHidden*1, //g
        out //o
        );
      in  =  (data_t *)lay.parameters[LSTM_H];
        // endPerf();

#ifdef DEBUG_LSTM
      printf("Results at: ");
      PrintTensor(lay.attributes[LAY_LSTM_HID], lay.parameters[LSTM_H]);
#endif
    }
    else if(lay.type == Conv2d)
    {
    #ifdef DEBUG_LSTM
      printf("Conv2D (%i->%i, ker=%i^2, h*w=%i*%i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT], lay.attributes[LAY_CONV_KER],lay.attributes[LAY_CONV_H],lay.attributes[LAY_CONV_W]);
      printf("Inputs in: ");
      PrintTensor(lay.attributes[LAY_LIN_IN]*lay.attributes[LAY_CONV_H]*lay.attributes[LAY_CONV_W], in);
#endif
      Conv2dLayer(&lay, lay.attributes[LAY_CONV_H], lay.attributes[LAY_CONV_W], in, out);
#ifdef DEBUG_LSTM
      printf("Conv2D (%i->%i, ker=%i^2, h*w=%i*%i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT], lay.attributes[LAY_CONV_KER],lay.attributes[LAY_CONV_H],lay.attributes[LAY_CONV_W]);
      printf("Results in: ");
      PrintTensor(lay.attributes[LAY_CONV_OUT]*lay.attributes[LAY_CONV_H]*lay.attributes[LAY_CONV_W], out);
#endif
      toFIRST ^= 1; 

      // switch buffers
      if(toFIRST) 
      {
        in  = &buffer[BUFFER_SIZE2];
        out = &buffer[0];
      }
      else 
      {
        in  = &buffer[0];
        out = &buffer[BUFFER_SIZE2];
      }

      // printf("inconv=%i, ", in[0]);


    }
// #endif
    else
    {
      printf("\033[91mERROR: not a valid layer\033[0m\n");
    }


    
  }
return &in[0]; // return address of output feature map
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _     _                         _                           ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |   (_)_ __   ___  __ _ _ __  | |    __ _ _   _  ___ _ __  ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |   | | '_ \ / _ \/ _` | '__| | |   / _` | | | |/ _ \ '__| ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |___| | | | |  __/ (_| | |    | |__| (_| | |_| |  __/ |    ////////////////////////////////////////////////////////////////////////////////////////////////////
// |_____|_|_| |_|\___|\__,_|_|    |_____\__,_|\__, |\___|_|    ////////////////////////////////////////////////////////////////////////////////////////////////////
//                                             |___/            ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 //////////////////////////////////////////////////////////////////////////////////////////////   
 //        _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _   // 
 //|      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  |   // 
 //|_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__|   // 
 //                                                                                          //
 //////////////////////////////////////////////////////////////////////////////////////////////
#if defined FMOUTTILING && !defined(ASIP) && defined MANUALLOOPUNFOLDING && defined VLIWEXT // obv vliw
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with the custom VLIW instructions for load and MAC
 *  Supports the following configurations:
 *  INPUTFMTILING false/true with input tile size 2
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10,12,14 (odd are currently not
 *                 supported as the SPR would need to be switched)
 *  FixedPt and SIMD and MANUALLOOPUNFOLDING only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
void NOINLINE LinearLayer (
        // Layer Attributes
  int inFeaturesSize, int outFeaturesSize,
  short hasBias,
        // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
        // Input and Output Features
  data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
{
  PROFILING_LINEAR_START

  int inFeaturesSizeP2 = inFeaturesSize/2;
  #ifdef FMINTILING
  int inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
  #else
  int inFeaturesSizeP4 = inFeaturesSizeP2;   // no input FM tiling
  #endif

  #if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
  #elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
  #else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
  #endif

  data_t  * bias_ptr   = bias;
  v2s     * weight_ptr = (v2s*)weight;
  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;

  // register definition for manual loop unfolding
  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  // register null
  register int x0 asm("x0");

  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;

  // Tile with largest tileOption
  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;

    // Select Tile Size
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
      // Manual loop unfolding
      // Inititalize accumulation registers with bias and shift accordingly
          #if OUTPUTBUFFER > 2
      temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 3
      temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 4
      temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 5
      temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 6
      temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 7
      temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 8
      temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 9
      temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 10
      temp8 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 11
      temp9 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 12
      temp10 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 13
      temp11 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 14
      temp12 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
          #endif
      temp13 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]<<(q_fraqP1);
      temp14 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]<<(q_fraqP1);

      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];

          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t) (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
               #if OUTPUTBUFFER > 2
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 3
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 4
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 5
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 6
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 7
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 10
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 11
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 12
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 13
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 14
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
               #endif
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );
// do it twice for FMINTILING
  #ifdef FMINTILING
               #if OUTPUTBUFFER > 2
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 3
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 4
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 5
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 6
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 7
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 10
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 11
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 12
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 13
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 14
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp2) );
               #endif
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
               #if OUTPUTBUFFER > 2
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 3
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 4
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 5
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 6
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 7
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               #endif
               #if OUTPUTBUFFER > 9
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 9
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 10
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 11
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 12
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 13
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 14
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
               #endif
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );
             }

  #endif

// Store the final results back to the memory
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
                #if OUTPUTBUFFER > 2
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 3
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 4
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 5
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 6
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 7
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 8
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 9
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 10
             outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = temp8>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 11
             outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = temp9>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 12
             outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = temp10>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 13
             outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = temp11>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 14
             outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = temp12>>(q_fraqP1);
                #endif
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = temp13>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = temp14>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

           }
           break;
     #endif
     #if OUTPUTBUFFER > 8
           case 8:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
            temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
            temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
            temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
            temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

          // }
          // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
            addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
            addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
            addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
            addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload 2nd weight

          in_addr = (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];

              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
  #ifdef FMINTILING
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
        // }
           }
           break;
     #endif
     #if OUTPUTBUFFER > 4
           case 4:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);

          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t)(((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );
  #ifdef FMINTILING
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
               // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }
           }
           break;
     #endif
           case 2:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
          // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
          // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            
            // int o_rel;
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
            // }
          }
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
          outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

        }
        break;
        case 1:
        for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
        {
          temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], temp0);
          }
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
        }
        break;
      }

  // move pointers for next iteration
      bias_ptr                = &bias_ptr[outFeatureTiles*outFeaturesPerTile];
      weight_ptr              = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(outFeatureTiles*outFeaturesPerTile))];
      outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
      outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
      if (outFeaturesSize_remain==0) break;
    }


    PROFILING_LINEAR_END
  }
// new implementation for output FM tiling and input FM tiling without VLIW!
#elif defined FMOUTTILING && !defined(ASIP) && defined MANUALLOOPUNFOLDING && !defined VLIWEXT 
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with the custom VLIW instructions for load and MAC
 *  Supports the following configurations:
 *  INPUTFMTILING false/true with input tile size 2
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10,12,14 (odd are currently not
 *                 supported as the SPR would need to be switched)
 *  FixedPt and SIMD and MANUALLOOPUNFOLDING only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
void NOINLINE LinearLayer (
        // Layer Attributes
  int inFeaturesSize, int outFeaturesSize,
  short hasBias,
        // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
        // Input and Output Features
  data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
{
  PROFILING_LINEAR_START
  int inFeaturesSizeP2 = inFeaturesSize/2;
  #ifdef FMINTILING
  int inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
  #else
  int inFeaturesSizeP4 = inFeaturesSizeP2;   // no input FM tiling
  #endif

  #if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
  #elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
  #else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
  #endif

  data_t  * bias_ptr   = bias;
  v2s     * weight_ptr = (v2s*)weight;
  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;

  // register definition for manual loop unfolding
  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  // register null
  register int x0 asm("x0");

  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;

  // Tile with largest tileOption
  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;

    // Select Tile Size
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
      // Manual loop unfolding
      // Inititalize accumulation registers with bias and shift accordingly
          #if OUTPUTBUFFER > 2
      temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 3
      temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 4
      temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 5
      temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 6
      temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 7
      temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 8
      temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 9
      temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 10
      temp8 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 11
      temp9 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 12
      temp10 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 13
      temp11 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 14
      temp12 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
          #endif
      temp13 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]<<(q_fraqP1);
      temp14 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]<<(q_fraqP1);

      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];

          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t)(((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
               #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp);
// do it twice for FMINTILING
  #ifdef FMINTILING
                #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp2);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp2);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp);
             }

  #endif

// Store the final results back to the memory
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
                #if OUTPUTBUFFER > 2
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 3
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 4
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 5
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 6
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 7
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 8
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 9
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 10
             outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = temp8>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 11
             outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = temp9>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 12
             outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = temp10>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 13
             outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = temp11>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 14
             outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = temp12>>(q_fraqP1);
                #endif
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = temp13>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = temp14>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

           }
           break;
     #endif
     #if OUTPUTBUFFER > 8
           case 8:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
            temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
            temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
            temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
            temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

          // }
          // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
            addr0 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
            addr2 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
            addr3 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
            addr4 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
            addr5 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
            addr6 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
            addr7 = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload 2nd weight

          in_addr = (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];

              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp) ;
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp) ;
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp) ;
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp) ;
  #ifdef FMINTILING
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp2);
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp2);
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp2);
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp2);
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp2);
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp2);
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp2);
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp) ;
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp) ;
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp) ;
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp) ;
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
        // }
           }
           break;
     #endif
     #if OUTPUTBUFFER > 4
           case 4:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);

          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t) (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
  #ifdef FMINTILING
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp2);
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp2);
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp2);
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
               // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }
           }
           break;
     #endif
           case 2:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
          // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
          // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            
            // int o_rel;
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp) ;
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
            // }
          }
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
          outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

        }
        break;
        case 1:
        for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
        {
          temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            SDOTP_GENERIC(temp0, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], inF_temp) ;
          }
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
        }
        break;
      }

  // move pointers for next iteration
      bias_ptr                = &bias_ptr[outFeatureTiles*outFeaturesPerTile];
      weight_ptr              = &((v2s*)weight_ptr)[(inFeaturesSizeP2*(outFeatureTiles*outFeaturesPerTile))];
      outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
      outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
      if (outFeaturesSize_remain==0) break;
    }


    PROFILING_LINEAR_END
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// _______ _______ _____  _____       _  _  _ _____ _______ _     _      _______ _______       _____  _     _ _______      _______ _____        _____ __   _  ///
// |_____| |______   |   |_____]      |  |  |   |      |    |_____|      |______ |  |  |      |     | |     |    |            |      |   |        |   | \  |  ///
// |     | ______| __|__ |            |__|__| __|__    |    |     |      |       |  |  |      |_____| |_____|    |            |    __|__ |_____ __|__ |  \_|  ///
//                                                                                                                                                            ///
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#elif defined ASIP && defined FMOUTTILING // LinearLayer Implementation for the ASIP tool and 
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with for the ASIP designer (extended tzscale)
 *  Supports the following configurations:
 *  INPUTFMTILING false (no input FM tiling support as not needed)
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10
 *  MANUALLOOPUNFOLDING false/true
 *  FixedPt and SIMD and  only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
  void NOINLINE LinearLayer (
        // Layer Attributes
    int inFeaturesSize, int outFeaturesSize,
    short hasBias,
        // Layer Parameters
    data_t * __restrict__ weight,
    data_t * __restrict__ bias,
        // Input and Output Features
    data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
  {
    PROFILING_LINEAR_START

    int inFeaturesSizeP2 = inFeaturesSize/2;

//  _  __    _                                                               _   
// / | \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_ 
// | |  | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __|
// | |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_ 
// |_|  | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|__ \___/ \__,_|\__|
//     /_/                |_|                                |__|              

//----------------------------------------------------------------
// 1a) Tile Output Channels
//----------------------------------------------------------------

    int tileOptions[] = {10,8,4,2, 1};
    int outFeaturesPerTile = 1;
    for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
     if(outFeaturesSize % tileOptions[i] == 0) {
      outFeaturesPerTile = tileOptions[i];
      break;
    }
  }

  int outFeatureTiles = outFeaturesSize/outFeaturesPerTile;


  switch(outFeaturesPerTile) {
   case 10:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    int32_t register_attribute ra2;int32_t register_attribute rb2;int32_t register_attribute rc2;int32_t register_attribute rd2;
    int32_t register_attribute rc3;int32_t register_attribute rd3;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    ra2 = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
    rb2 = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
    rc2 = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
    rd2 = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
    rc3 = (int32_t)bias[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
    rd3 = (int32_t)bias[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);


    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];                                                          

               ra=ext_dotp_(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i], ra); // lwinc x23, 0(x20)
               rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);      // lwinc x25, 0(x11); sdotp x7, x23, x25 
               rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);      // lwinc x24, 0(x11); sdotp x7, x23, x24 
               rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);      // lwinc xA, 0(xB); sdotp xC, x23, xB
               ra2 = ra2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rb2 = rb2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rc2 = rc2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rd2 = rd2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rc3 = rc3 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+8)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rd3 = rd3 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+9)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
             }
             outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+4)] = ra2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+5)] = rb2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+6)] = rc2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+7)] = rd2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+8)] = rc3>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+9)] = rd3>>(q_fraqP1);


   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;

   case 8:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    int32_t register_attribute ra2;int32_t register_attribute rb2;int32_t register_attribute rc2;int32_t register_attribute rd2;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    ra2 = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
    rb2 = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
    rc2 = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
    rd2 = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
     rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
     rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
     rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
     ra2 = ra2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4)) + i]);
     rb2 = rb2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5)) + i]);
     rc2 = rc2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6)) + i]);
     rd2 = rd2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+4)] = ra2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+5)] = rb2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+6)] = rc2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+7)] = rd2>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;
   case 4:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
     rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
     rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
     rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;
 case 2: // HOWTO duplicate and comment out not needed lines
 for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
      int32_t register_attribute ra;int32_t register_attribute rb;//int32_t register_attribute rc;int32_t register_attribute rd;
      ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
      // rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
      // rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) { 
       v2s inF_temp = ((v2s*)inFeatures)[i];
       ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
       rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
               // rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
               // rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
      } // }
       // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      // outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      // outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   case 1: // HOWTO duplicate and comment out not needed lines
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);

    for(int i=0; i<inFeaturesSizeP2; i++) chess_loop_range(1,) { 

     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
   } 
   outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;

}// switch outFeaturesPerTile


}

 ///////////////////////////////////////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______      _____ _______  _____          //
 // |     \ |______ |______ |_____| |     | |         |           |   |  |  | |_____] |       //
 // |_____/ |______ |       |     | |_____| |_____    |         __|__ |  |  | |       |_____  //
 //                                                                                           //
 ///////////////////////////////////////////////////////////////////////////////////////////////
#else // any other case
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer (standard implementation)
 *  Supports the following configurations:
 *  => (PULP-RISCY && !VLIWEXT || ASIP && !FMOUTTILING) && !FMINTILING
 *  => FMOUTINILING false (no input FM tiling support as not needed)
 *  => FMOUTTILING false/true with output tile sizes 1,2,4,8,10
 *  => FixedPt, SIMD || FLOAT
 *  => MANUALLOOPUNFOLDING not supported for non-SIMD
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */void NOINLINE LinearLayer (
        // Layer Attributes
int inFeaturesSize, int outFeaturesSize,
short hasBias,
        // Layer Parameters
data_t * __restrict__ weight,
data_t * __restrict__ bias,
        // Input and Output Features
data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
 {
  PROFILING_LINEAR_START


// TODO: currently it is not supporte to halve odd number of feature map size with SIMD
// either fill it up with 0 or set the weight to 0
// If this needs to be supported, round down the iterations and add the very last contribution separately.
  int inFeaturesSizeP2 = (inFeaturesSize)/2;

//  _  __    _                                                               _   
// / | \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_ 
// | |  | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __|
// | |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_ 
// |_|  | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|__ \___/ \__,_|\__|
//     /_/                |_|                                |__|              

//----------------------------------------------------------------
// 1a) Tile Output Channels
//----------------------------------------------------------------
#ifdef FMOUTTILING

 const int outFeaturesPerTile = Min(outFeaturesSize, OUTPUTBUFFER); // Find maximum possible feature tile size
 int outFeatureTiles = (int)(outFeaturesSize-1)/outFeaturesPerTile+1; // output channels per tile (round it up)

 // printf("outFeaturesSize=%i, outFeaturesPerTile=%i, outFeatureTiles=%i", outFeaturesSize, outFeaturesPerTile, outFeatureTiles);
 for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
 {
//----------------------------------------------------------------
// 1b) Do not tile Output Channels
//----------------------------------------------------------------
#else // not FMOUTTILING
   const int outFeaturesPerTile = 1;  
   const int o_rel = 0;
   int o_tile=0;
   for (int o=0; o< outFeaturesSize; o++) 
   {
    o_tile = o;
#endif // FMOUTTILING
// ===============================================================

//  ____   __    _       _ _     _                       
// |___ \  \ \  (_)_ __ (_) |_  | |_ ___ _ __ ___  _ __  
//   __) |  | | | | '_ \| | __| | __/ _ \ '_ ` _ \| '_ \ 
//  / __/   | | | | | | | | |_  | ||  __/ | | | | | |_) |
// |_____|  | | |_|_| |_|_|\__|  \__\___|_| |_| |_| .__/ 
//         /_/                                    |_|    
#ifdef FixedPt
//----------------------------------------------------------------
// 2c) Fixed-Pt, not ASIP
//----------------------------------------------------------------
    
// manual loop unfolding to full utilize the registers
#ifdef MANUALLOOPUNFOLDING
    register int32_t ra,rb,rc,rd,re,rf,rg,rh, ri, rj, rk,rl, rm,rn,ro;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
# if OUTPUTBUFFER>1
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>2
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>3
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>4
    re = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>5
    rf = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>6
    rg = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>7
    rh = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>8
    ri = (int32_t)bias[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>9
    rj = (int32_t)bias[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>10
    rk = (int32_t)bias[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>11
    rl = (int32_t)bias[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>12
    rm = (int32_t)bias[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>13
    rn = (int32_t)bias[o_tile*outFeaturesPerTile+13]<<(q_fraqP1);
# endif
# if OUTPUTBUFFER>14
    ro = (int32_t)bias[o_tile*outFeaturesPerTile+14]<<(q_fraqP1);
# endif

#else 
    int32_t  temp[OUTPUTBUFFER];
    for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      temp[o_rel] = (int32_t)bias[o_tile*outFeaturesPerTile+o_rel]<<(q_fraqP1);
    }
#endif


//----------------------------------------------------------------
// 2d) Floating point
//----------------------------------------------------------------     
#else // not FixedPt
    data_t temp[OUTPUTBUFFER];
    temp = bias[o];
#endif // not FixedPt
// ===============================================================   


//  _____  __    _                                                    _       
// |___ /  \ \  | | ___   ___  _ __     _____   _____ _ __    ___    (_)_ __  
//   |_ \   | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   | | '_ \ 
//  ___) |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__    | | | | |
// |____/   | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|___|_|_| |_|
//         /_/                |_|                               |_____|       

//----------------------------------------------------------------
// 3a) Iterate over N/2 pair (SIMD) of input channels where N=inFeaturesSize
//----------------------------------------------------------------
#ifdef SIMD
    for(int i=0; i<inFeaturesSizeP2; i++) {
     v2s inF_temp = ((v2s*)inFeatures)[i];
#else
//----------------------------------------------------------------
// 3b) Iterate over (inFeaturesSize) inputChannels
//----------------------------------------------------------------
     for(int i=0; i<inFeaturesSize; i++) {
#endif
// ===============================================================         



//  _  _    __    _                                                               _     _   _ _      
// | || |   \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_  | |_(_) | ___ 
// | || |_   | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __| | __| | |/ _ \
// |__   _|  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_  | |_| | |  __/
//    |_|    | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|___\___/ \__,_|\__|  \__|_|_|\___|
//          /_/                |_|                               |_____|                             
#if defined FMOUTTILING && !defined MANUALLOOPUNFOLDING
//----------------------------------------------------------------
// 4a) Loop over tile for PULP
//----------------------------------------------------------------

// NO MANUAL LOOP UNROLL
for (int o_rel=0; o_rel < outFeaturesPerTile; o_rel++) { // chess_loop_count(4) chess_flatten_loop //chess_loop_range(1,4)
// No MANUAL LOOP UNROLL

#endif
// =============================================================== 
//////////////////////////////////////////////////////////////////
//  ____   __    ___                         __  __    _    ____ 
// | ___|  \ \  |_ _|_ __  _ __   ___ _ __  |  \/  |  / \  / ___|
// |___ \   | |  | || '_ \| '_ \ / _ \ '__| | |\/| | / _ \| |    
//  ___) |  | |  | || | | | | | |  __/ |    | |  | |/ ___ \ |___ 
// |____/   | | |___|_| |_|_| |_|\___|_|    |_|  |_/_/   \_\____|
//         /_/                                                   
#ifdef FixedPt
#  ifdef SIMD
//----------------------------------------------------------------
// 5b) SIMD on PULP with intriniscs
//----------------------------------------------------------------
#     if !defined MANUALLOOPUNFOLDING
  temp[o_rel] = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
#     else    
  ra = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i], ra);
#       if OUTPUTBUFFER>1
  rb = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i], rb);
#       endif
#       if OUTPUTBUFFER>2
  rc = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i], rc);
#       endif
#       if OUTPUTBUFFER>3
  rd = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i], rd);
#       endif
#       if OUTPUTBUFFER>4
  re = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4)) + i], re);
#       endif
#       if OUTPUTBUFFER>5
  rf = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5)) + i], rf);
#       endif
#       if OUTPUTBUFFER>6
  rg = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6)) + i], rg);
#       endif
#       if OUTPUTBUFFER>7
  rh = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7)) + i], rh);
#       endif
#       if OUTPUTBUFFER>8
  ri = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+8)) + i], ri);
#       endif
#       if OUTPUTBUFFER>9
  rj = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+9)) + i], rj);
#       endif
#       if OUTPUTBUFFER>10
  rk = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10)) + i], rk);
#       endif
#       if OUTPUTBUFFER>11
  rl = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11)) + i], rl);
#       endif
#       if OUTPUTBUFFER>12
  rm = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12)) + i], rm);
#       endif
#       if OUTPUTBUFFER>13
  rn = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+13)) + i], rn);
#       endif
#       if OUTPUTBUFFER>14
  ro = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+14)) + i], ro);
#       endif


#     endif
#  else // SIMD
//----------------------------------------------------------------
// 5d) Fixed-Point, not SIMD
//----------------------------------------------------------------
  temp[o_rel] += (int32_t)((inFeatures[i]*weight[inFeaturesSize*(o_tile*outFeaturesPerTile+o_rel) + i]));
#     if defined MANUALLOOPUNFOLDING
  printf("WARN: for simplicity currently no MANUAL loop unfolding for non-SIMD fixed-point");
#     endif
#  endif
#else
//----------------------------------------------------------------
// 5e) Floating Point
//----------------------------------------------------------------
  temp[o_rel] += inFeatures[i]*weight[inFeaturesSize*(o_tile*outFeaturesPerTile+o_rel) + i];
#     if defined MANUALLOOPUNFOLDING
  printf("WARN: for simplicity currently no MANUAL loop unfolding for float");
#     endif
#endif
}

#if defined FMOUTTILING && !defined MANUALLOOPUNFOLDING
}
#endif //FMOUTTILING

// NO MANUAL LOOP UNFOLD
#if !defined MANUALLOOPUNFOLDING || !defined FixedPt
for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
# ifdef FixedPt
  outFeatures[(o_tile*outFeaturesPerTile+o_rel)] = temp[o_rel]>>(q_fraqP1);
# else
  outFeatures[(o_tile*outFeaturesPerTile+o_rel)] = temp[o_rel];
#endif
}
#else
outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
#       if OUTPUTBUFFER>1
outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>2
outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>3
outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>4
outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>5
outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>6
outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>7
outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>8
outFeatures[(o_tile*outFeaturesPerTile+8)] = ri>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>9
outFeatures[(o_tile*outFeaturesPerTile+9)] = rj>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>10
outFeatures[(o_tile*outFeaturesPerTile+10)] = rk>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>11
outFeatures[(o_tile*outFeaturesPerTile+11)] = rl>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>12
outFeatures[(o_tile*outFeaturesPerTile+12)] = rm>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>13
outFeatures[(o_tile*outFeaturesPerTile+13)] = rn>>(q_fraqP1);
#       endif
#       if OUTPUTBUFFER>14
outFeatures[(o_tile*outFeaturesPerTile+14)] = ro>>(q_fraqP1);
#       endif

#endif



}

PROFILING_LINEAR_END
}
#endif



/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////   ____                 ____     _ _                           /////////////////////////
/////  / ___|___  _ ____   _|___ \ __| | |    __ _ _   _  ___ _ __  /////////////////////////
///// | |   / _ \| '_ \ \ / / __) / _` | |   / _` | | | |/ _ \ '__| /////////////////////////
///// | |__| (_) | | | \ V / / __/ (_| | |__| (_| | |_| |  __/ |    /////////////////////////
/////  \____\___/|_| |_|\_/ |_____\__,_|_____\__,_|\__, |\___|_|    /////////////////////////
/////                                              |___/            /////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//         _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _ //
// |      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  | //
// |_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__| //
//                                                                                         //
/////////////////////////////////////////////////////////////////////////////////////////////                                                                                      
#if defined(VLIWEXT) // RISCY implementation with the lw-sdopt-VLIW
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  input channels need to be multiple of 4 or 2 (with/without FMINTILING)
 *  Supporte configurations:
 *  > VLIWEXT 
 *  > SIMD only
 *  > FMIN and FMOUTILING
 *  > MANUALLOOPUNFOLDING true
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {

   #if OUTPUTBUFFER > 8
 int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
   #elif OUTPUTBUFFER > 4
 int tileOptions[] = {OUTPUTBUFFER,4,2,1};
   #elif OUTPUTBUFFER > 2
 int tileOptions[] = {OUTPUTBUFFER,2,1};
   #elif OUTPUTBUFFER > 1
 int tileOptions[] = {2,1};
   #else
 int tileOptions[] = {1};
   #endif
 int outFeaturesPerTile = 1;
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
   data_t * bias_ptr   = _layer->parameters[CONV_BIAS];
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   data_t  * outFeatures_ptr = outFeatures;

   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;

   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   #ifdef FMINTILING
   c_in_max = _layer->attributes[LAY_CONV_IN]/4;
   #else
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   #endif
   register int x0 asm("x0");
   register int32_t temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
   register uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7;
   // const int outFeaturesPerTile = 2;
   unsigned int param_kh_base = 0;
   int outFeatureTiles;
   int outFeaturesSize_remain = _layer->attributes[LAY_CONV_OUT];

  // Tile with largest tileOption
   for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
       for(int h_out =0; h_out < h_im_out; h_out++) 
       {
         for(int w_out=0;w_out<w_im_out; w_out++)
         {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders

            #if OUTPUTBUFFER > 2
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 3
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 4
           temp2 = bias_ptr[outFeaturesPerTile*c_out+2] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 5
           temp3 = bias_ptr[outFeaturesPerTile*c_out+3] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 6
           temp4 = bias_ptr[outFeaturesPerTile*c_out+4] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 7
           temp5 = bias_ptr[outFeaturesPerTile*c_out+5] << q_fraqP1;
            #endif
           temp6 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr2  = (uint32_t) &((v2s*)param_simd)[param_id_base+2*output_channel_offset];
                                             addr3  = (uint32_t) &((v2s*)param_simd)[param_id_base+3*output_channel_offset];
                                             addr4  = (uint32_t) &((v2s*)param_simd)[param_id_base+4*output_channel_offset];
                                             addr5  = (uint32_t) &((v2s*)param_simd)[param_id_base+5*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-2)];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-1)];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                 #if OUTPUTBUFFER > 2
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr2) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 3
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 4
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 5
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 6
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 7
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
                 #endif

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
// do it twice for FMINTILING
#ifdef FMINTILING
                 #if OUTPUTBUFFER > 2
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr2) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 3
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 4
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 5
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 6
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 7
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
                 #endif

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );


#endif
                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

                  #if OUTPUTBUFFER > 2
            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 3
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 4
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp2) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 5
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp3) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 6
            outFeatures_ptr[(outFeaturesPerTile*c_out+4)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp4) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 7
            outFeatures_ptr[(outFeaturesPerTile*c_out+5)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp5) >> q_fraqP1);
                  #endif

            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
     #endif
     #if OUTPUTBUFFER > 4
    case 4:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 =  bias_ptr[(outFeaturesPerTile*c_out+1)] << q_fraqP1;
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
           temp6 = bias_ptr[outFeaturesPerTile*c_out+2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+3]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*2];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*3];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),   "+r" (addr2) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
                    #ifdef FMINTILING
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),   "+r" (addr2) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
                    #endif

                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);

            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
#endif
    case 2:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*1];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr6) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr7) : "r" (x0) ); // preload first weight

                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];


                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr6) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr7) : "r" (inF_temp) );
                    #ifdef FMINTILING
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr6) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr7) : "r" (inF_temp2) );
                    #endif

                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
    case 1:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
          // temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                 // addr6  = &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                #ifdef FMINTILING
                for(int i=0; i <  2*c_in_max;i++) // i=c_in                              
                #else
                for(int i=0; i <  c_in_max;i++) // i=c_in                             
                #endif
                 
                 {
                  temp6  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                   param_simd[param_id_base + i], temp6);
                }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
  }


      // move pointers for next iteration
  bias_ptr                = &bias_ptr[outFeaturesPerTile*outFeatureTiles];
      // param_simd              = &((v2s*)param_simd)[output_channel_offset*(outFeatureTiles*outFeaturesPerTile)];
  outFeatures_ptr         = &outFeatures_ptr[outFeaturesPerTile*outFeatureTiles*h_im_out*w_im_out];
  outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

  if (outFeaturesSize_remain==0) break;
}
return 0;
}
#elif defined(FMOUTTILING) // RISCY implementation with the lw-sdopt-VLIW
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  input channels need to be multiple of 4 or 2 (with/without FMINTILING)
 *  Supporte configurations:
 *  > VLIWEXT 
 *  > SIMD only
 *  > FMIN and FMOUTILING
 *  > MANUALLOOPUNFOLDING true
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {

   #if OUTPUTBUFFER > 8
 int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
   #elif OUTPUTBUFFER > 4
 int tileOptions[] = {OUTPUTBUFFER,4,2,1};
   #elif OUTPUTBUFFER > 2
 int tileOptions[] = {OUTPUTBUFFER,2,1};
   #elif OUTPUTBUFFER > 1
 int tileOptions[] = {2,1};
   #else
 int tileOptions[] = {1};
   #endif
 int outFeaturesPerTile = 1;
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
   data_t * bias_ptr   = _layer->parameters[CONV_BIAS];
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   data_t  * outFeatures_ptr = outFeatures;

   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;

   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   // #ifdef FMINTILING
   // c_in_max = _layer->attributes[LAY_CONV_IN]/4;
   // #else
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   // #endif
   
   register_attribute int32_t temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
   register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7;
   // const int outFeaturesPerTile = 2;
   unsigned int param_kh_base = 0;
   int outFeatureTiles;
   int outFeaturesSize_remain = _layer->attributes[LAY_CONV_OUT];

  // Tile with largest tileOption
   for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
       for(int h_out =0; h_out < h_im_out; h_out++) 
       {
         for(int w_out=0;w_out<w_im_out; w_out++)
         {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders

            #if OUTPUTBUFFER > 2
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 3
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 4
           temp2 = bias_ptr[outFeaturesPerTile*c_out+2] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 5
           temp3 = bias_ptr[outFeaturesPerTile*c_out+3] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 6
           temp4 = bias_ptr[outFeaturesPerTile*c_out+4] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 7
           temp5 = bias_ptr[outFeaturesPerTile*c_out+5] << q_fraqP1;
            #endif
           temp6 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr2  = (uint32_t) &((v2s*)param_simd)[param_id_base+2*output_channel_offset];
                                             addr3  = (uint32_t) &((v2s*)param_simd)[param_id_base+3*output_channel_offset];
                                             addr4  = (uint32_t) &((v2s*)param_simd)[param_id_base+4*output_channel_offset];
                                             addr5  = (uint32_t) &((v2s*)param_simd)[param_id_base+5*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-2)];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-1)];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                 #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
              
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
// do it twice for FMINTILING
  // #ifdef FMINTILING
  //               #if OUTPUTBUFFER > 2
  //             SDOTP_GENERIC(temp, ((v2s*)addr0)[2*i+1], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 3
  //             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 4
  //             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 5
  //             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 6
  //             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 7
  //             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
  //              #endif
              
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp2);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp2);
  // #endif
            // }
            }
  #ifdef FMINTILING
          if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
             }

  #endif
                  
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

                  #if OUTPUTBUFFER > 2
            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 3
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 4
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp2) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 5
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp3) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 6
            outFeatures_ptr[(outFeaturesPerTile*c_out+4)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp4) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 7
            outFeatures_ptr[(outFeaturesPerTile*c_out+5)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp5) >> q_fraqP1);
                  #endif

            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
     #endif
     #if OUTPUTBUFFER > 4
    case 4:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 =  bias_ptr[(outFeaturesPerTile*c_out+1)] << q_fraqP1;
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
           temp6 = bias_ptr[outFeaturesPerTile*c_out+2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+3]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*2];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*3];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];

                    SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
                    SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
                    SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
                    SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
                    // #ifdef FMINTILING
                    //   SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
                    // #endif

                  }
  // #ifdef FMINTILING
  //         if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
  //              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
  //              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
  //              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  //              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
  //              // MANUAL loop unfolding
  //              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
  //              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
  //             SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
  //             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
  //            }

  // #endif
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);

            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
#endif
    case 2:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*1];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr6) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr7) : "r" (x0) ); // preload first weight

                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];


                    SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
                    SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
                    // #ifdef FMINTILING
                    // SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
                    // SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
                    // #endif

                  }
  //             #ifdef FMINTILING
  //         if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
  //              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
  //              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
  //              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  //              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
  //              // MANUAL loop unfolding
  //              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
  //              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
  //            }

  // #endif
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
    case 1:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
          // temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                 // addr6  = &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                #ifdef FMINTILING
                for(int i=0; i <  2*c_in_max;i++) // i=c_in                              
                #else
                for(int i=0; i <  c_in_max;i++) // i=c_in                             
                #endif
                 
                 {
                  temp6  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                   param_simd[param_id_base + i], temp6);
                }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
  }


      // move pointers for next iteration
  bias_ptr                = &bias_ptr[outFeaturesPerTile*outFeatureTiles];
      // param_simd              = &((v2s*)param_simd)[output_channel_offset*(outFeatureTiles*outFeaturesPerTile)];
  outFeatures_ptr         = &outFeatures_ptr[outFeaturesPerTile*outFeatureTiles*h_im_out*w_im_out];
  outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

  if (outFeaturesSize_remain==0) break;
}
return 0;
}


 /////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______  //
 // |     \ |______ |______ |_____| |     | |         |     //
 // |_____/ |______ |       |     | |_____| |_____    |     //
 //                                                         //
 /////////////////////////////////////////////////////////////
  #else // no vliw
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  Implements the 2D convolution layer (standard implementation)
 *  Supports the following configurations:
 *  => No VLIW Support (see special implementation)
 *  => FMOUTINILING false (not support, not benefitial)
 *  => FMOUTTILING false/true
 *  => FixedPt, SIMD || FLOAT
 *  => MANUALLOOPUNFOLDING not implemented
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
// Layer Attributes
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {
  //          printf("delete, just for test 4");
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
#ifdef SIMD
   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;
#else
   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN];
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN];
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN];
#endif
   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   #ifdef FixedPt
   #ifdef SIMD
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   #endif
   #endif
   const int outFeaturesPerTile = 1;
   for(int c_out = 0; c_out < _layer->attributes[LAY_CONV_OUT]/outFeaturesPerTile; c_out++) {
    for(int h_out =0; h_out < h_im_out; h_out++) 
    {
     for(int w_out=0;w_out<w_im_out; w_out++)
     {

               int kh_slide_start = Max(-h_out, -h_ker_half);           // Handle borders
               int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
#ifdef SIMD
               int32_t temp = 0;

               int32_t temp1 = 0;
#else
               int32_t  temp = 0;
#endif
               unsigned int param_kh_base = outFeaturesPerTile*c_out * output_channel_offset;
               for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
               {
                  int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
                  int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders


                  unsigned int param_kw_base = param_kh_base \
                  + (kh+h_ker_half) * kernel_H_offset;
                  unsigned int feat_kw_base  = (h_out+kh)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
                  for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                  {
                   int param_id_base = param_kw_base \
                                                +(kw+w_ker_half)*kernel_W_offset; // filter tap
                                                int feat_id_base  = feat_kw_base \
                                                +(w_out+kw)* _layer->attributes[LAY_CONV_IN]/2;
                     for(int i=0; i <  c_in_max;i++) // i=c_in
                     {
                        unsigned int param_id; // = c_out * output_channel_offset \
                                                + c_in  \
                                                + (kh+h_ker_half) * kernel_H_offset\
                                                 +(kw+w_ker_half)*kernel_W_offset; // filter tap
                        unsigned int feat_id; //  = (h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]+(w_out+kw)* _layer->attributes[LAY_CONV_IN] + c_in;
                        //printf("%i\n", param_id);

#ifdef FixedPt
#ifdef SIMD
                        // param_id = c_out * output_channel_offset \
                        //                         + c_in  \
                        //                         + (kh+h_ker_half) * kernel_H_offset\
                        //                         +(kw+w_ker_half)*kernel_W_offset; // filter tap
                        // feat_id  = (h_out+kh)*w_im_out*_layer->attributes[LAY_CONV_IN]/2\
                        //            +(w_out+kw)* _layer->attributes[LAY_CONV_IN]/2 + c_in;
                        // param_id = param_id_base + c_in;
                        // feat_id  = feat_id_base + c_in;

#ifndef ASIP
                        temp = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                         param_simd[param_id_base + i], temp);
#else // ASIP
                        temp = temp + ((v2s*)inFeatures)[feat_id_base + i] * param_simd[param_id_base + i];
#endif // ASIP
#else // not SIMD
                        temp +=  ((_layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                          + i  \
                          + (kh+h_ker_half) * kernel_H_offset\
                          +(kw+w_ker_half)*kernel_W_offset)] \
                        * inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                          +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]));
                         // printf("temp=%x+=%x*%x\n", temp, _layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                                                + i  \
                                                + (kh+h_ker_half) * kernel_H_offset\
                                                 +(kw+w_ker_half)*kernel_W_offset)], inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                                  +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]);
#endif // SIMD
#else // FixedPt
                                  temp +=  ((_layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                                    + i  \
                                    + (kh+h_ker_half) * kernel_H_offset\
                                    +(kw+w_ker_half)*kernel_W_offset)] \
                                  * inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                                  +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]));// >> (q_fraqP1));

#endif // FixedPt

                                }

                              }

                            }




#ifdef SIMD
                            outFeatures[outFeaturesPerTile*c_out*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1)+ _layer->parameters[CONV_BIAS][outFeaturesPerTile*c_out];
               // outFeatures[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1)+ _layer->parameters[CONV_BIAS][(outFeaturesPerTile*c_out+1)];
#else
                            outFeatures[c_out*h_im_out*w_im_out+h_out*w_im_out+w_out] = (temp >> (q_fraqP1))+ _layer->parameters[CONV_BIAS][c_out];
#endif // end SIMD
                          }
                        }
                      }

                      return 0;
                    }
 #endif
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
////  _____               _     _                        ///// 
//// |_   _|_      _____ | |   (_)_ __   ___  __ _ _ __  /////
////   | | \ \ /\ / / _ \| |   | | '_ \ / _ \/ _` | '__| /////
////   | |  \ V  V / (_) | |___| | | | |  __/ (_| | |    /////
////   |_|   \_/\_/ \___/|_____|_|_| |_|\___|\__,_|_|    /////
////                                                     /////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// _______ _______ _____  _____       ___________      _______ _____        _____ __   _  ______
// |_____| |______   |   |_____]      |______|            |      |   |        |   | \  | |  ____
// |     | ______| __|__ |            |______|            |    __|__ |_____ __|__ |  \_| |_____|
//
/////////////////////////////////////////////////////////////// 

/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (ASIP with OutputFMTiling)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  ASIP&SIMD&FMOUTTILING
 *  FMOUTTILING of 8,4,2,1 tile size
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
#ifdef FixedPt
#if defined FixedPt && defined FMOUTTILING && !defined VLIWEXT && defined ASIP
                    void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
                      int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
                      data_t * __restrict__ weight1,
                      data_t * __restrict__ weight2,
                      data_t * __restrict__ bias1,
                      data_t * __restrict__ bias2,
        // Input and Output Features
                      data_t * __restrict__ inFeatures1,
                      data_t * __restrict__ inFeatures2,
                      data_t * __restrict__ outFeatures)
                    {
                      int tileOptions[] = {10,8,4,2, 1};

                      int outFeaturesPerTile = 1;
// find appropriate tiling (TODO: do it like in RISCY)
                      for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
                       if(outFeaturesSize % tileOptions[i] == 0) {
                        outFeaturesPerTile = tileOptions[i];
                        break;
                      }
                    }
                    int outFeatureTiles = outFeaturesSize/outFeaturesPerTile;


                    int inFeaturesSize1P2=inFeaturesSize1/2;
                    int inFeaturesSize2P2=inFeaturesSize2/2;
                    switch(outFeaturesPerTile) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      case 10:
                      for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
                        int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
                        int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
                        ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
                        rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
                        rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
                        rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
                        re = ((int32_t)bias1[o_tile*outFeaturesPerTile+4]+(int32_t)bias2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
                        rf = ((int32_t)bias1[o_tile*outFeaturesPerTile+5]+(int32_t)bias2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
                        rg = ((int32_t)bias1[o_tile*outFeaturesPerTile+6]+(int32_t)bias2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
                        rh = ((int32_t)bias1[o_tile*outFeaturesPerTile+7]+(int32_t)bias2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
                        ri = ((int32_t)bias1[o_tile*outFeaturesPerTile+8]+(int32_t)bias2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
                        rj = ((int32_t)bias1[o_tile*outFeaturesPerTile+9]+(int32_t)bias2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);

                        for(int i=0; i<inFeaturesSize1P2; i++) { 
                          v2s inF_temp = ((v2s*)inFeatures1)[i];         
                          SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
                          SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
                          SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
                          SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);
                          SDOTP_GENERIC(re, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+4)) + i]);
                          SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+5)) + i]);
                          SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+6)) + i]);
                          SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+7)) + i]);
                          SDOTP_GENERIC(ri, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+8)) + i]);
                          SDOTP_GENERIC(rj, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+9)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)

      for(int i=0; i<inFeaturesSize2P2; i++)
      {
        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
        SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
        SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
        SDOTP_GENERIC(re, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+4)) + i]);
        SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+5)) + i]);
        SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+6)) + i]);
        SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+7)) + i]);
        SDOTP_GENERIC(ri, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+8)) + i]);
        SDOTP_GENERIC(rj, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+9)) + i]);

      }

#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      rc = rc>>(q_fraqP1);
      rd = rd>>(q_fraqP1);
      re = re>>(q_fraqP1);
      rf = rf>>(q_fraqP1);
      rg = rg>>(q_fraqP1);
      rh = rh>>(q_fraqP1);
      ri = ri>>(q_fraqP1);
      rj = rj>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd;
        outFeatures[(o_tile*outFeaturesPerTile+4)] = re; 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = rf;
        outFeatures[(o_tile*outFeaturesPerTile+6)] = rg; 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = rh;
        outFeatures[(o_tile*outFeaturesPerTile+8)] = ri; 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = rj; break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_tanh(re); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_tanh(rf);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_tanh(rg); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_tanh(rh);
        outFeatures[(o_tile*outFeaturesPerTile+8)] = generic_tanh(ri); 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = generic_tanh(rj); break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_sig(re); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_sig(rf);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_sig(rg); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_sig(rh);
        outFeatures[(o_tile*outFeaturesPerTile+8)] = generic_sig(ri); 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = generic_sig(rj); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+8)] = ri>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+9)] = rj>>(q_fraqP1); 
#endif
    }
break; // case 10
case 8:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
  int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  re = ((int32_t)bias1[o_tile*outFeaturesPerTile+4]+(int32_t)bias2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  rf = ((int32_t)bias1[o_tile*outFeaturesPerTile+5]+(int32_t)bias2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  rg = ((int32_t)bias1[o_tile*outFeaturesPerTile+6]+(int32_t)bias2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  rh = ((int32_t)bias1[o_tile*outFeaturesPerTile+7]+(int32_t)bias2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);


  for(int i=0; i<inFeaturesSize1P2; i++)
  {

    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
    SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
    SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);
    SDOTP_GENERIC(re, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+4)) + i]);
    SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+5)) + i]);
    SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+6)) + i]);
    SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+7)) + i]);
         } // for(int i=0; i<inFeaturesSizeP2; i++)

         for(int i=0; i<inFeaturesSize2P2; i++)
         {
          v2s inF_temp = ((v2s*)inFeatures2)[i];         
          SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
          SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
          SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
          SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
          SDOTP_GENERIC(re, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+4)) + i]);
          SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+5)) + i]);
          SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+6)) + i]);
          SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+7)) + i]);
        }

#ifdef DOACTONTHEFLY
        ra = ra>>(q_fraqP1);
        rb = rb>>(q_fraqP1);
        rc = rc>>(q_fraqP1);
        rd = rd>>(q_fraqP1);
        re = re>>(q_fraqP1);
        rf = rf>>(q_fraqP1);
        rg = rg>>(q_fraqP1);
        rh = rh>>(q_fraqP1);
        switch(activationFunction) {
          case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
          outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = rd;
          outFeatures[(o_tile*outFeaturesPerTile+4)] = re; 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = rf;
          outFeatures[(o_tile*outFeaturesPerTile+6)] = rg; 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = rh; break;
          case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
          outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd);
          outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_tanh(re); 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_tanh(rf);
          outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_tanh(rg); 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_tanh(rh); break;
          case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
          outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd);
          outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_sig(re); 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_sig(rf);
          outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_sig(rg); 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_sig(rh); break;
        }
#else
        outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
#endif
      }
break; // case 8

case 4:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
  int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);





  for(int i=0; i<inFeaturesSize1P2; i++)
  {
    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
    SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
    SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);

      } // for(int i=0; i<inFeaturesSizeP2; i++)
      
      for(int i=0; i<inFeaturesSize2P2; i++)
      {
        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
        SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
        SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
      }

#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      rc = rc>>(q_fraqP1);
      rd = rd>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd; break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd); break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
#endif
    }
break; // case 4


case 2:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);   
  for(int i=0; i<inFeaturesSize1P2; i++)
  {
    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)

      for(int i=0; i<inFeaturesSize2P2; i++)
      {

        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);      
      }
#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;  break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);  break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);; 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);;
#endif
    }
break; // case 2
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
} // swtich(outFeaturesPerTile)
} // TwoLinearLayersAccumulate

 ///////////////////////////////////////////////////////////////////////////////////////////////
 //         _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _   //
 // |      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  |   //
 // |_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__|   //
 //                                                                                           //
 /////////////////////////////////////////////////////////////////////////////////////////////// 
#elif !defined(ASIP) && defined(SIMD) && defined(VLIWEXT) && defined(FMOUTTILING)
/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (PULP and VLIW implementation)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  VLIW+SIMD
 *  FMOUTTILING false, true
 *  FMINTILING true
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
        // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{


  PROFILING_TWOLINEAR_START

#if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
#elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
#else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
#endif



// todo add loop tiling
  data_t * bias_ptr1   = bias1;

  data_t * bias_ptr2   = bias2;

  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;

  register int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register uint32_t  in_addr;
  register int x0 asm("x0");
  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
  v2s     * weight_ptr; 
  v2s     * weight_ptr1=(v2s*)weight1;
  v2s     * weight_ptr2=(v2s*)weight2;
  int inFeaturesSizeP2, inFeaturesSizeP4;
  data_t     * inFeatures;

  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    if(outFeatureTiles == 0) continue;

    switch(outFeaturesPerTile) {
   #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:

     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
        // printf("o_tile=%i\n", o_tile);
        #if OUTPUTBUFFER > 2
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 3
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 4
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 5
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 6
      temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 7
      temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 8
      temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 9
      temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 10
      temp8 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+8]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 11
      temp9 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+9]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 12
      temp10 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+10]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+10])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 13
      temp11 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+11]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+11])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 14
      temp12 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+12]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+12])<<(q_fraqP1);
        #endif
      temp13 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2])<<(q_fraqP1);
      temp14 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1])<<(q_fraqP1);



      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }



      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];
        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
        in_addr = (uint32_t)(((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );

#ifdef FMINTILING
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp2) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp2) );
#endif
          // }
          }
#ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
                    v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
            
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );

          }

#endif
} // loop for fm1 and fm2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
              #if OUTPUTBUFFER > 2
outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
              #endif
              #if OUTPUTBUFFER > 3
outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              #endif
              #if OUTPUTBUFFER > 4
outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
              #endif
              #if OUTPUTBUFFER > 5
outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              #endif
              #if OUTPUTBUFFER > 6
outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
              #endif
              #if OUTPUTBUFFER > 7
outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
              #endif
              #if OUTPUTBUFFER > 8
outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
              #endif
              #if OUTPUTBUFFER > 9
outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
              #endif
              #if OUTPUTBUFFER > 10
outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = shiftAndAct(temp8, activationFunction);
              #endif
              #if OUTPUTBUFFER > 11
outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = shiftAndAct(temp9, activationFunction);
              #endif
              #if OUTPUTBUFFER > 12
outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = shiftAndAct(temp10, activationFunction);
              #endif
              #if OUTPUTBUFFER > 13
outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = shiftAndAct(temp11, activationFunction);
              #endif
              #if OUTPUTBUFFER > 14
outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = shiftAndAct(temp12, activationFunction);
              #endif
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = shiftAndAct(temp13, activationFunction);
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = shiftAndAct(temp14, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

}
break;
   #endif
   #if OUTPUTBUFFER > 8
case 8:

for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
{

  temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);

        // }
        // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
  for(int turn=0; turn<2; turn++) {
    if(turn==0) {
      weight_ptr = weight_ptr1;
      inFeaturesSizeP2 = inFeaturesSize1/2;
      inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }
      addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
      addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
      addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
      addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
      addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
      addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
      addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
      addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

        in_addr = (((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
#ifdef FMINTILING
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
#endif
          // }
          }
#ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );

          }
#endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
      // }
    }
    break;
   #endif
   #if OUTPUTBUFFER > 4
    case 4:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = weight_ptr2;
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        // }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


       asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
       asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

       in_addr = (uint32_t)(((v2s*)inFeatures));
       for(int i=0; i<inFeaturesSizeP4; i++) {
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );
#ifdef FMINTILING
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr1) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr2) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr3) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr0) : "r" (inF_temp2) );
#endif
          // }
          }
# ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr1) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr0) : "r" (inF_temp) );

          }
# endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }
    }
    break;
#endif
    case 2:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
        // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
        // }

      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = &weight_ptr1[0];
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = &weight_ptr2[0];
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
        for(int i=0; i<inFeaturesSizeP2; i++) {
          v2s inF_temp = ((v2s*)inFeatures)[i];
          
          // int o_rel;
          // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr0) : "r" (inF_temp) );
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
          // }
        }
      } // fm in1 in2 
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
              // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

    }
    break;
    case 1:


    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {
     for(int turn=0; turn<2; turn++) {
      if(turn==0) {
        weight_ptr = &weight_ptr1[0];
        inFeaturesSizeP2 = inFeaturesSize1/2;
        inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = &weight_ptr2[0];
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) {
        v2s inF_temp = ((v2s*)inFeatures)[i];
        temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], temp0);
      }
   } // fm in1 and in2 
   outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
 }
 break;
}
 // updat pointer for next iteration
bias_ptr1                = &bias_ptr1[outFeatureTiles*outFeaturesPerTile];
bias_ptr2                = &bias_ptr2[outFeatureTiles*outFeaturesPerTile];
weight_ptr1              = &weight_ptr1[(inFeaturesSize1/2*(outFeatureTiles*outFeaturesPerTile))];
weight_ptr2              = &weight_ptr2[(inFeaturesSize2/2*(outFeatureTiles*outFeaturesPerTile))];
outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
if (outFeaturesSize_remain==0) break;
}
PROFILING_TWOLINEAR_END
}
#elif !defined(ASIP) && defined(SIMD) && !defined(VLIWEXT) && defined(FMOUTTILING)
/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (PULP and VLIW implementation)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  VLIW+SIMD
 *  FMOUTTILING false, true
 *  FMINTILING true
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
        // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{

  PROFILING_TWOLINEAR_START

#if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
#elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
#else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
#endif



// todo add loop tiling
  data_t * bias_ptr1   = bias1;

  data_t * bias_ptr2   = bias2;

  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;

  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
  v2s     * weight_ptr; 
  v2s     * weight_ptr1=(v2s*)weight1;
  v2s     * weight_ptr2=(v2s*)weight2;
  int inFeaturesSizeP2, inFeaturesSizeP4;
  data_t     * inFeatures;

  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    if(outFeatureTiles == 0) continue;

    switch(outFeaturesPerTile) {
   #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:

     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
        // printf("o_tile=%i\n", o_tile);
        #if OUTPUTBUFFER > 2
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 3
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 4
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 5
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 6
      temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 7
      temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 8
      temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 9
      temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 10
      temp8 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+8]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 11
      temp9 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+9]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 12
      temp10 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+10]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+10])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 13
      temp11 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+11]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+11])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 14
      temp12 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+12]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+12])<<(q_fraqP1);
        #endif
      temp13 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2])<<(q_fraqP1);
      temp14 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1])<<(q_fraqP1);



      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        // NO FMINTILING as this does not give any benefit (to implement uncomment all #ifdef FMINTLING parts and change indexing to 2i and 2i+1)
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      }



      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];
        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
        in_addr = (uint32_t)(((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 3
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 4
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 5
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 6
            SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 7
            SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp) ;
             #endif
             #if OUTPUTBUFFER > 9
            SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 9
            SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 10
            SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 11
            SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 12
            SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 13
            SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 14
            SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
             #endif
            SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp);
            SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp);

// #ifdef FMINTILING
//              #if OUTPUTBUFFER > 2
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 3
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 4
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 5
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 6
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 7
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2) ;
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 10
//             SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 11
//             SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 12
//             SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 13
//             SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 14
//             SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp2);
//              #endif
//             SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp2);
//             SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp2);
// #endif
          // }
          }
// #ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//                     v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
//             // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
            
//             // MANUAL loop unfolding
//             // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
//             // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
//                           #if OUTPUTBUFFER > 2
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 3
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 4
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 5
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 6
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 7
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp) ;
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 10
//             SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 11
//             SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 12
//             SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 13
//             SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 14
//             SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
//              #endif
//             SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp);
//             SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp);

//           }

// #endif
} // loop for fm1 and fm2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
              #if OUTPUTBUFFER > 2
outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
              #endif
              #if OUTPUTBUFFER > 3
outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              #endif
              #if OUTPUTBUFFER > 4
outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
              #endif
              #if OUTPUTBUFFER > 5
outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              #endif
              #if OUTPUTBUFFER > 6
outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
              #endif
              #if OUTPUTBUFFER > 7
outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
              #endif
              #if OUTPUTBUFFER > 8
outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
              #endif
              #if OUTPUTBUFFER > 9
outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
              #endif
              #if OUTPUTBUFFER > 10
outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = shiftAndAct(temp8, activationFunction);
              #endif
              #if OUTPUTBUFFER > 11
outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = shiftAndAct(temp9, activationFunction);
              #endif
              #if OUTPUTBUFFER > 12
outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = shiftAndAct(temp10, activationFunction);
              #endif
              #if OUTPUTBUFFER > 13
outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = shiftAndAct(temp11, activationFunction);
              #endif
              #if OUTPUTBUFFER > 14
outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = shiftAndAct(temp12, activationFunction);
              #endif
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = shiftAndAct(temp13, activationFunction);
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = shiftAndAct(temp14, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

}
break;
   #endif
   #if OUTPUTBUFFER > 8
case 8:

for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
{

  temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);

        // }
        // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
  for(int turn=0; turn<2; turn++) {
    if(turn==0) {
      weight_ptr = weight_ptr1;
      inFeaturesSizeP2 = inFeaturesSize1/2;
      inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }
      addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
      addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
      addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
      addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
      addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
      addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
      addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
      addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

        in_addr = (((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
            SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
            SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
            SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
            SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
// #ifdef FMINTILING
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
// #endif
          // }
          }
// #ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//             v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);

//           }
// #endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
      // }
    }
    break;
   #endif
   #if OUTPUTBUFFER > 4
    case 4:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = weight_ptr2;
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        // }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


       // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
       // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

       in_addr = (uint32_t)(((v2s*)inFeatures));
       for(int i=0; i<inFeaturesSizeP4; i++) {
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
// #ifdef FMINTILING
            
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
// #endif
          // }
          }
// # ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//             v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);

//           }
// # endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }
    }
    break;
#endif
    case 2:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
        // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
        // }

      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = &weight_ptr1[0];
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = &weight_ptr2[0];
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
        for(int i=0; i<inFeaturesSizeP2; i++) {
          v2s inF_temp = ((v2s*)inFeatures)[i];
          
          // int o_rel;
          // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
          SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
          SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
          // }
        }
      } // fm in1 in2 
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
              // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

    }
    break;
    case 1:


    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {
     // iterate through both input FMs
     for(int turn=0; turn<2; turn++) {
      if(turn==0) {
        weight_ptr = &weight_ptr1[0];
        inFeaturesSizeP2 = inFeaturesSize1/2;
        inFeatures = inFeatures1;
      } else {
        weight_ptr = &weight_ptr2[0];
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
      }
      #ifdef FMINTILING
      inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
      #else
      inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
      #endif
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) {
        v2s inF_temp = ((v2s*)inFeatures)[i];
        temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], temp0);
      }
   } // fm in1 and in2 
   outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
 }
 break;
}
 // updat pointer for next iteration
bias_ptr1                = &bias_ptr1[outFeatureTiles*outFeaturesPerTile];
bias_ptr2                = &bias_ptr2[outFeatureTiles*outFeaturesPerTile];
weight_ptr1              = &weight_ptr1[(inFeaturesSize1/2*(outFeatureTiles*outFeaturesPerTile))];
weight_ptr2              = &weight_ptr2[(inFeaturesSize2/2*(outFeatureTiles*outFeaturesPerTile))];
outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
if (outFeaturesSize_remain==0) break;
}
PROFILING_TWOLINEAR_END
}
 ///////////////////////////////////////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______      _____ _______  _____          //
 // |     \ |______ |______ |_____| |     | |         |           |   |  |  | |_____] |       //
 // |_____/ |______ |       |     | |_____| |_____    |         __|__ |  |  | |       |_____  //
 //                                                                                           //
 ///////////////////////////////////////////////////////////////////////////////////////////////
#else // no FM Tiling or not ASIP
/** @brief Calculates Two Linear Layers which are accumulated.
 *
 *  Some networks like LSTM calculate a FC layer for two different input (e.g. x and c)
 *  which are then summed together to calculate the output feature map. 
 *  This is an optimization to avoid storing back all the intermediate output FM's to
 *  calculate the 2nd layer.
 *
 *  @param inFeaturesSize1 Number of input neurons of the FC layer 1
 *  @param inFeaturesSize2 Number of input neurons of the FC layer 2
 *  @param outFeaturesSize Number of output neurons
 *  @param activationFunction Type of activation function (ACT_NONE: no activation function is used, ACT_TANH: tangent hyperbolicus, ACT_SIG: sigmoid)
 *  @param weight1 Weights of FC1
 *  @param weight2 Weights of FC2
 *  @param bias1 Bias FC1
 *  @param bias2 Bias FC2
 *  @param inFeatures1 Pointer to input FM for FC1
 *  @param inFeatures2 Pointer to input FM for FC2
 *  @param outFeatures Pointer where to store output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
        // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{


  PROFILING_TWOLINEAR_START
  int inFeaturesSize1P2=inFeaturesSize1/2;
  int inFeaturesSize2P2=inFeaturesSize2/2;
  for (int o=0; o< outFeaturesSize; o++) 
  {
    outFeatures[o] = True?
    bias1[o]+bias2[o]:
    ((data_t)0);
    int32_t temp;
    temp = ((int32_t)bias1[o]+(int32_t)bias2[o])<<(q_fraqP1);

#ifdef SIMD

    for(int i=0; i<inFeaturesSize1P2; i++)
#else
      for(int i=0; i<inFeaturesSize1; i++)
#endif
      {
#ifdef SIMD
#ifdef ASIP 
        temp = temp + ((v2s*)inFeatures1)[i] * ((v2s*)weight1)[(inFeaturesSize1P2*o) + i];
#else //not ASIP
        temp = __SUMDOTP2(((v2s*)inFeatures1)[i], ((v2s*)weight1)[(inFeaturesSize1P2*o) + i], temp);
#endif //ASIP
#else // not SIMD
        temp += inFeatures1[i]*weight1[inFeaturesSize1*o + i];
#endif  
      }

#ifdef SIMD
      for(int i=0; i<inFeaturesSize2P2; i++)
#else
        for(int i=0; i<inFeaturesSize2; i++)
#endif
        {
#ifdef SIMD
#ifdef ASIP 
          temp = temp + ((v2s*)inFeatures2)[i] * ((v2s*)weight2)[(inFeaturesSize2P2*o) + i];
#else // not ASIP
          temp = __SUMDOTP2(((v2s*)inFeatures2)[i], ((v2s*)weight2)[(inFeaturesSize2P2*o) + i], temp);
#endif //ASIP
#else // not SIMD
          temp += inFeatures2[i]*weight2[inFeaturesSize2*o + i];
#endif // end SIMD

        }



#ifdef DOACTONTHEFLY
        temp = temp>>(q_fraqP1); 
        switch(activationFunction) {
          case ACT_NONE: outFeatures[o] = temp; break;
          case ACT_TANH: outFeatures[o] = generic_tanh(temp); break;
          case ACT_SIG:  outFeatures[o] = generic_sig(temp); break;
        }
#else
        outFeatures[o] = temp>>(q_fraqP1);
#endif



      }
      PROFILING_TWOLINEAR_END
    }
#endif
#endif

#ifndef FixedPt
    void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
      int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction,
        // Layer Parameters
      data_t * __restrict__ weight1,
      data_t * __restrict__ weight2,
      data_t * __restrict__ bias1,
      data_t * __restrict__ bias2,
        // Input and Output Features
      data_t * __restrict__ inFeatures1,
      data_t * __restrict__ inFeatures2,
      data_t * __restrict__ outFeatures)
    {

      PROFILING_TWOLINEAR_START
      int inFeaturesSize1P2=inFeaturesSize1/2;
      int inFeaturesSize2P2=inFeaturesSize2/2;
      for (int o=0; o< outFeaturesSize; o++) 
      {
        outFeatures[o] = True?
        bias1[o]+bias2[o]:
        ((data_t)0);
        data_t temp;
        temp = bias1[o]+bias2[o];



        for(int i=0; i<inFeaturesSize1; i++)
        {
         temp += inFeatures1[i]*weight1[inFeaturesSize1*o + i];
       }
       for(int i=0; i<inFeaturesSize2; i++)
       {
         temp += inFeatures2[i]*weight2[inFeaturesSize2*o + i];
       }
       outFeatures[o] = temp;

     }
     PROFILING_TWOLINEAR_END
   }
#endif





/** @brief Calculates point-wise Addition of Tensors (A+=B)
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Accumulation Tensor
 *  @param FeaturesB Addend Tensor
 */
    void NOINLINE AddTensor (
        // Layer Attributes
      int TensorSize,
        // Layer Parameters
      data_t * __restrict__ FeaturesA,
      data_t * __restrict__ FeaturesB)
    {
      PROFILING_ADDT_START

#ifdef ASIP
  // TODO implement SIMD add on tzscale
      for(int o=0; o<TensorSize; o++)
        FeaturesA[o] += FeaturesB[o];

      return;
#else //ASIP

#ifdef SIMD
      int TensorSizeP2 = TensorSize/2;
      v2s * SIMD_FeaturesA = (v2s*) FeaturesA;
      v2s * SIMD_FeaturesB = (v2s*) FeaturesB;
      for(int o=0; o<TensorSizeP2; o++)
        SIMD_FeaturesA[o] += SIMD_FeaturesB[o];
#else
      for(int o=0; o<TensorSize; o++)
        FeaturesA[o] += FeaturesB[o];
#endif
      PROFILING_ADDT_END
#endif
    }

/** @brief Calculates point-wise Multiplication of Tensors (A*=B) also known as Hadamard Product
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Accumulation Tensor
 *  @param FeaturesB Multiplicand Tensor
 */
    void NOINLINE HadMulTensor (
        // Layer Attributes
      int TensorSize,
        // Layer Parameters
      data_t * __restrict__ FeaturesA,
      data_t * __restrict__ FeaturesB)
    {
      PROFILING_HADM_START
      for (int o=0; o< TensorSize; o++) 
      {
#ifdef FixedPt
       FeaturesA[o] = (FeaturesA[o]*FeaturesB[o])>>(q_fraqP1);
#else
       FeaturesA[o] *= FeaturesB[o];
#endif
     }
     PROFILING_HADM_END
   }


/** @brief Copy of Tensor A to B
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Source Tensor
 *  @param FeaturesB Destination Tensor
 */
   void NOINLINE CopyTensor (
        // Layer Attributes
    int TensorSize,
        // Layer Parameters
    data_t * __restrict__ FeaturesA,
    data_t * __restrict__ FeaturesB)
   {
    PROFILING_COPY_START
    for (int o=0; o< TensorSize; o++) 
    {
      FeaturesA[o] = FeaturesB[o];
    }
    PROFILING_COPY_END
  }
/** @brief In-Place application of tangent hyperbolic on Tensor
 *
 *  @param TensorSize Input Value
 *  @param Features Input and Output of Activation Fucntion
 */
#ifdef ASIP_USETANHSIG
  void NOINLINE TanhLayer (
        // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Features)
  {
    PROFILING_TANH_START
    for(int i=0; i<TensorSize; i++) {
      Features[i] = tzscale_tanh(Features[i]); //tzscale_tanh(Features[i],0);
    }
    PROFILING_TANH_END
  }
#else
  void NOINLINE TanhLayer (
        // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Features)
  {
    PROFILING_TANH_START
    for (int o=0; o< TensorSize; o++) 
      Features[o] = Tanh(Features[o]);

    PROFILING_TANH_END
  }
#endif



/** @brief In-Place application of sigmoid activation on Tensor
 *
 *  @param TensorSize Input Value
 *  @param Features Input and Output of Activation Fucntion
 */
#ifdef ASIP_USETANHSIG
  void NOINLINE SigLayer (
        // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Features)
  {
    PROFILING_TANH_START

    for(int i=0; i<TensorSize; i++) {
      //ext_nop();chess_separator_scheduler();
      Features[i] = tzscale_sig(Features[i]); //tzscale_tanh(Features[i],0);
    }
    PROFILING_TANH_END
  }
#else
  void NOINLINE SigLayer (
        // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Features)
  {
    PROFILING_TANH_START
    for (int o=0; o< TensorSize; o++) 
      Features[o] = sig(Features[o]);

    PROFILING_TANH_END
  }
#endif





/** @brief Fills Tensor with contstant
 *
 *  @param TensorSize Input Value
 *  @param Tensor Input and Output of Activation Fucntion
 *  @param fillValue constant value
 */
  void NOINLINE fillTensor (
  // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Tensor,
    data_t fillValue)
  {
    PROFILING_FILL_START
    for (int o=0; o< TensorSize; o++) 
    {
      Tensor[o] = fillValue;
    }
    PROFILING_FILL_END
  }













// inline data_t  ALWAYS_INLINE  Tanh_old(data_t value) {
// #ifdef TURNOFF
//   #ifdef ASIP  
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
// #else // ASIP
//   #ifndef USE_INTRINSICS
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
//   #else // USET_INTRINSICS
// // [START clipur version]
//    int tmp = (((int)value+18432-2048)>>11);
//    asm volatile ("p.clipur" " %[c], %[a],%[b]\n"
//         : [c] "=r" (tmp)
//         : [a] "r" (tmp), [b] "r" (lb_lut_numelements)); // TODO: check puts p.clipu instead!!
// // [END clipur version]
//   #endif // USET_INTRINSICS
// #endif // ASIP
//   return ((lut_Tanh_m[tmp]*value) >> (q_fraqP1))+lut_Tanh_q[tmp];

// #else // TURNOFF
// #ifdef FixedPt 
//   //printf("%i, ", PIHalf);
//   if(Abs(value) < tanh_threshold) 
//   {

//     int x_power = value;
//     int partSum = value;

//     for(int i = 1; i<4; i++) {
//       x_power = ((x_power * value) >> (q_fraqP1)) * value >> (q_fraqP1);
//       // printf("%i, ", x_power);
//       partSum = partSum + (tanh_coeff[i]*x_power >> (q_fraqP1));
//     }
//     // printf("%i==%i, ", partSum, (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_fraqP1)))+1))*(1<<(q_fraqP1))));
//     return partSum;
//   } else {
//     // printf("tanh");
//     return (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_frac)))+1))*(1<<(q_frac)));
//   }  
// #else // FixedPt
//          return (1.0f - 2.0/(expTailor(tailorPrecission, 2*value)+1));
// #endif // FixedPt
//   #endif // TURNOFF
// }
// // inline v2s ALWAYS_INLINE sig_SIMD(v2s value)
// // {
// //    // TODO: NOT WORKING at the moment
// // #ifdef SIMD
// // #ifdef TURNOFF
// //   // return value;
// // // printf("qwer\n");
// //   int tmp = Max(Min(8, Abs(((int)value+2048)/4096)),0);
// //   return ((lut_fakeTanh_m[tmp]*value) >> (q_fraqP1))+lut_fakeTanh_q[tmp];
// // #endif // TURNOFF
// // #endif // SIMD

// //   printf("This function should not be called this way!");

// // }


// inline data_t ALWAYS_INLINE sig_old(data_t value)
// {

// #ifdef TURNOFF
// #ifdef ASIP
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
// #else // ASIP
//   #ifndef USE_INTRINSICS
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
//   #else // USET_INTRINSICS
// // [START clipur version]
//    int tmp = (((int)value+2048)/4096);
//    asm volatile ("p.clipur" " %[c], %[a],%[b]\n"
//         : [c] "=r" (tmp)
//         : [a] "r" (tmp), [b] "r" (lb_lut_numelements)); // TODO: check puts p.clipu instead!!
// // [END clipur version]
//   #endif // USET_INTRINSICS
// #endif //ASIP
//   return ((lut_sig_m[tmp]*value) >> (q_fraqP1))+lut_sig_q[tmp];

// #else
//   #ifdef FixedPt
//     if(Abs(value) < sig_threshold) 
//   {
//     int x_power = value;
//     int partSum = sig_coeff[0]+ (sig_coeff[1]*value >> (q_fraqP1));;

//     for(int i = 2; i<4; i++) {
//       x_power = (x_power * value >> (q_fraqP1)) * value >> (q_fraqP1);
//       // printf("(%i), ", sig_coeff[3]);
//       partSum = partSum + ((sig_coeff[i]*x_power) >> (q_frac));
      
//     }
//     // printf("%i==%i, ", partSum, (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_fraqP1)))+1))*(1<<(q_fraqP1))));
//     return partSum;
//   } else {
//          // printf("sig, %i>>%i:", value, q_frac);
//          //printFloat((data_t)((1.0/(1.0+expTailor(tailorPrecission, -(float)(value)/(1<<(q_frac)))))*(1<<(q_frac))));

//          return (data_t)((1.0/(1.0+expTailor(tailorPrecission, -(float)(value)/(1<<(q_frac)))))*(1<<(q_frac)));
//   }
//   #else

//          return 1.0/(1.0+expTailor(tailorPrecission, -((float)value)));
//   #endif
// #endif  
// }





/** @brief Calculates an RNN layer
 *
 *  Calculates an RNN layer based on 
 *  h_t = \\tanh(w_{ih} x_t + b_{ih}  +  w_{hh} h_{(t-1)} + b_{hh})
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 *  @param hiddenFeatures Hidden Feature Map
 */
void NOINLINE RNNLayer (
        // Layer Attributes
  int inFeaturesSize, int hiddenFeaturesSize,
        // Layer Parameters
  data_t * __restrict__ weight_ih_l,
  data_t * __restrict__ weight_hh_l,
  data_t * __restrict__ bias_ih_l,
  data_t * __restrict__ bias_hh_l,
        // Input and Output Features
  data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures, // out and hidden
        // Hidden Features
        data_t * __restrict__ hiddenFeatures)
{
  for(int seq=0; seq< rnn_seqSize; seq++) {
      LinearLayer(hiddenFeaturesSize,hiddenFeaturesSize,True,(data_t*)weight_hh_l, bias_hh_l, hiddenFeatures, outFeatures); //w_{hh} h_{(t-1)}
      LinearLayer(inFeaturesSize,hiddenFeaturesSize,True,(data_t*)weight_ih_l, bias_ih_l, inFeatures+seq*inFeaturesSize, hiddenFeatures); //w_{ih} x_t + b_{ih} 

      AddTensor(hiddenFeaturesSize, outFeatures, hiddenFeatures);
      TanhLayer(hiddenFeaturesSize, outFeatures);
      CopyTensor(hiddenFeaturesSize, hiddenFeatures, outFeatures);
    }
  }

/** @brief Calculates an LSTM layer
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param lstm_h hidden state tensor
 *  @param lstm_c cell state tensor
 *  @param lstm_f forget gate activation tensor
 *  @param inFeatures input feature map
 *  @param lstm_i input/update gate activation tensor
 *  @param lstm_g g tensor 
 *  @param lstm_o output gate tensor
 */
  void NOINLINE LSTMLayer (
// Layer Attributes
    int inFeaturesSize, int hiddenFeaturesSize,
        // Layer Parameters
    data_t * __restrict__ weight_ih_l,
    data_t * __restrict__ weight_hh_l,
    data_t * __restrict__ bias_ih_l,
    data_t * __restrict__ bias_hh_l,
        // Input and Output Features
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ lstm_h,
        // Hidden Features
    data_t * __restrict__ lstm_c,
        // intermediate nodes
    data_t * __restrict__ lstm_f,
    data_t * __restrict__ lstm_i,
    data_t * __restrict__ lstm_g,
    data_t * __restrict__ lstm_o
    )
  {
    PROFILING_LSTM_START
  #ifdef DEBUG_LSTM
    printf("lstm_in: ");PrintTensor(inFeaturesSize, inFeatures);
    #endif
    for(int seq=0; seq< lstm_seqSize; seq++) {

  //it=σ(Wiixt+bii+Whih(t−1)+bhi)
      TwoLinearLayersAccumulate (
          // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 

          // Layer Parameters
          weight_ih_l+0*inFeaturesSize*hiddenFeaturesSize, // weight1
          weight_hh_l+0*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
          bias_ih_l+0*hiddenFeaturesSize,   // bias1
          bias_hh_l+0*hiddenFeaturesSize,   // bias2 
          inFeatures+seq*inFeaturesSize,    // in1
          lstm_h,        // in2
          lstm_i);       // out
#ifdef DEBUG_LSTM
      printf("lstm_i: ");PrintTensor(hiddenFeaturesSize, lstm_i);
#endif
#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_i);
#endif
#ifdef DEBUG_LSTM
      printf("lstm_i: ");PrintTensor(hiddenFeaturesSize, lstm_i);
#endif
  //ft=σ(Wif xt+bif+Whf h(t−1)+bhf)
      TwoLinearLayersAccumulate (
          // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 

          // Layer Parameters
          weight_ih_l+1*inFeaturesSize*hiddenFeaturesSize, // weight1
          weight_hh_l+1*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
          bias_ih_l+1*hiddenFeaturesSize,   // bias1
          bias_hh_l+1*hiddenFeaturesSize,   // bias2 
          inFeatures+seq*inFeaturesSize,    // in1
          lstm_h,        // in2
          lstm_f);       // out
        #ifdef DEBUG_LSTM
      printf("lstm_f: ");PrintTensor(hiddenFeaturesSize, lstm_f);
    #endif
#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_f);
#endif
    #ifdef DEBUG_LSTM
      printf("lstm_f: ");PrintTensor(hiddenFeaturesSize, lstm_f);
    #endif

    //gt=tanh(Wigxt+big+Whgh(t−1)+bhg)
      TwoLinearLayersAccumulate (
          // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_TANH, 

          // Layer Parameters
          weight_ih_l+2*inFeaturesSize*hiddenFeaturesSize, // weight1
          weight_hh_l+2*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
          bias_ih_l+2*hiddenFeaturesSize,   // bias1
          bias_hh_l+2*hiddenFeaturesSize,   // bias2 
          inFeatures+seq*inFeaturesSize,    // in1
          lstm_h,        // in2
          lstm_g);       // out
    #ifdef DEBUG_LSTM
      printf("lstm_g: ");PrintTensor(hiddenFeaturesSize, lstm_g);
    #endif
#ifndef DOACTONTHEFLY
      TanhLayer(hiddenFeaturesSize, lstm_g);
#endif
    #ifdef DEBUG_LSTM
      printf("lstm_g: ");PrintTensor(hiddenFeaturesSize, lstm_g);
    #endif

    //ot=σ(Wioxt+bio+Whoh(t−1)+bho)
      TwoLinearLayersAccumulate (
          // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize,ACT_SIG, 

          // Layer Parameters
          weight_ih_l+3*inFeaturesSize*hiddenFeaturesSize, // weight1
          weight_hh_l+3*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
          bias_ih_l+3*hiddenFeaturesSize,   // bias1
          bias_hh_l+3*hiddenFeaturesSize,   // bias2 
          inFeatures+seq*inFeaturesSize,    // in1
          lstm_h,        // in2
          lstm_o);       // out
#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_o);
#endif
    #ifdef DEBUG_LSTM
      printf("lstm_o: ");PrintTensor(hiddenFeaturesSize, lstm_o);
    #endif
    //ct=ft*c(t−1)+it*gt
      HadMulTensor(hiddenFeaturesSize, lstm_c, lstm_f);
      HadMulTensor(hiddenFeaturesSize, lstm_i, lstm_g);
      AddTensor(hiddenFeaturesSize, lstm_c, lstm_i);
    #ifdef DEBUG_LSTM
      printf("lstm_c: ");PrintTensor(hiddenFeaturesSize, lstm_c);
    #endif
    //ht=ottanh(ct)
    CopyTensor(hiddenFeaturesSize, lstm_h, lstm_c); // h=c
    TanhLayer(hiddenFeaturesSize, lstm_h); // tanh(c_t)
    HadMulTensor(hiddenFeaturesSize, lstm_h, lstm_o);
    #ifdef DEBUG_LSTM
    printf("lstm_h: ");PrintTensor(hiddenFeaturesSize, lstm_h);
    #endif
  }   
  PROFILING_LSTM_END

}

/** @brief Print 2D Tensor
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 */
void PrintTensor2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray
  )
{
   // for 1d array -> set dim2 = 1
 for (int o=0; o< dim2; o++) 
 {
  printf("[");
  for(int i=0; i<dim1; i++)
  {
         // int temp = (int)(dataArray[dim2*o+i]*1000)%1000;

   printFloat(dataArray[dim2*o+i]);
   printf(", ");
 }
 printf("], ");
}
printf("\n");
}
 /** @brief Print 1D Tensor
 *  @param dim1 length of tensor
 *  @param dataArray data to be printed
 */
void PrintTensor (
        // Layer Attributes
  int dim1,
  data_t * __restrict__ dataArray
  )
{
 PrintTensor2D(dim1, 1, dataArray);
}

 /** @brief Print difference of 1D Tensor
 *  @param dim1 length
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 */
data_t PrintTensorDiff (
        // Layer Attributes
  int dim1,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array
  )
{
 return PrintTensorDiff2D(dim1, 1, dataArray, data2Array);
}
 /** @brief Print difference of 2D Tensor
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 */
data_t PrintTensorDiff2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array
  )
{
   // for 1d array -> set dim2 = 1
  int temp_sum = 0;
  for (int o=0; o< dim2; o++) 
  {
    printf("[");
    for(int i=0; i<dim1; i++)
    {
     data_t temp = dataArray[dim2*o+i]-data2Array[dim2*o+i];
     printFloat(temp);
     temp_sum += (int)temp*(int)temp;
     printf(", ");
   }
   printf("], ");
 }
 printf("\n");
 temp_sum /= (dim1*dim2);
 printf("mse= %d", temp_sum);
 return (data_t)temp_sum;
}
 /** @brief Calculates average quadratic error
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 *  @param error pointer to store error
 */
void error2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array,
  data_t * __restrict__ error
  )
{
   // for 1d array -> set dim2 = 1
 data_t temp = 0;
 for (int o=0; o< dim2; o++) 
 {

  for(int i=0; i<dim1; i++)
  {
   data_t temp2 = dataArray[dim2*o+i]-data2Array[dim2*o+i];
   temp += temp2*temp2;
 }

}
(*error) =  temp/dim2/dim1;
}
 /** @brief Prints a float value (used on RISC-Y without float unit) (currently deactivated)
 *  @param value 
 */
void printFloat(data_t value) {
// #ifndef ASIP
// #ifdef FixedPt
//     // printf("%i", value);

  // float tmp = ((float)value)/(1<<(q_frac));

  // if(tmp > -1.0f && tmp < 0.0f) {
  //   printf("-%i.%03i",  (int)(tmp), (int)Abs(tmp*1000)%1000);
  // }
  // else
  // {
  //   printf("%i.%03i",  (int)(tmp), (int)Abs(tmp*1000)%1000);
  // }
// #else
//   if(value > -1.0f && value < 0.0f) {
//     printf("-%i.%03i",  (int)(value), (int)Abs(value*1000)%1000);
//     // printf("%1.3f",  (value));
//   }
//   else
//   {
//     printf("%i.%03i",  (int)(value), (int)Abs(value*1000)%1000);
//     // printf("%1.3f",  (value));
//   }
// #endif
// #else
  printf("%i",  (int)(value));
// #endif
  




}
/** @brief Taylor Extension of the e^x function
 *
 *  @param n number of taylor coefficients
 *  @param x input value
 *  @return approximate value of e^x
 */
      inline float  ALWAYS_INLINE  expTailor(int n, float x) 
      { 
        float sum = 1.0f;

        for (int i = n - 1; i > 0; --i ) 
          sum = 1 + x * sum / i; 

        return sum; 
      } 

/** @brief Signum Function
 *
 *  @param value
 *  @return sgn(value)
 */
inline data_t ALWAYS_INLINE  Sgn(data_t value) { // zero is positive!
  return (value >= (data_t)0.0)?+1:-1;
}
#ifdef ASIP
inline data_t Min(data_t a, data_t b) {return (((a)<(b))?(a):(b));}
inline data_t Max(data_t a, data_t b) {return (((a)>(b))?(a):(b));}
inline data_t Abs(data_t a)           {return (((a)>(0.0))?(a):(-a));}
inline float  Min(float  a, float  b) {return (((a)<(b))?(a):(b));}
inline float  Max(float  a, float  b) {return (((a)>(b))?(a):(b));}
inline float  Abs(float  a)           {return (((a)>(0.0))?(a):(-a));}
inline int  Min(int  a, int  b) {return (((a)<(b))?(a):(b));}
inline int  Max(int  a, int  b) {return (((a)>(b))?(a):(b));}
inline int  Abs(int  a)           {return (((a)>(0.0))?(a):(-a));}
#else
 /** @brief Intrinsic for the PULP tanh extension
 *  @param tanh_value
 */
inline int pulpRNNExt_tanh(int tanh_value) {
  int tmp;
  asm volatile("pl.tanh %0, %1" : "=r" (tmp) : "r" (tanh_value) );
  return tmp;
}
 /** @brief Intrinsic for the PULP sigmoid extension
 *  @param sig_value
 */
inline int pulpRNNExt_sig(int sig_value) {
  int tmp;
  asm volatile("pl.sig %0, %1" : "=r" (tmp) : "r" (sig_value) );
  return tmp;
}

#endif

#ifdef PULP_USETANHSIG
/// Select tanh function to be used
inline data_t generic_tanh(data_t value) {return pulpRNNExt_tanh(value);}
/// Select sigmoid function to be used
inline data_t generic_sig(data_t value) {return pulpRNNExt_sig(value);}
#else
inline data_t generic_tanh(data_t value) {return Tanh(value);}
inline data_t generic_sig(data_t value) {return sig(value);}
#endif