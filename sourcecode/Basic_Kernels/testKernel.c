/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
 *  @file testKernel.c
 *  @brief Test Program to test basic Kernels implemented in BasicKernel.c and the benchmark suite
 *
 *  Test program for Basic ML kernels
 *
 * @author Renzo Andri (andrire)
 */

#include <stdio.h>
#include <config.h>
#ifdef ASIP

#else
  // #include "pulp.h"
  // #include "rt/rt_api.h"
  #include "config_profiling.h"
#endif
#include "basicKernel.h"
/// @cond DOXYGEN_EXCLUDE
#include "benchmarks.h"
/// @endcond
// #include "string.h"
// #include <math.h>
// #include <stdlib.h>


#ifdef PROFILING
int event_mask;
#endif 


// buffer to store intermediate FM
RT_L2_DATA data_t buffer[BUFFER_SIZE];


int main()
{
  data_t tmp_avgerror = 0;

#ifdef ASIP
  long cycles_before = chess_cycle_count();
#endif


  // pointer to output FM
 data_t * m0_OutAct;

  #ifdef PROFILING

 numFunctionCalls = 0;
    rt_perf_init(&perf);// global performance counter
    int maskset[] = {\
      (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR), 
      (1<<RT_PERF_ACTIVE_CYCLES), 
      (1<<RT_PERF_LD_STALL), 
      (1<<RT_PERF_JR_STALL), 
      (1<<RT_PERF_IMISS), 
      (1<<RT_PERF_LD), 
      (1<<RT_PERF_ST), 
      (1<<RT_PERF_JUMP), 
      (1<<RT_PERF_BRANCH), 
      (1<<RT_PERF_BTAKEN), 
      (1<<RT_PERF_RVC), 
      (1<<RT_PERF_LD_EXT), 
      (1<<RT_PERF_ST_EXT), 
      (1<<RT_PERF_LD_EXT_CYC), 
      (1<<RT_PERF_ST_EXT_CYC), 
      (1<<RT_PERF_TCDM_CONT)\
    };
    for(unsigned int i=0; i < sizeof(maskset)/sizeof(int); i++) {
      rt_perf_conf(&perf, maskset[i]);
      rt_perf_reset(&perf);

      PROFILING_ALL_START
  #endif 
  // #ifdef PRINTF_ACTIVE
     printf("%s\n", "Start");
  // #endif
  #ifdef MODEL0 
      m0_OutAct = inferNetwork(model0, DEPTH0, m0_In, buffer);
      putchar('\n');
  // putchar('\n');
     
  // PrintTensor((int)(sizeof(m0_Out)/sizeof(data_t)), m0_OutAct);
  // PrintTensor((int)(sizeof(m0_Out)/sizeof(data_t)), m0_Out);
  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m0_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m0_Out)/sizeof(data_t)), m0_Out);
      PrintTensorDiff((int)(sizeof(m0_Out)/sizeof(data_t)), m0_OutAct, m0_Out);
  #endif
  #endif
  #ifdef MODEL1 
      m0_OutAct = inferNetwork(model1, DEPTH1, m1_In, buffer);
      putchar('\n');
      PrintTensor((int)(sizeof(m1_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m1_Out)/sizeof(data_t)), m1_Out);
      PrintTensorDiff((int)(sizeof(m1_Out)/sizeof(data_t)), m0_OutAct, m1_Out);
  #endif
  #ifdef MODEL2 
      m0_OutAct = inferNetwork(model2, DEPTH2, m2_In, buffer);
      putchar('\n');

  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m2_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m2_Out)/sizeof(data_t)), m2_Out);
      PrintTensorDiff((int)(sizeof(m2_Out)/sizeof(data_t)), m0_OutAct, m2_Out);
  #endif
  #endif


  #ifdef MODEL3 
      m0_OutAct = inferNetwork(model3, DEPTH3, m3_In, buffer);
      putchar('\n');
    // putchar('\n');
  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m3_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m3_Out)/sizeof(data_t)), m3_Out);
      PrintTensorDiff((int)(sizeof(m3_Out)/sizeof(data_t)), m0_OutAct, m3_Out);
  #endif
// printf("model3 was executed");
  #endif
  #ifdef MODEL4 
  // printf("MODEL4 is not implemented, due to missing information.");
    // putchar('\n');
  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m4_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m4_Out)/sizeof(data_t)), m4_Out);
      PrintTensorDiff((int)(sizeof(m4_Out)/sizeof(data_t)), m0_OutAct, m4_Out);
#endif
// printf("model3 was executed");
  #endif
  #ifdef MODEL5 

      m0_OutAct = inferNetwork(model5, DEPTH5, m5_In, buffer);
      putchar('\n');
      // putchar('\n');
  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m5_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m5_Out)/sizeof(data_t)), m5_Out);
      PrintTensorDiff((int)(sizeof(m5_Out)/sizeof(data_t)), m0_OutAct, m5_Out);
  #endif
// printf("model3 was executed");
  #endif
//printf("deleteme and following\n");
// PrintTensor((int)(sizeof(m6_Out)/sizeof(data_t)), m6_Out); 
 #ifdef MODEL6 
      // PrintTensor((int)(sizeof(m6_In)/sizeof(data_t)), m6_In);
      m0_OutAct = inferNetwork(model6, DEPTH6, m6_In, buffer);
      putchar('\n');
      PrintTensor((int)(sizeof(m6_Out)/sizeof(data_t)), m6_Out);
      PrintTensor((int)(sizeof(m6_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensorDiff((int)(sizeof(m6_Out)/sizeof(data_t)), m0_OutAct, m6_Out);
#ifdef PRINTF_ACTIVE
     // PrintTensor((int)(sizeof(m6_Out)/sizeof(data_t)), m0_OutAct);
//      PrintTensor((int)(sizeof(m6_Out)/sizeof(data_t)), m6_Out);
//      PrintTensorDiff((int)(sizeof(m6_Out)/sizeof(data_t)), m0_OutAct, m6_Out);
#endif
// printf("model3 was executed");
  #endif
  #ifdef MODEL7 
      m0_OutAct = inferNetwork(model7, DEPTH7, m7_In, buffer);
  #ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m7_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m7_Out)/sizeof(data_t)), m7_Out);
      PrintTensorDiff((int)(sizeof(m7_Out)/sizeof(data_t)), m0_OutAct, m7_Out);
  #endif
  #endif

  #ifdef MODEL8 
      m0_OutAct = inferNetwork(model8, DEPTH8, m8_In, buffer);
      putchar('\n');
      // putchar('\n');
#ifdef PRINTF_ACTIVE
      PrintTensor((int)(sizeof(m8_Out)/sizeof(data_t)), m0_OutAct);
      PrintTensor((int)(sizeof(m8_Out)/sizeof(data_t)), m8_Out);
      PrintTensorDiff((int)(sizeof(m8_Out)/sizeof(data_t)), m0_OutAct, m8_Out);
#endif
// printf("model8 was executed");
// 
  #endif

  #ifdef MODEL9 
   // printf("%s\n", "model9 start");
      m0_OutAct = inferNetwork(model9, DEPTH9, m9_In, buffer);
      putchar('\n');
   #ifdef PRINTF_ACTIVE
   PrintTensor((int)(sizeof(m9_Out)/sizeof(data_t)), m0_OutAct);//printf("%s\n", "model9 start");
   PrintTensor((int)(sizeof(m9_Out)/sizeof(data_t)), m9_Out);//printf("%s\n", "model9 start");
   PrintTensorDiff((int)(sizeof(m9_Out)/sizeof(data_t)), m0_OutAct, m9_Out);//printf("%s\n", "model9 start");
   #endif
  #endif

  #ifdef MODEL10 
   m0_OutAct = inferNetwork(model10, DEPTH10, m10_In, buffer);
   putchar('\n');
  #ifdef PRINTF_ACTIVE
   PrintTensor((int)(sizeof(m10_Out)/sizeof(data_t)), m0_OutAct);//printf("%s\n", "model9 start");
   PrintTensor((int)(sizeof(m10_Out)/sizeof(data_t)), m10_Out);//printf("%s\n", "model9 start");
   PrintTensorDiff((int)(sizeof(m10_Out)/sizeof(data_t)), m0_OutAct, m10_Out);//printf("%s\n", "model9 start");
   #endif
  #endif


#ifdef PRINTF_ACTIVE
   putchar('\n');
  #endif
#ifdef PROFILING

   PROFILING_ALL_END
   rt_perf_save(&perf);
 }
#endif 
#ifdef PRINTF_ACTIVE
 // printf("END\n");
 // printf("ende\n");
#endif

#ifdef PROFILING

 int models = 0;
#ifdef MODEL0 
 models |= 1<<0;
#endif
#ifdef MODEL1 
 models |= 1<<1;
#endif
#ifdef MODEL2 
 models |= 1<<2;
#endif
#ifdef MODEL3 
 models |= 1<<3;
#endif
#ifdef MODEL4 
 models |= 1<<4;
#endif
#ifdef MODEL5 
 models |= 1<<5;
#endif
#ifdef MODEL6 
 models |= 1<<6;
#endif
#ifdef MODEL7 
 models |= 1<<7;
#endif
#ifdef MODEL8 
 models |= 1<<8;
#endif
#ifdef MODEL9 
 models |= 1<<9;
#endif
#ifdef MODEL10 
 models |= 1<<10;
#endif

 printf("%s, %i, ", CODE_SEGMENT, models);

 printf("%d,", numFunctionCalls*sizeof(int)/sizeof(maskset));
 printf("%d,", rt_perf_get(&perf, RT_PERF_CYCLES));
 printf("%d,", rt_perf_get(&perf, RT_PERF_INSTR));
 printf("%d,", rt_perf_get(&perf, RT_PERF_ACTIVE_CYCLES));
 printf("%d,", rt_perf_get(&perf, RT_PERF_LD_STALL));
 printf("%d,", rt_perf_get(&perf, RT_PERF_JR_STALL));
 printf("%d,", rt_perf_get(&perf, RT_PERF_IMISS));
 printf("%d,", rt_perf_get(&perf, RT_PERF_LD));
 printf("%d,", rt_perf_get(&perf, RT_PERF_ST));
 printf("%d,", rt_perf_get(&perf, RT_PERF_JUMP));
 printf("%d,", rt_perf_get(&perf, RT_PERF_BRANCH));
 printf("%d,", rt_perf_get(&perf, RT_PERF_BTAKEN));
 printf("%d,", rt_perf_get(&perf, RT_PERF_RVC));
 printf("%d,", rt_perf_get(&perf, RT_PERF_LD_EXT));
 printf("%d,", rt_perf_get(&perf, RT_PERF_ST_EXT));
 printf("%d,", rt_perf_get(&perf, RT_PERF_LD_EXT_CYC));
 printf("%d,", rt_perf_get(&perf, RT_PERF_ST_EXT_CYC));
 printf("%d,", rt_perf_get(&perf, RT_PERF_TCDM_CONT));


#endif



#ifdef ASIP
#  ifdef PRINTF_ACTIVE
 long cycles_after = chess_cycle_count();
 int chess_storage(X31) eoc =tmp_avgerror;
 printf("eof = %i", eoc);
#    ifdef PRINTF_ACTIVE
 printf("The result is %d\nCycle executed: %ld\n", 0, cycles_after - cycles_before);
#    endif
#  endif
#endif
 return 0;

}
