/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
*  @file test.c
*  @brief Verification of the functionality of the new RNN instructions.
*
*  Test program for teh RNN instructions
*
* @author Renzo Andri (andrire)
*/
#include <rt/rt_api.h>
/// Sample values to which are used to calculated expected and actual values
#define VALUE0 (signed short)0xF191
#define VALUE1 (signed short)0xF192
#define VALUE2 (signed short)0xF193
#define VALUE3 (signed short)0xF194
#define VALUE4 (signed short)0xF195
#define VALUE5 (signed short)0xF196
#define VALUE6 (signed short)0xF197
#define VALUE7 (signed short)0xF198
#define VALUE8 (signed short)0xF193
#define VALUE9 (signed short)0xF192
#define VALUEA (signed short)0xF19B
#define VALUEB (signed short)0xF194

#define PARAM0 (signed short)0x1223
#define PARAM1 (signed short)0x1225
#define PART1 (signed int) 0xF2223333
#define PART2 (signed int) 0xF2210077
#define PART3 (signed int) 0xF2245678
#define PART4 (signed int) 0xF2212345



RT_L2_DATA static short myVect[12+4];

typedef int iss_cpu_state_t;

/// LUT tables for piecewise linear approximation
const short lut_Tanh_m[16] = {4021, 3563, 2835, 2070, 1418, 929, 592, 370, 228, 140, 86, 52, 32, 19, 12, 7};
const int lut_Tanh_q[16] = {17060, 512067, 2012407, 4361003, 7021506, 9510743, 11575189, 13158594, 14311861, 15123015, 15679911, 16055709, 16306104, 16471340, 16579558, 16650000};
const short lut_sig_m[16] = {1019, 988, 930, 850, 758, 660, 563, 472, 391, 319, 258, 207, 165, 131, 104, 82};
const int lut_sig_q[16] = {8389671, 8423495, 8544906, 8789991, 9169470, 9670607, 10264318, 10914030, 11583389, 12241371, 12864661, 13437943, 13952921, 14406803, 14800713, 15138308};


/** @brief This is the software implementation of the tanh and sigmoid instruction used as comparison
*
*  Iterates through all the layers while passing the intermediate FM with a double 
*  buffering approach
*
*  @param iss_cpu_state_t pointer to the isa simulator cpu state (not used here and can be 0-pointer)
*  @param value to be calculated
*  @param isSig boolean (1=sigmoid, 0=tanh)
*  @return tanh or sigmoid of value
*/
static inline unsigned int lib_TANHorSIG(iss_cpu_state_t *s, unsigned int a, short isSig) {
  unsigned int lutsize = 16;
  unsigned int value1 = 4096;     // 1 in Q3,12
  unsigned int value0p999 = 4095; // smallest number smaller than 1
  int m;
  int q;
  int q_signed;
  unsigned short sign = (a>>31) & 0x1;
  int tmp;
  int mac_result;
  int mac_result_signed;
  int abs_a;
  // calculate absolute value
  if(sign==0x1) {
    abs_a = (~a);
  } else {
    abs_a = (a);
  }
  tmp = abs_a>>(13-3); // get index of LUT
  if(tmp>=lutsize) { // if number higher than the range than take the saturated value (e.g. tanh(\infty)=1)
    if(isSig == 0) { // tanh
      return (sign==0x1)?(int)-value1: (int)value1;
    } else { // sig
      return (sign==0x1)?(int)0: (int)value1;
    }
  } else {
    if(isSig == 0) {
      m = lut_Tanh_m[tmp];
      q =lut_Tanh_q[tmp];
    } else {
      m = lut_sig_m[tmp];
      q =lut_sig_q[tmp];
    }
    mac_result = (m*abs_a+q)>>12;
    mac_result_signed = (sign==1)? ~mac_result : mac_result;
    if(isSig==1 && sign==1) {
      return (unsigned int) (value0p999+(mac_result_signed)); // 1-(mx+q)=4096+(~mac_result+1)=4095+(~mac_result)
    } else {
      return (unsigned int) mac_result_signed;
    }
  }
}


/** @brief Tanh wrapper 
*
*  Calculates tanh with a bit-wise SW model
*
*  @param iss_cpu_state_t pointer to the isa simulator cpu state (not used here and can be 0-pointer)
*  @param value to be calculated
*  @return tanh or sigmoid of value
*/
static inline unsigned int lib_TANH(iss_cpu_state_t *s, unsigned int a) { return lib_TANHorSIG(s, a, 0);}

/** @brief Sigmoid wrapper 
*
*  Calculates sigmoid with a bit-wise SW model
*
*  @param iss_cpu_state_t pointer to the isa simulator cpu state (not used here and can be 0-pointer)
*  @param value to be calculated
*  @return tanh or sigmoid of value
*/
static inline unsigned int lib_SIG(iss_cpu_state_t *s, unsigned int a) { return lib_TANHorSIG(s, a, 1);}

static int errors = 0;
static int numTests = 0;
/** @brief Helper Function to print the result
*
*  Calculates sigmoid with a bit-wise SW model
*
*  @param test Name of the test
*  @param acutal actual value
*  @param expected expected value
*/
static inline void print_result(char* test, int actual, int expected) {
  printf("%-80.80s", test); printf(":\t");
  numTests++;
  if(expected == actual) {
    printf("%8.8x vs. Exp. %8.8x [\033[0;32msuccess\033[0m] \n", actual, expected);
  }
  else {
    printf("%8.8x vs. Exp. %8.8x [\033[1;31mfailed\033[0m] \n", actual, expected);
    errors++;
  }
}
/** @brief Main function
*
*  Runs the set of tests for the RNN extensions
*
*  @return exit code
*/
int main()
{
  int tanh_result, sig_result;
  // if(0) {
  char str[20];
  for(int i=-5*4096; i<5*4096;i+=4096*0.128) {
   asm volatile("pl.tanh %0, %1" : "=r" (tanh_result) : "r" (i) );
   asm volatile("pl.sig %0, %1" : "=r" (sig_result) : "r" (i) );
   sprintf(str, "4096*tanh(%i/4096)=", i);
   print_result(str, tanh_result, lib_TANH((iss_cpu_state_t *) 0x0, i));
   sprintf(str, "4096*sigmoid(%i/4096)=", i);
   print_result(str, sig_result, lib_SIG((iss_cpu_state_t *) 0x0, i));
   }
   // }

  printf("Verification for pl.sdotsp.h extension\n"); 

  myVect[0] = VALUE0;
  myVect[1] = VALUE1;
  myVect[2] = VALUE2;
  myVect[3] = VALUE3;
  myVect[4] = VALUE4;
  myVect[5] = VALUE5;
  myVect[6] = VALUE6;
  myVect[7] = VALUE7;
  myVect[8] = VALUE8;
  myVect[9] = VALUE9;
  myVect[0xA] = VALUEA;
  myVect[0xB] = VALUEB;
  myVect[0xC] = 1;
  myVect[0xD] = 1;

  short parameters[4];
  parameters[0] = PARAM0;
  parameters[1] = PARAM1;
  signed int result, result2, result3, result4, result5;
  unsigned int temp;

  volatile int value = 0x12345678;

  short * myVect_currPtr = &myVect[0];
  short * myVect_2ndPtr  = &myVect[0xA];

  result = PART1; 


  temp = (parameters[1]<<16) | parameters[0];
  result2 = PART2; 
  result3 = PART3; 
  result4 = PART4;
  register int x0 asm("x0");

  asm volatile("p.lw %0, 0(%1)" : "=r" (result5) : "r" (myVect_currPtr)); 
  asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (myVect_currPtr) : "r" (x0) ); // preload no compute SPR=0x2222_1111
  asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (myVect_currPtr) : "r" (x0) ); // preload no compute SPR=0x4444_3333


  // SPR = 0x2222_1111
  asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (result), "+r" (myVect_currPtr) : "r" (temp) );
  // result = 0x2222*0x0005+0x1111*0x0003 + 0x2222_3333 = 0x22231110
  // SPR = 0x6666_0x5555

  asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (result2), "+r" (myVect_currPtr) : "r" (temp) );
  // SPR = 0x8888_0x7777
  // result = (5,3)*(0x4444, 0x3333)+0x00110077
  //        = 0x15554 + 0x9999 + 0x00110077
  //        = 0x1EEED + 0x00110077 = 0x0012EF64;
  asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (result2), "+r" (myVect_currPtr) : "r" (temp) );
  // SPR[0] = 0xAAAA_0x9999
  // result = (5,3)*(0x6666, 0x5555)+0x0012EF64
  //        = 0x1FFFE + 0xFFFF + 0x0012EF64
  //        = 0x2FFFD + 0x0012EF64 = 0x15EF61;

  asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (result3), "+r" (myVect_currPtr) : "r" (((v2s*)parameters)[0]) ); // does not work
  // SPR[1] = 0xCCCC_BBBB
  // result = (5,3)*(0x8888, 0x7777)+0x12345678
  //        = 0x2AAA8 + 0x16665 + 0x12345678
  //        = 0x4110D + 0x12345678 = 0x12386785;
  asm volatile("p.lw %0, 4(%1!)" : "=r" (temp), "+r" (myVect_2ndPtr));
  asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (result4), "+r" (myVect_currPtr) : "r" (temp) );
  print_result("Test 5: Load just before special instruction", result4, VALUE8*VALUEA+VALUE9*VALUEB+PART4);
  print_result("Test 1: Load before prefetch-load Act", result5, ((unsigned short)VALUE1)<<16|(unsigned short)VALUE0);
  print_result("Test 2: Simple Prefetch Act", result, VALUE1*PARAM1+VALUE0*PARAM0+PART1);
  print_result("Test 3: Acc. Forward Act", result2, PARAM1*VALUE5+PARAM0*VALUE4+(PARAM1*VALUE3+PARAM0*VALUE2)+PART2);
  print_result("Test 4: Acc. with SIMD cast (including negative operand 0x8888!)", result3, PARAM1*VALUE7+PARAM0*VALUE6+PART3);

  if(errors==0) {
    printf("\n>> \033[0;32mAll %i verification tests were successful!\033[0m \n", numTests, numTests);
    return 0;
  } else {
    printf("\n>> \033[1;31m%i/%i verification tests failed!\033[0m \n", errors, numTests);
    return -1; // return error exit code for CI
  }

}

