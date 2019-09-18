# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
import torch
import numpy

nn = torch.nn;

mode = "fixedPt"
# mode = "float"
q_format = 3.12
linear_inFeaturesSize = 16
linear_outFeaturesSize =16

rnn_inFeaturesSize = 16
rnn_hiddenFeaturesSize = 16

lstm_inFeaturesSize = 16
lstm_hiddenFeaturesSize = 16


f = open("config.h", "w")
f.write("#define FixedPt 1" if mode=="fixedPt" else "//#define FixedPt 1"); f.close()


# helper functions
def float2fixedPt(q_format, number):
   p_int = int(q_format)
   p_frac = int((q_format-p_int)*100)
   # print(number)
   temp =  int(number * 2**(p_frac)) & (2**(1+p_int+p_frac)-1)
   # print(number)
   # print("=")
   if temp > (2**(p_int+p_frac)-1): # it is a negative number :)
      # print (-int((2**(1+p_int+p_frac)-temp)));
      return -int((2**(1+p_int+p_frac)-temp))
   # print(temp)
   return temp

def num2format(number):
   if mode == "fixedPt":
      return float2fixedPt(q_format, number)
   else:
      return float(number)

def _1DTensor2C(var_name, tensor):
   # print(tensor)
   # print(len(tensor[0]))
   length = len(tensor)
   tensor2D = False
   tensor3D = False
   if length == 1:
      length = len(tensor[0]);
      if length == 1:
         length = len(tensor[0][0]);
         tensor3D = True
      tensor2D = True;
   tmp = ""
   tmp +="RT_L2_DATA data_t "+var_name+"["+str(length)+"] = "
   tmp +=  "{"
   for i in range(0, length):
      if i!=0:
         tmp += ", "
      if tensor3D:
         tmp += str(num2format(tensor.data[0][0][i]))
      elif tensor2D:
         tmp += str(num2format(tensor.data[0][i]))
      else:
         tmp += str(num2format(tensor.data[i]))
   tmp += "};"
   return tmp

def _2DTensor2C(var_name, tensor):
   # print(tensor)
   # print(len(tensor[0]))
   tmp = ""
   tmp += "RT_L2_DATA data_t "+var_name+"["+str(len(tensor))+"]["+str(len(tensor[0]))+"] = "
   tmp += "{"
   for j in range(0, len(tensor)):
      if j!=0:
            tmp += ", "
      tmp +="{"
      for i in range(0, len(tensor[0])):
         if i!=0:
            tmp += ", "
         tmp += str(num2format(tensor.data[j][i]))
      tmp += "}"
   tmp += "};"
   return tmp


if __name__ == "__main__":
   # Linear Layer
   inFeaturesSize = linear_inFeaturesSize
   outFeaturesSize =linear_outFeaturesSize
   LinLay = nn.Linear(inFeaturesSize, outFeaturesSize, True)
   inputFM = torch.randn(1, inFeaturesSize)


   # happy debug
   # inputFM.data[0][0] = -1
   # inputFM.data[0][1] = 2
   # # inputFM.data[2] = 0
   # LinLay.weight.data[0][0] = -1
   # LinLay.weight.data[0][1] = -1
   # LinLay.weight.data[1][0] = 0
   # LinLay.weight.data[1][1] = 0
   LinLay.bias.data[0] = 0
   LinLay.bias.data[1] = 0
   LinLay.bias.data[0] = 0
   LinLay.bias.data[1] = 0
   outputFM = LinLay(inputFM)

   prefix = "linLay_"
   print("char * testName = \"LinearLayer\";");
   print(_1DTensor2C(prefix+"OutExp", outputFM))
   print("RT_L2_DATA data_t "+prefix+"OutAct["+str(len(outputFM[0]))+"];")
   print("/*");print(outputFM);print("*/")
   print(_1DTensor2C(prefix+"In", inputFM))
   print("/*");print(inputFM);print("*/")
   print(_1DTensor2C(prefix+"Bias", LinLay.bias))
   print(_2DTensor2C(prefix+"Weights", LinLay.weight))

   print("int "+prefix+"inFeatureSize = "+str(inFeaturesSize)+";");
   print("int "+prefix+"outFeatureSize = "+str(outFeaturesSize)+";");

   #RNN
   print("// RNN Parameter")
   
   numLayers = 1
   inFeaturesSize = rnn_inFeaturesSize
   hiddenFeaturesSize = rnn_hiddenFeaturesSize
   seq_len = 1
   prefix = "rnn_"

   #size(inpuFM)=(seq_len, batch, input_size)
   
   inputFM = torch.randn(seq_len, 1, inFeaturesSize);
   
   # print(inputFM)
   # inputFM.data[0][0][0] = 1 
   # inputFM.data[0][0][1] = 0

   #                                                              Bias, BatchFirst, DropOut, Bidir
   RNNLay = nn.RNN(inFeaturesSize, hiddenFeaturesSize, numLayers, True, False,      0,       False)
   # RNNLay.weight_hh_l0.data.fill_(1)
   # RNNLay.weight_ih_l0.data.fill_(1)
   # RNNLay.bias_hh_l0.data.fill_(0)
   # RNNLay.bias_ih_l0.data.fill_(0)



   #size = seq_len, batch, num_directions*hidden_size
   print("/*")
   outputFM = RNNLay.forward(inputFM);
   print("*/")

   layer_id = 0
   
   print(_2DTensor2C(prefix+"weight_ih_l"+str(layer_id), eval("RNNLay.weight_ih_l"+str(layer_id))))
   print(_2DTensor2C(prefix+"weight_hh_l"+str(layer_id), eval("RNNLay.weight_hh_l"+str(layer_id))))
   print(_1DTensor2C(prefix+"bias_ih_l"+str(layer_id), eval("RNNLay.bias_ih_l"+str(layer_id))))
   print(_1DTensor2C(prefix+"bias_hh_l"+str(layer_id), eval("RNNLay.bias_hh_l"+str(layer_id))))




   print(_2DTensor2C(prefix+"In", inputFM.reshape(seq_len, inFeaturesSize)))
   print(_1DTensor2C(prefix+"OutExp", outputFM[0][seq_len-1][0]))
   print("RT_L2_DATA data_t "+prefix+"OutAct["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"hiddenNode["+str(hiddenFeaturesSize)+"];")
   print("int "+prefix+"inFeatureSize = "+str(inFeaturesSize)+";");
   print("int "+prefix+"hiddenFeatureSize = "+str(hiddenFeaturesSize)+";");
   print("int "+prefix+"seqSize = "+str(seq_len)+";");
   print("/*")
   print(outputFM);
   print("*/")


   ## LSTM
   print("// LSTM Parameter")
   numLayers = "multilayer not implemented"
   inFeaturesSize =lstm_inFeaturesSize
   hiddenFeaturesSize = lstm_hiddenFeaturesSize
   seq_len = 1
   prefix = "lstm_"
   LSTMLay = torch.nn.LSTM(inFeaturesSize, hiddenFeaturesSize)


   inputFM = torch.randn(seq_len, 1, inFeaturesSize)
   print("/*")
   outputFM = LSTMLay.forward(inputFM)
   print("*/")
   layer_id = 0
   print(_2DTensor2C(prefix+"weight_ih_l"+str(layer_id), eval("LSTMLay.weight_ih_l"+str(layer_id))))
   print(_2DTensor2C(prefix+"weight_hh_l"+str(layer_id), eval("LSTMLay.weight_hh_l"+str(layer_id))))
   print(_1DTensor2C(prefix+"bias_ih_l"+str(layer_id), eval("LSTMLay.bias_ih_l"+str(layer_id))))
   print(_1DTensor2C(prefix+"bias_hh_l"+str(layer_id), eval("LSTMLay.bias_hh_l"+str(layer_id))))


   print(_2DTensor2C(prefix+"In", inputFM.reshape(seq_len, inFeaturesSize)))
   print(_1DTensor2C(prefix+"oExp", outputFM[0][seq_len-1][0]))
   print("RT_L2_DATA data_t "+prefix+"oAct["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"h["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"c["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"f["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"i["+str(hiddenFeaturesSize)+"];")
   print("RT_L2_DATA data_t "+prefix+"g["+str(hiddenFeaturesSize)+"];")

   print("int "+prefix+"inFeatureSize = "+str(inFeaturesSize)+";");
   print("int "+prefix+"hiddenFeatureSize = "+str(hiddenFeaturesSize)+";");
   print("int "+prefix+"seqSize = "+str(seq_len)+";");
   print("/*")
   print(outputFM);
   print("*/")
