# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

# Notes Regarding dimension the system either take the actual values or assume 4 antennas/radios and 3 frequency bands

import torch
import numpy
import os
import sys
sys.path.insert(0, '../')
from math import ceil
from pyTorch_Kernels import _1DTensor2C, _2DTensor2C, num2format
from enum import Enum
from functools import reduce
nn=torch.nn

numAntenna = 4
freqBands = 3

copyright="// Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri"

def bash(cmd):
   command = os.popen(cmd)
   tmp = command.read()
   command.close()
   return tmp

class myLSTM(nn.LSTM):
   def __init__(self, inNodes, hiddenNodes):
      super().__init__(inNodes, hiddenNodes)
      num_directions = 2 if self.bidirectional else 1
      max_batch_size = 1 # batch size 1
      # self.hx = input.new_zeros(self.num_layers * num_directions,
      #                            max_batch_size, self.hidden_size,
      #                            requires_grad=False)

      # self.hx = torch.randn(self.num_layers * num_directions,
                                 # max_batch_size, self.hidden_size)
      self.hx = (torch.randn(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size), torch.randn(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size))
   def forward(self, input):
      # print(self.hx[0])
      print("hx=")
      print(self.hx)

      output, hn = super().forward(input, self.hx)
      print("output=")
      print(output)
      print("hn=")
      print(hn)
      self.hx = hn
      return hn[0]
      # self.hx = tmp[1]
      # return tmp[0]
inputFM = torch.randn(1, 1, 3)
a=myLSTM(3,4)
a.forward(inputFM)

# x=
# (tensor([[[-1.8549, -1.7508]]]), tensor([[[0.3502, 1.2633]]]))
# (tensor([[[-1.8549, -1.7508]]]), tensor([[[0.3502, 1.2633]]]))
# output=
# tensor([[[0.0660, 0.6513]]], grad_fn=<CatBackward>)
# hn=
# (tensor([[[0.0660, 0.6513]]], grad_fn=<ViewBackward>), tensor([[[0.3251, 1.1595]]], grad_fn=<ViewBackward>))


def figlet(txt):
   print(bash('figlet "'+txt+'"'))

def dbgPrint(text):
   print(text)

class Importance(Enum):
   LOW = 0
   MED = 1
   HIGH = 2
   URG = 3
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def todo(what, importance=Importance.LOW):
   if importance==Importance.LOW:
      print(bcolors.WARNING+"TODO: "+what+bcolors.ENDC)
   else:
      print(bcolors.FAIL+"TODO: "+what+bcolors.ENDC)
def info(what):
   print(bcolors.BOLD+"Info: "+what+bcolors.ENDC)
def error(what):
   print(bcolors.FAIL+"ERROR: "+what+bcolors.ENDC)

class netModel():
   def __init__(self, model, h_im=0, w_im=0):
        self.model = model
        self.numLayers = len(model)
        self.h_im = h_im
        self.w_im = w_im
        if self.numLayers == 0:
            self.in_features = 0
            self.out_features = 0
            return;
        if isinstance(model[0], nn.Linear):
            self.in_features  = model[0].in_features
        elif isinstance(model[0], myLSTM):
            self.in_features  = model[0].input_size
        elif isinstance(model[0], nn.Conv2d):
            self.in_features  = model[0].in_channels
        else: 
             error(str(type(model[0]))+"not defined")
        if isinstance(model[self.numLayers-1], nn.Linear):
            self.out_features = model[self.numLayers-1].out_features
        elif isinstance(model[self.numLayers-1], myLSTM):
            self.out_features = model[self.numLayers-1].hidden_size
        elif isinstance(model[self.numLayers-1], nn.Conv2d):
            self.out_features  = model[self.numLayers-1].out_channels
        else: 
             error(str(type(model[self.numLayers-1]))+"not defined")
   def __repr__(self):
        return "netModel(model={}, numLayers={}, in_features={}, out_features{})".format(self.model.__repr__(), self.numLayers,  self.in_features,  self.out_features)

   def numParams(netModels):
        numParams = 0;
        
        if isinstance(netModels, netModel):
         netModels = list([netModels])

        for _netModel in netModels:
           for layer in _netModel.model.modules():
               if isinstance(layer, nn.Sequential):
                   a=None#skip for now
               elif isinstance(layer, nn.Linear):
   #                 print(layer.weight.size()[0]*layer.weight.size()[1])
   #                 print(layer.bias.size()[0])
                   numParams += layer.weight.size()[0]*layer.weight.size()[1]
                   numParams += layer.bias.size()[0]
               elif isinstance(layer, myLSTM):
                   if layer.num_layers >1:
                       error("Multi-layer LSTM blocks not implemnted. use several units")
                   numParams += layer.weight_ih_l0.size()[0]*layer.weight_ih_l0.size()[1]+layer.bias_ih_l0.size()[0];
                   numParams += layer.weight_hh_l0.size()[0]*layer.weight_hh_l0.size()[1]+layer.bias_hh_l0.size()[0];
                   numParams += 0
               elif isinstance(layer, nn.Conv2d):
                   numParams +=  reduce(lambda x, y: x*y, layer.weight.size(), 1)+layer.bias.size()[0];
               else: 
                   error(str(type(layer))+" not defined")
        return numParams
   def printReport(netModels, sumRow=False):
      if isinstance(netModels, netModel):
         netModels = list([netModels])
      it = 1
      for _netModel in netModels:
         print("║{:2d}│{:10d}k│".format(it, ceil(netModel.numParams(_netModel)/1000)), end="")

         print("")
         it += 1

      print("╠{:═^2s}╪{:═^11s}╪╣".format("",""))
      print("║{:2s}│{:10d}k│║".format(" t", ceil(netModel.numParams(netModels)/1000)), end="")
      print("\n╚{:═^2s}╧{:═^11s}╧╝".format("", ""))
      # print(netModel.numParams(netModels))

   def exportModel(netModels, h_im=0, w_im=0):
      if isinstance(netModels, netModel):
         netModels = list([netModels])
      print('asdf')
      data_f = open("benchmarks.h",'w')
      write2file = lambda x : data_f.write(x+"\n")
      write2file(copyright)
      moveFilePointer = lambda x : data_f.seek(data_f.tell() - x, os.SEEK_SET)
      # print(model)
      # print(len(model))
      #
      # input
      modelID = 0
      for _netModel in netModels:
         print(_netModel)
         layID = 0 
         prefix = "m{}_".format(modelID)
         if len(_netModel.model) == 0:
            error("empty model")
            modelID += 1
            continue;
         _h_im = _netModel.h_im if _netModel.h_im != 0 else h_im
         _w_im = _netModel.w_im if _netModel.w_im != 0 else w_im
         inputFM = torch.randn(1, _netModel.in_features) if isinstance(_netModel.model[0], nn.Linear) else \
         torch.randn(1, _netModel.in_features, _h_im, _w_im) if isinstance(_netModel.model[0], nn.Conv2d) else \
         torch.randn(1, 1, _netModel.in_features);
         # inputFM = inputFM.fill_(torch.ones(2).mul(3)
         info("rnn/lstm first layers are accounted as 1x1xin for batch=1, seq=1")

         # debug session
         # inputFM[0][1].data.fill_(0)
         # inputFM[0][0].data.fill_(0)
         # inputFM[0][0][0][0].data.fill_(1)
         # inputFM[0][0][0][1].data.fill_(2)
         # inputFM[0][0][0][2].data.fill_(4)
         # inputFM[0][0][0][3].data.fill_(8)
         # inputFM[0][0][0][4].data.fill_(16)

         # inputFM[0][0][1][0].data.fill_(32*1)
         # inputFM[0][0][1][1].data.fill_(32*2)
         # inputFM[0][0][1][2].data.fill_(32*4)
         # inputFM[0][0][1][3].data.fill_(32*8)
         # inputFM[0][0][1][4].data.fill_(32*16)



         # print(inputFM)
         write2file("// Model")
         write2file("#ifdef MODEL"+str(modelID))
         # write2file(_1DTensor2C(prefix+"In", inputFM.clone().view(-1)))
         # inputFM.data.fill_(1)
         if isinstance(_netModel.model[0], nn.Conv2d): 
          write2file(_1DTensor2C(prefix+"In", inputFM.permute(0,2,3,1).clone().view(-1)))
         else:
          write2file(_1DTensor2C(prefix+"In", inputFM.clone().view(-1)))
         
         print("inputfm=")
         print(inputFM)
         write2file("#define DEPTH{} {}\n".format(modelID, len(_netModel.model)))
         netDef_c = "struct layer model{}[{}] = {{".format(modelID, len(_netModel.model))
         for layer in _netModel.model.children():
            # print(layer)
            info(str(layID))
            netDef_c += ", \\\n " if layID != 0 else "\\\n "
            if isinstance(layer, nn.Linear):
               
               dbgPrint("Linear Layer")
               write2file("// Linear Layer")

         #
               inFeaturesSize = layer.in_features
               outFeaturesSize = layer.out_features
               
               write2file("// inputFM.size = "+inputFM.size().__repr__()+"\n");

               if len(inputFM.size()) == 4:
                inputFM = inputFM.view(-1)
               outputFM = layer.forward(inputFM)
               

               write2file("// outputFM.size = "+outputFM.size().__repr__()+"\n");
         #
               prefix = "m{}_linear{}_".format(modelID, layID)
         #       print(outputFM)
               write2file("/*\n");
               write2file(_1DTensor2C(prefix+"OutExp", outputFM))
               write2file(str(outputFM))
               write2file("*/\n");
               # write2file("data_t "+prefix+"OutExp["+str(len(outputFM[0]))+"];")
               # print(_1DTensor2C(prefix+"In", inputFM))
               write2file(_1DTensor2C(prefix+"Bias", layer.bias))
               write2file(_2DTensor2C(prefix+"Weights", layer.weight))
         #
               print("int "+prefix+"inFeatureSize = "+str(inFeaturesSize)+";")
               print("int "+prefix+"outFeatureSize = "+str(outFeaturesSize)+";")
              
               netDef_c += "{{.type=LINEAR, .attributes={{{},{},{},{},{}}}, ".format(inFeaturesSize, outFeaturesSize, 0,0,0)
               netDef_c += ".parameters={{{},{}[0],{},{},{},{}}}}}".format(prefix+"Bias", prefix+"Weights",0,0,0,0)
            elif isinstance(layer, myLSTM):
               dbgPrint("LSTM")
               write2file("// LSTM Layer")
               seq_len = 1 # sequence length is 1! TODO
               inFeaturesSize = layer.input_size
               outFeaturesSize = layer.hidden_size
               hiddenFeaturesSize = outFeaturesSize
               # print(inputFM)
               prefix = "m{}_lstm{}_".format(modelID, layID)
               print(layer.hx[0])
               write2file(_1DTensor2C(prefix+"h", layer.hx[0]))
               write2file(_1DTensor2C(prefix+"c", layer.hx[1]))
               write2file("// inputFM.size = "+inputFM.size().__repr__()+"\n");
               outputFM = layer.forward(inputFM)

               write2file("// outputFM.size = "+outputFM.size().__repr__()+"\n");
               write2file("/*\n");
               write2file(_1DTensor2C(prefix+"OutExp", outputFM))
               write2file("*/\n");
               # print(outputFM)

               
               
               layer_id = 0
               write2file(_2DTensor2C(prefix+"weight_ih_l"+str(layer_id), eval("layer.weight_ih_l"+str(layer_id))))
               write2file(_2DTensor2C(prefix+"weight_hh_l"+str(layer_id), eval("layer.weight_hh_l"+str(layer_id))))
               write2file(_1DTensor2C(prefix+"bias_ih_l"+str(layer_id), eval("layer.bias_ih_l"+str(layer_id))))
               write2file(_1DTensor2C(prefix+"bias_hh_l"+str(layer_id), eval("layer.bias_hh_l"+str(layer_id))))

               write2file("/*\n");
               write2file(_2DTensor2C(prefix+"In", inputFM.reshape(seq_len, inFeaturesSize)))
               write2file("*/\n");
               # print(_1DTensor2C(prefix+"oExp", outputFM[0][seq_len-1][0]))
               # write2file("data_t "+prefix+"oAct["+str(hiddenFeaturesSize)+"];")

               # write2file("data_t "+prefix+"h["+str(hiddenFeaturesSize)+"];")
               # write2file("data_t "+prefix+"c["+str(hiddenFeaturesSize)+"];")
               # write2file("data_t "+prefix+"f["+str(hiddenFeaturesSize)+"];")
               # write2file("data_t "+prefix+"i["+str(hiddenFeaturesSize)+"];")
               # write2file("data_t "+prefix+"g["+str(hiddenFeaturesSize)+"];")

               print("int "+prefix+"inFeatureSize = "+str(inFeaturesSize)+";");
               print("int "+prefix+"hiddenFeatureSize = "+str(hiddenFeaturesSize)+";");
               print("int "+prefix+"seqSize = "+str(seq_len)+";");
               print("/*")
               print(outputFM);
               print("*/")

               netDef_c += "{{.type=LSTM, .attributes={{{},{},{},{},{}}}, ".format(inFeaturesSize, hiddenFeaturesSize, 0,0,0)
               netDef_c += ".parameters={{{}[0],{}[0],{},{},{},{}}}}}".format(prefix+"weight_ih_l"+str(layer_id),prefix+"weight_hh_l"+str(layer_id),prefix+"bias_ih_l"+str(layer_id),prefix+"bias_hh_l"+str(layer_id), prefix+"h", prefix+"c")
            elif isinstance(layer, nn.Conv2d):
              write2file("// Conv2D Layer")
              # layer.weight.data.fill_(2**-5)
              # layer.bias.data.fill_(1)
              inFeaturesSize = layer.in_channels
              outFeaturesSize = layer.out_channels
              kernelSize = layer.kernel_size[0]
              assert(layer.kernel_size[0]==layer.kernel_size[1]), "not supported"
              # print(inputFM)
              prefix = "m{}_Conv2d{}_".format(modelID, layID)

              tmp ="RT_L2_DATA data_t {}weight[{}] = {{".format(prefix, reduce(lambda x, y: x*y, layer.weight.size(), 1))
              dimension_order = [0,2,3,1];   
              print("weight=") 
              print(layer.weight.data)
              # layer.weight.data.fill_(1)
              for a in range(0, layer.weight.size()[dimension_order[0]]):
                for b in range(0, layer.weight.size()[dimension_order[1]]):
                  for c in range(0, layer.weight.size()[dimension_order[2]]):
                    for d in range(0, layer.weight.size()[dimension_order[3]]):
                      # print(num2format(layer.weight.data[eval(chr(ord('a')+dimension_order.index(0)))][eval(chr(ord('a')+dimension_order.index(1)))][eval(chr(ord('a')+dimension_order.index(2)))][eval(chr(ord('a')+dimension_order.index(3)))]))
                      tmp +=str(num2format(layer.weight.data[eval(chr(ord('a')+dimension_order.index(0)))][eval(chr(ord('a')+dimension_order.index(1)))][eval(chr(ord('a')+dimension_order.index(2)))][eval(chr(ord('a')+dimension_order.index(3)))])) # take order like in dimension_order
                      tmp +=", "
              write2file(tmp)
              moveFilePointer(3)

              write2file("};\n");
              print("bias=") 
              print(layer.bias.data)
              write2file(_1DTensor2C(prefix+"bias", layer.bias))

              outputFM = layer.forward(inputFM)
              print("out=") 
              print(outputFM)
              netDef_c += "{{.type=Conv2d, .attributes={{{},{},{},{},{}}}, ".format(inFeaturesSize, outFeaturesSize, kernelSize, _h_im, _w_im)
              netDef_c += ".parameters={{{},{},{},{},{},{}}}}}".format(prefix+"weight",prefix+"bias",0,0,0,0)
            else:
               error("not implemented")
            inputFM = outputFM.clone()
            layID += 1
            
         write2file(_1DTensor2C("m{}_Out".format(modelID), outputFM.clone().view(-1)))
         netDef_c += "};"
         write2file(netDef_c)   
         write2file("#endif")
         modelID += 1
      data_f.close()
#end class netModel


# U+250x   ─  ━  │  ┃  ┄  ┅  ┆  ┇  ┈  ┉  ┊  ┋  ┌  ┍  ┎  ┏

# U+251x   ┐  ┑  ┒  ┓  └  ┕  ┖  ┗  ┘  ┙  ┚  ┛  ├  ┝  ┞  ┟

# U+252x   ┠  ┡  ┢  ┣  ┤  ┥  ┦  ┧  ┨  ┩  ┪  ┫  ┬  ┭  ┮  ┯

# U+253x   ┰  ┱  ┲  ┳  ┴  ┵  ┶  ┷  ┸  ┹  ┺  ┻  ┼  ┽  ┾  ┿

# U+254x   ╀  ╁  ╂  ╃  ╄  ╅  ╆  ╇  ╈  ╉  ╊  ╋  ╌  ╍  ╎  ╏

# U+255x   ═  ║  ╒  ╓  ╔  ╕  ╖  ╗  ╘  ╙  ╚  ╛  ╜  ╝  ╞  ╟

# U+256x   ╠  ╡  ╢  ╣  ╤  ╥  ╦  ╧  ╨  ╩  ╪  ╫  ╬  ╭  ╮  ╯

# U+257x   ╰  ╱  ╲  ╳  ╴  ╵  ╶  ╷  ╸  ╹  ╺  ╻  ╼  ╽  ╾  ╿


models = []
########################################################################################
#  ____       __    ___  _                                                             #
# |  _ \ ___ / _|  / _ \/ |                                                            #
# | |_) / _ \ |_  | | | | |                                                            #
# |  _ <  __/  _| | |_| | |                                                            #
# |_| \_\___|_|    \___/|_|                                                            #
## Proactive Resource Management in LTE-U Systems: A Deep Learning Perspective         #
########################################################################################

l1_lstm_inNodes = 10 # M (guessed number)
l1_lstm_hiddenNodes = 70 
l1_lstm_outNodes = 70 # (guessed number)
l2_mlp_outNodes = 70 # (guessed number)
l3_lstm_hiddenNodes = 70
l3_lstm_outNodes = 4 # J available (guessed number)

models.append(netModel(nn.Sequential(myLSTM(l1_lstm_inNodes, l1_lstm_hiddenNodes),
                        nn.Linear(l1_lstm_hiddenNodes, l2_mlp_outNodes, True),
                        myLSTM(l2_mlp_outNodes, l3_lstm_outNodes))))
info("Model 1 created.")

########################################################################################
#  ____       __    ___ ____                                                           #
# |  _ \ ___ / _|  / _ \___ \                                                          #
# | |_) / _ \ |_  | | | |__) |                                                         #
# |  _ <  __/  _| | |_| / __/                                                          #
# |_| \_\___|_|    \___/_____|                                                         #
##Deep Multi-User Reinforcement Learning for Distributed Dynamic Spectrum Access
########################################################################################
#                        ? +-----------+    +-----+ k+1    +------+
#                       +->+Value Layer+----+ acc +------->+Output|
# +-----+ 2k+2 +------+ |  +-----------+    +-+---+        +------+
# |Input+----->+ LSTM +-+                     |
# +-----+      +------+ |? +---------------+  |
#                       +->+Advantage Layer+--+
#                          +---------------+
# Hints: input and output are clear, the rest is not explained in detail

K= freqBands
l1_lstm_inNodes = 2*K+2 #(actual)
l1_lstm_hiddenNodes = l1_lstm_inNodes #(guessed)
l2_mlp_outNodes = 2*(K+1) #(It is assumed that the topology is a single-layer FC, the  #
                         #parallel networks can be merged in case of FC)
models.append(netModel(nn.Sequential(myLSTM(l1_lstm_inNodes, l1_lstm_hiddenNodes),
                        nn.Linear(l1_lstm_hiddenNodes, l2_mlp_outNodes, True))))
info("Model 2 created.")

########################################################################################
#  ____       __    ___ _____                                                          #
# |  _ \ ___ / _|  / _ \___ /                                                          #
# | |_) / _ \ |_  | | | ||_ \                                                          #
# |  _ <  __/  _| | |_| |__) |                                                         #
# |_| \_\___|_|    \___/____/                                                          #
## Deep reinforcement learning for resource allocation in V2V communications           #
########################################################################################

## Ref 2: 
l1_mlp_inNodes = 2*freqBands   # actual numbers
l2_mlp_inNodes = 500  # actual numbers
l3_mlp_inNodes = 250  # actual numbers
l4_mlp_inNodes = 120  # actual numbers
l5_mlp_inNodes = 2*freqBands   # guessed number

models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l2_mlp_inNodes, True),
                        nn.Linear(l2_mlp_inNodes, l3_mlp_inNodes, True),
                        nn.Linear(l3_mlp_inNodes, l4_mlp_inNodes, True),
                        nn.Linear(l4_mlp_inNodes, l5_mlp_inNodes, True))))
info("Model 3 created.")
########################################################################################
#  ____       __    ___  _  _                                                          #
# |  _ \ ___ / _|  / _ \| || |                                                         #
# | |_) / _ \ |_  | | | | || |_                                                        #
# |  _ <  __/  _| | |_| |__   _|                                                       #
# |_| \_\___|_|    \___/   |_|                                                         #
## Learning to optimize: Training deep neural networks for wirelessresource management #
########################################################################################

## Ref 2: 
# K single-antenna transceivers
K = numAntenna

# model 2
l1_mlp_inNodes    = K*K   # guessed number
l234_mlp_inNodes  = 200   # acutal number
l4_mlp_outNodes   = K     # guessed number


models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l234_mlp_inNodes, True),
                        nn.Linear(l234_mlp_inNodes, l234_mlp_inNodes, True),
                        nn.Linear(l234_mlp_inNodes, l234_mlp_inNodes, True),
                        nn.Linear(l234_mlp_inNodes, l4_mlp_outNodes, True))))
info("Model 4 created.")
#################################################################################################
#  ____       __    ___  ____                                                                   #
# |  _ \ ___ / _|  / _ \| ___|                                                                  #
# | |_) / _ \ |_  | | | |___ \                                                                  #
# |  _ <  __/  _| | |_| |___) |                                                                 #
# |_| \_\___|_|    \___/|____/                                                                  #
## A reinforcement learning approach to power control and rate adaptation in cellular networks  #
#################################################################################################
# assembles with different #layers and #neurons
todo("Ref 05: Currently not implemented because of its unknown topology and parameters", Importance.HIGH)
models.append(netModel(nn.Sequential()))
#################################################################################################
#  ____       __    ___   __                                                                    #
# |  _ \ ___ / _|  / _ \ / /_                                                                   #
# | |_) / _ \ |_  | | | | '_ \                                                                  #
# |  _ <  __/  _| | |_| | (_) |                                                                 #
# |_| \_\___|_|    \___/ \___/                                                                  #
## Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks              #
#################################################################################################
M = 20 # length of history
numStates = 2 
l1_mlp_inNodes = 5*M # actual number
l1_mlp_outNodes = 64
l2_mlp_outNodes = 64
l3_mlp_outNodes = 64
l4_mlp_outNodes = 64
l5_mlp_outNodes = 64
l6_mlp_outNodes = numStates
models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l1_mlp_outNodes, True),
                        nn.Linear(l1_mlp_outNodes, l2_mlp_outNodes, True),
                        nn.Linear(l2_mlp_outNodes, l3_mlp_outNodes, True),
                        nn.Linear(l3_mlp_outNodes, l4_mlp_outNodes, True),
                        nn.Linear(l4_mlp_outNodes, l5_mlp_outNodes, True),
                        nn.Linear(l5_mlp_outNodes, l6_mlp_outNodes, True))))

info("Model 6 created.")

#################################################################################################
#  ____       __    ___ _____                                                                   #
# |  _ \ ___ / _|  / _ \___  |                                                                  #
# | |_) / _ \ |_  | | | | / /                                                                   #
# |  _ <  __/  _| | |_| |/ /                                                                    #
# |_| \_\___|_|    \___//_/                                                                     #
## Learning Optimal Resource Allocations in Wireless Systems                                    #
#################################################################################################
m = numAntenna # guessed
l1_mlp_inNodes = m 
l1_mlp_outNodes = 32
l2_mlp_outNodes = 16
l3_mlp_outNodes = m
models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l1_mlp_outNodes, True),
                        nn.Linear(l1_mlp_outNodes, l2_mlp_outNodes, True),
                        nn.Linear(l2_mlp_outNodes, l3_mlp_outNodes, True))))


info("Model 7 created.")

#################################################################################################
#  ____       __    ___   ___                                                                   #
# |  _ \ ___ / _|  / _ \ ( _ )                                                                  #
# | |_) / _ \ |_  | | | |/ _ \                                                                  #
# |  _ <  __/  _| | |_| | (_) |                                                                 #
# |_| \_\___|_|    \___/ \___/                                                                  #
## Deep Reinforcement Learning for Distributed Dynamic Power Allocation in Wireless Networkss   #
#################################################################################################
c = 5 # or numAntenna # number of nodes
l1_mlp_inNodes = 10*c+7 
l1_mlp_outNodes = 200
l2_mlp_outNodes = 100
l3_mlp_outNodes = 40
l4_mlp_outNodes = 10
models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l1_mlp_outNodes, True),
                        nn.Linear(l1_mlp_outNodes, l2_mlp_outNodes, True),
                        nn.Linear(l2_mlp_outNodes, l3_mlp_outNodes, True),
                        nn.Linear(l3_mlp_outNodes, l4_mlp_outNodes, True))))
info("Model 8 created.")

#################################################################################################
#  ____       __    ___   ___                                                                   #
# |  _ \ ___ / _|  / _ \ / _ \                                                                  #
# | |_) / _ \ |_  | | | | (_) |                                                                 #
# |  _ <  __/  _| | |_| |\__, |                                                                 #
# |_| \_\___|_|    \___/   /_/                                                                  #
## Deep Learning for Radio Resource Allocation in Multi-Cell Networks                           #
#################################################################################################
K = numAntenna # #cells
U = None # #users
F = freqBands # #sub-bands

l1_mlp_inNodes = K*K*(F+1)
l1_mlp_outNodes = 1080
l2_mlp_outNodes = 720
l3_mlp_outNodes = 360
l4_mlp_outNodes = 180
#models.append(netModel(nn.Sequential()))
#error('model9 deactivated')

models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l1_mlp_outNodes, True),
                        nn.Linear(l1_mlp_outNodes, l2_mlp_outNodes, True),
                        nn.Linear(l2_mlp_outNodes, l3_mlp_outNodes, True),
                        nn.Linear(l3_mlp_outNodes, l4_mlp_outNodes, True))))

#info("Model 9 created.")
#################################################################################################
#  ____       __   _  ___                                                                       #
# |  _ \ ___ / _| / |/ _ \                                                                      #
# | |_) / _ \ |_  | | | | |                                                                     #
# |  _ <  __/  _| | | |_| |                                                                     #
# |_| \_\___|_|   |_|\___/                                                                      #
## Deep Power Control: Transmit Power Control Scheme b. on CNN                                  #
#################################################################################################
conv_hin = 10 # real number
conv_win = 10 
l1234567_cout = 8
l1234567_kernel = 3
l1234567_strides = 1
l8_mlp_inNodes = conv_hin*conv_win*l1234567_cout;
l8_mlp_outNodes = 10
# models = [];

models.append(netModel(nn.Sequential(
# nn.Conv2d(1,l1234567_cout,l1234567_kernel,l1234567_strides, True),
# nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
# nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
# nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
# nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
# nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
nn.Conv2d(l1234567_cout,l1234567_cout,l1234567_kernel,l1234567_strides, True), 
nn.Linear(l8_mlp_inNodes, l8_mlp_outNodes, True)), conv_hin, conv_win)
)

# models.append(netModel(nn.Sequential()))
info("Model 10 created.")
#################################################################################################
#  ____       __   _ _                                                                          #
# |  _ \ ___ / _| / / |                                                                         #
# | |_) / _ \ |_  | | |                                                                         #
# |  _ <  __/  _| | | |                                                                         #
# |_| \_\___|_|   |_|_|                                                                         #
## Deep reinforcement learning for dynamic multichannel access in wireless networks             #
#################################################################################################

N = 16 # channels
M = 16 # previous actions

l1_mlp_inNodes = M*N*2
l1_mlp_outNodes = 200
l2_mlp_outNodes = 200
l3_mlp_outNodes = N

models.append(netModel(nn.Sequential(nn.Linear(l1_mlp_inNodes, l1_mlp_outNodes, True),
                        nn.Linear(l1_mlp_outNodes, l2_mlp_outNodes, True),
                        nn.Linear(l2_mlp_outNodes, l3_mlp_outNodes, True),
                        nn.Linear(l3_mlp_outNodes, l4_mlp_outNodes, True))))
info("Model 11 created.")

# temp = nn.Conv2d(2,3,3,1, True)
# temp.weight.data[0][0].fill_(1)
# temp.weight.data[0][1].fill_(0)
# temp.weight.data[1].fill_(0)
# temp.weight.data[2].fill_(0)


# temp.bias.data.fill_(0)
# model_conv = netModel(nn.Sequential(temp));
# model_conv.exportModel(4,5)


# for model in models:
netModel.exportModel(models)
# net_test=netModel(nn.Sequential(nn.Linear(8, 8, True),))
# net_test.model[0].bias.data.fill_(0)
# net_test.model[0].weight.data[0][0].fill_(0)

# net_test.model[0].weight.data[0][1].fill_(0)

# net_test.model[0].weight.data[1][0].fill_(0)

# net_test.model[0].weight.data[1][1].fill_(2)

# netModel.exportModel(net_test)


netModel.printReport(models, True)

