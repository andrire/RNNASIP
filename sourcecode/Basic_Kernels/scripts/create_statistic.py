# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
import re
import sys
from collections import deque
import pickle

start_token = 'Start';
if len(sys.argv) > 2:
   start_token = sys.argv[2];
log_file_name = sys.argv[1]
log_file = open(log_file_name)

start_found = 0
histo_dict = {}
instr_groups = {}
total = 0

instrCntStart = 0
instrCntEnd = 0  #todo fix this impl.

MININSTR = 1000
MAXLENGTH = 6

MINCYCLESHOW = 1
TOPXSHOW = 100

ID_FUNC = 4

print("""Usage create_statistics.py TRACE_FILE START_STRING
   """)

#addressing
instr_groups["lui"] = 'init' # sets the upper 20 bits (immediate) to register       lui rd, Immediate
instr_groups["c.mv"] = 'mv' # move data from register to register                     mv rs, rd
instr_groups["c.beqz"] = 'jump' # branch equal                     mv rs, rd
instr_groups["addi"] = 'add'
instr_groups["lw"] = 'load'

class memoryElement():
   def __init__(self, name):  
      self.memory = name
      self.address = 0
      self.accSize = 0
      self.writes = 0
      self.reads = 0
      self.access = {}

class myDeque(deque):
   def __repr__(self):
      temp =list(self)
      return temp.__repr__()

class instrOrderTracker():
   def __init__(self, num):
      self.num = num
      self.instrs = []
      for i in range(0, num):
         self.instrs.append(myDeque(["nop"]*(i+1)))
      # self.instrs = deque()
   def addNewestRmLeast(self, instr):
      for i in range(0, self.num):
         self.instrs[i].append(instr)
      for i in range(0, min(self.num, len(self.instrs[self.num-1])-1)):
         tmp = self.instrs[i].popleft()
      # if len(self.instrs) > self.num:
      #    tmp = self.instrs.pop()
   def __str__(self):
      return str(list(self.instrs))
   def getInstrs(self):
      temp = list()
      # for i in range(0, self.num):
      #    if i+1 == len(self.instrs[i]):
      #       temp.append(self.instrs[i])
      # return temp
      return list(self.instrs)

def dict_acc(dict_name, element, count):
  if element in dict_name:
         dict_name[element] += count
  else:
         dict_name[element] = count
def dict_mergeAcc(list_of_dicts):
    tmp = {}
    for _dict in list_of_dicts:
        for key in list(list_of_dicts[_dict].keys()):
            dict_acc(tmp, key, list_of_dicts[_dict][key])
    return tmp

def dict_createIfNexist(dict_name, key, value):
  if not(key in dict_name):
      dict_name[key] = value

def dict_sum(dict_name):
  acc = 0
  for key in dict_name:
    acc += dict_name.get(key)
  return acc


instrPerFunc = {}
cyclesPerFunc = {}

memList = {}
topLevel_only = True
topLevelFuncList = ["main", "inferNetwork", "LinearLayer", "Conv2dLayer", "LSTMLayer", "TwoLinearLayersAccumulate", "SigLayer", "TanhLayer", "HadMulTensor", "AddTensor", "CopyTensor" ];

countForTopLevelFunc = {}
# dict in which functions it also should be counter (TODO: Please be aware, that this does not work if the same function is executed by several functions.)
countForTopLevelFunc["inferNetwork"] = ["main"]
secondStage = ["Conv2dLayer","LinearLayer", "LSTMLayer"];
secondStageTop = ["inferNetwork", "main"]
for i in range(0, len(secondStage)):
  countForTopLevelFunc[secondStage[i]] = secondStageTop

thirdStageTop = ["main", "inferNetwork", "LSTMLayer"]
thirdStage = ["TwoLinearLayersAccumulate", "SigLayer", "TanhLayer", "HadMulTensor", "AddTensor", "CopyTensor" ];
for i in range(0, len(thirdStage)):
  countForTopLevelFunc[thirdStage[i]] = thirdStageTop

#showFuncs = list(["main", "inferNetwork", "LSTMLayer", "LinearLayer", "RNNLayer", "Conv2dLayer", "SigLayer", "TanhLayer", ]);


print("Statistics for "+str(start_token))
instrs = instrOrderTracker(MAXLENGTH) # p.lh', 'p.lh', 'mul', 'c.srai', 'c.add', 'p.exths'
line_next = log_file.readline()
line_split_next=line_next.split()
currTopLevelFunc = "main"
while True:
   line = line_next
   line_split=line_split_next
   # print((line_split))
   if len(line_split) == 0 and start_found == 1:
      break;
   if len(line_split) == 0 and start_found == 0:
      line = log_file.readline()
      line_split=line.split()
      if len(line_split) == 0:
         break
   if line_split[0]==start_token and start_found == 0:
      start_found=1
      line = log_file.readline()
      line_split=line.split()
      instrCntStart = int(line_split[1][:-1])
      
   line_next = log_file.readline()
   line_split_next=line_next.split()
   
   if len(line_split_next) == 0 and start_found == 1:
      
      line_next = log_file.readline()
      line_split_next=line_next.split()
      if (len(line_split_next) == 0):
        instrCntEnd = instrCntStart;
      else:
        instrCntEnd = int(line_split_next[1][:-1])
      break;
   while(len(line_split_next) < 5 and start_found == 1):
       line_next = log_file.readline()
       line_split_next=line_next.split()
       if len(line_split_next) == 0:
           break;
   # elif line_split[0]==finish_token:
   #    break;
   if start_found == 0:
      continue;
   if len(line_split)>5:
      # print(line_split)
      if re.compile(".*insn.*").match(line_split[2]): # it is a normal instruction
         # instrCntEnd = int(line_split[1][:-1])
         instr = line_split[7]
         if line_split[9].find('!') != -1:
           instr += '!'
         if re.compile("pl\.sdotsp\.h\..*").match(instr):
            instr = "pl.sdotsp.h"
         instrs.addNewestRmLeast(instr)
         total = total + 1
         for instr_comb in instrs.getInstrs(): 
            # print(instr_comb)
            if str(instr_comb) in histo_dict:
               histo_dict[str(instr_comb)] += 1
            else:
               histo_dict[str(instr_comb)] = 1
            # print ("%12s: %6i" % (str(instr_comb), histo_dict[str(instr_comb)]))
         tmp = line_split[ID_FUNC].find(':') # remove line number
         curr_func =line_split[ID_FUNC][:tmp]
         if topLevel_only:
             if curr_func in topLevelFuncList:
                 currTopLevelFunc = curr_func
         else:
           currTopLevelFunc = curr_func #for all functions
         
         dict_createIfNexist(instrPerFunc, currTopLevelFunc, {})
         dict_createIfNexist(cyclesPerFunc, currTopLevelFunc, {})
         dict_acc(instrPerFunc[currTopLevelFunc], instr, 1)
         #print(line_split_next[1][:-1])
         #print(line_split[1][:-1])
         try:
            dict_acc(cyclesPerFunc[currTopLevelFunc], instr, int(line_split_next[1][:-1])-int(line_split[1][:-1]))
            assert(int(line_split_next[1][:-1])-int(line_split[1][:-1])>0)

         except Exception as e:
            print(str(e)+"\n LineSplit is :")
            print(line_split_next)


         for topFuncs in countForTopLevelFunc.get(currTopLevelFunc, {}):
           dict_acc(instrPerFunc[topFuncs], instr, 1)
           dict_acc(cyclesPerFunc[topFuncs], instr, int(line_split_next[1][:-1])-int(line_split[1][:-1]))

         continue;
      else:
         if re.compile(".*Memory.*").match(line_split[4]):
            memory = line_split[2][26:-6]
            address = int(line_split[7][:-1], 16)
            accSize = int(line_split[9][:-1], 16)
            writeOrRead = int(line_split[11][:-1])
            if not(memory in memList):
               memList[memory] = memoryElement(memory)
            memList[memory].accSize += accSize;
            memList[memory].writes += writeOrRead;
            memList[memory].reads += (1-writeOrRead)     
            if not(address in  memList[memory].access):
               memList[memory].access[address] = 1
            else:
               memList[memory].access[address] += 1

            # print(accSize)
            # print(line_split)
         pass
   else:
      continue;
# print(histo_dict)
log_file.close();
if True:
    for key in sorted(histo_dict, key=histo_dict.__getitem__, reverse=True):
        if histo_dict[key] >= MININSTR:
            print ("%2i, %12s: %6i" % (key.count(',')+1, key, histo_dict[key]))
    print(\
    """--------------------
    %12s: %6i
    --------------------""" % ("total",  total))

for key in memList:
    print ("%12s: %6i, %6i" % (key, memList[key].reads, memList[key].writes))
    # print(memList[key].access)


print("{:>12}" .format("Instr."), end='') #{:^16}
for i in topLevelFuncList: #list(cyclesPerFunc.keys()):
  print("{:^16.16}".format(i), end='')
print()

col_format = "{:>8}{:>8}"
print("{:>12}".format("Instr."), end='')


for i in range(0, len(topLevelFuncList)):
  print((col_format).format("cycles", "instrs"), end='')
print()

instr_total = dict_mergeAcc(instrPerFunc)
cycles_total = dict_mergeAcc(cyclesPerFunc)
i = 0
for key in sorted(cycles_total, key=cycles_total.__getitem__, reverse=True):
  if i >= TOPXSHOW:
    break
  i=i+1
  if (cyclesPerFunc.get('main', {}).get(key,0))<MINCYCLESHOW:
    continue
  print(("{:>12}").format(key), end='')
  for element in topLevelFuncList: #list(cyclesPerFunc.keys()):
      print(col_format.format(cyclesPerFunc.get(element, {}).get(key,0), instrPerFunc.get(element, {}).get(key, 0)), end='')
  #print(col_format.format(cyclesPerFunc.get("LinearLayer", {}).get(key,0), instrPerFunc.get("LinearLayer", {}).get(key,0)), end='')
  #print(col_format.format(cyclesPerFunc.get("LSTMLayer", {}).get(key,0), instrPerFunc.get("LSTMLayer", {}).get(key,0)), end='')
  
  print()

print(("{:-<"+str(12+16*(len(topLevelFuncList)))+"}").format(''))
print(("{:>12}").format("sum"), end='')
#print(col_format.format(dict_sum(cyclesPerFunc.get("Conv2dLayer", {})), dict_sum(instrPerFunc.get("Conv2dLayer", {}))), end='')
#print(col_format.format(dict_sum(cyclesPerFunc.get("LinearLayer", {})), dict_sum(instrPerFunc.get("LinearLayer", {}))), end='')
#print(col_format.format(dict_sum(cyclesPerFunc.get("LSTMLayer", {})), dict_sum(instrPerFunc.get("LSTMLayer", {}))), end='')
for element in topLevelFuncList: #list(cyclesPerFunc.keys()):
      print(col_format.format(dict_sum(cyclesPerFunc.get(element, {})), dict_sum(instrPerFunc.get(element, {}))), end='')


print() 
print(("{:-<"+str(12+16*(len(topLevelFuncList)))+"}").format(''))
print("{:>12}" .format("Instr."), end='')
for i in topLevelFuncList: #list(cyclesPerFunc.keys()):
  print("{:^16.16}".format(i), end='')
print()


print("Start: {}, End: {}, Duration in cycles: {}".format(instrCntStart, instrCntEnd-1, instrCntEnd-instrCntStart-1))

pickle_dump = {}
pickle_dump["topLevelFuncList"] = topLevelFuncList;
pickle_dump["cyclesPerFunc"] = cyclesPerFunc;
pickle_dump["instrPerFunc"] = instrPerFunc;

# json.dump(json_dump, open(log_file_name+".json", "w+"))

with open(log_file_name+".pkl", 'wb+') as f:                                                                                                                                                                                                                          
  pickle.dump(pickle_dump, f, pickle.HIGHEST_PROTOCOL)
