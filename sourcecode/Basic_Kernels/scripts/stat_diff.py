# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
import pickle
import sys 

def load_obj(file ):
     with open(file, 'rb') as f:
         return pickle.load(f)

class func():
  def __init__(self, name, funcList):  
    self.name  = name
    self.funcList = funcList
    self.cycles = 0
    self.instrs = 0
    self.instr_dict = {}
    self.cycle_dict = {}
  def __str__(self):
    return "%12s %7i %7i" % (self.name, self.cycles, self.instrs)
def dict_mergeAcc(list_of_dicts):
    tmp = {}
    for _dict in list_of_dicts:
        for key in list(list_of_dicts[_dict].keys()):
            dict_acc(tmp, key, list_of_dicts[_dict][key])
    return tmp
def dict_sum(dict_name):
  acc = 0
  for key in dict_name:
    acc += dict_name.get(key)
  return acc    
def dict_acc(dict_name, element, count):
  if element in dict_name:
         dict_name[element] += count
  else:
         dict_name[element] = count


riscv_data = load_obj(sys.argv[1])
tzscale_data = load_obj(sys.argv[2])

topLevelFuncList = riscv_data["topLevelFuncList"];
cyclesPerFunc = riscv_data["cyclesPerFunc"];
instrPerFunc = riscv_data["instrPerFunc"];
tzscale_listOfLayers = tzscale_data["listOfLayers"];
tzscale_cycle_dict = tzscale_data["cycle_dict"];
tzscale_instr_dict = tzscale_data["instr_dict"];

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

for key in sorted(instr_total, key=instr_total.__getitem__, reverse=True):
  print(("{:>12}").format(key), end='')
  for element in topLevelFuncList: #list(cyclesPerFunc.keys()):
      #for tzscale_corresp_element in tzscale_listOfLayers:
         # if(tzscale_corresp_element.name == element):
           print(col_format.format(cyclesPerFunc.get(element, {}).get(key,0), instrPerFunc.get(element, {}).get(key, 0)), end='') #int(tzscale_corresp_element.cycle_dict.get(key,0))
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

tzscale_listOfLayers = tzscale_data["listOfLayers"];
tzscale_cycle_dict = tzscale_data["cycle_dict"];
tzscale_instr_dict = tzscale_data["instr_dict"];

print("{:>12}{:^16}" .format("Instr.", "Total"), end='')
for i in tzscale_listOfLayers: #list(cyclesPerFunc.keys()):
  print("{:^16.16}".format(i.name), end='')
print()

# pickle_dump = {}


#print(("{:>12}"+col_format+col_format+col_format+col_format).format("Instr.", "cycles", "instrs", "cycles", "instrs", "cycles", "instrs", "cycles", "instrs"))
col_format = "{:>8}{:>8}"
print("{:>12}".format("Instr."), end='')
for i in range(0, 1+len(tzscale_listOfLayers)):
  print((col_format).format("cycles", "instrs"), end='')
print()

print(("{:-<"+str(12+16*(1+len(tzscale_listOfLayers)))+"}").format(''))
for key in sorted(tzscale_cycle_dict, key=tzscale_cycle_dict.__getitem__, reverse=True):
  print(("{:>12}"+col_format).format(key, tzscale_cycle_dict.get(key,0), tzscale_instr_dict.get(key, 0)), end='')
  for element in tzscale_listOfLayers: #list(cyclesPerFunc.keys()):
      print(col_format.format(element.cycle_dict.get(key,0), element.instr_dict.get(key, 0)), end='')

#  print(col_format.format(conv2D.cycle_dict.get(key,0), conv2D.instr_dict.get(key, 0)), end='')
#  print(col_format.format(linearFunc.cycle_dict.get(key,0), linearFunc.instr_dict.get(key, 0)), end='')
#  print(col_format.format(lstm.cycle_dict.get(key,0), lstm.instr_dict.get(key, 0)), end='')
  print()
print(("{:-<"+str(12+16*(1+len(tzscale_listOfLayers)))+"}").format(''))
print(("{:>12}"+col_format).format("sum", dict_sum(tzscale_cycle_dict), dict_sum(tzscale_instr_dict)), end='')
for element in tzscale_listOfLayers: #list(cyclesPerFunc.keys()):
  print(col_format.format(dict_sum(element.cycle_dict), dict_sum(element.instr_dict)), end='')
#print(col_format.format(dict_sum(conv2D.cycle_dict), dict_sum(conv2D.instr_dict)), end='')
#print(col_format.format(dict_sum(linearFunc.cycle_dict), dict_sum(linearFunc.instr_dict)), end='')
#print(col_format.format(dict_sum(lstm.cycle_dict), dict_sum(lstm.instr_dict)), end='')
print() 
print(("{:-<"+str(12+16*(1+len(tzscale_listOfLayers)))+"}").format(''))
   


# store data conveniently
