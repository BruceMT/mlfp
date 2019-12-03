my_dict ={"java":100, "python":112, "c":11} 
  
# list out keys and values separately 
key_list = list(my_dict.keys()) 
val_list = list(my_dict.values()) 
  

# one-liner 
print(list(my_dict.keys())[list(my_dict.values()).index()])