def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:  
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")
    print("\n")