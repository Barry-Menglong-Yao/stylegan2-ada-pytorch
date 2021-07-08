import torch
import torch.nn as nn 
def test_1():
    real_img = torch.randn(2, 2,2, requires_grad=True)
    reconstructed_img = torch.randn(2, 2,2)
    loss = torch.nn.MSELoss(reduction='none')
    loss_Emain_reconstruct = loss(reconstructed_img, real_img)
    print(loss_Emain_reconstruct)
    loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
    print(loss_Emain_reconstruct)
    loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
    print(loss_Emain_reconstruct)
    
    loss2 = torch.nn.MSELoss( )
    loss_Emain_reconstruct2 = loss2(reconstructed_img, real_img)
    print(loss_Emain_reconstruct2)
    print(" ")

def test_hook():
    import torch 
    a = torch.ones(5)
    a.requires_grad = True

    b = 2*a

    b.retain_grad()   # Since b is non-leaf and it's grad will be destroyed otherwise.

    c = b.mean()

    c.backward()

    print(a.grad, b.grad)

    # Redo the experiment but with a hook that multiplies b's grad by 2. 
    a = torch.ones(5)

    a.requires_grad = True

    b = 2*a

    # b.retain_grad()

    b.register_hook(lambda x: print(x))  

    b.mean().backward() 


    print(a.grad, b.grad)

def test_hook2():
    import torch 
    import torch.nn as nn

    class myNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3,10,2, stride = 2)
            self.relu = nn.ReLU()
            self.flatten = lambda x: x.view(-1)
            self.fc1 = nn.Linear(160,5)
        
        
        def forward(self, x):
            x = self.relu(self.conv(x))
            return self.fc1(self.flatten(x))
    

    net = myNet()

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

    net.conv.register_backward_hook(hook_fn)
    net.fc1.register_backward_hook(hook_fn)
    inp = torch.randn(1,3,8,8)
    out = net(inp)

    (1 - out.mean()).backward()

def test_conv():
    # With square kernels and equal stride
    m = nn.Conv2d(3, 32,   kernel_size= 3, stride= 2, padding  = 1)
    # # non-square kernels and unequal stride and with padding
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    # # non-square kernels and unequal stride and with padding and dilation
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    input = torch.randn(32, 3, 32, 32)
    output = m(input)
    print(output.shape)

if __name__ == '__main__':
    test_conv()