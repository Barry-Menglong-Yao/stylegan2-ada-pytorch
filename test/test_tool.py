import torch
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


if __name__ == '__main__':
    test_1()