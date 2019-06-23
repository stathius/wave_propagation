import torch


def convert_BSHW_to_SBCHW(x):
    """
    input : x torch tensor
    output: x torch tensor
    Converts a torch tensor given in dimensions:
    Batch_size * Sequence_Length * Height * Width (B S H W)
    to
    Sequence_Length * Batch_size * Input_Channels  * Height * Width (S B C H W)
    """
    assert len(x.shape) == 4

    x = x.unsqueeze(0)
    x = x.permute(2,1,0,3,4)
    return x

def convert_SBCHW_to_BSHW(x):
    """
    input : x torch tensor
    output: x torch tensor
    Converts a torch tensor given in dimensions:
    Sequence_Length * Batch_size * Input_Channels  * Height * Width (S B C H W)
    to
    Batch_size * Sequence_Length * Height * Width (B S H W)
    """    
    assert len(x.shape) == 5

    x = x.squeeze()
    x = x.permute(1,0,2,3)

    return x