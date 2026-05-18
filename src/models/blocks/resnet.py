import torch 
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_sizes: list, strides: list, out_channels: list = None, dropout_probability: float = 0.2, norm_groups: int = None, swish: bool = True):
        """
        Custtomizable ResNet block. Executes x = skip -> Conv(x) -> Norm(x) -> Activation(x) + Conv(skip) -> Dropout(x).

        Args:
        - in_channels: (int) In channels for first convolution
        - kernel_sizes: (list[list]) Kernel sizes for each convolution
        - strides: (list[int]) Strides for each convolution, length must match length of kernel_sizes
        - out_channels: (list[int]) Out channels for each convolution, length must match length of kernel_sizes. Default: Keep channels
        - dropout_probability: (float) Dropout probability
        - norm_groups: (int) Num of groups to use for GroupNorm
        - swish: (bool) Whether to use swish activation, default True, otherwise uses Relu
        
        """
        
        super().__init__()

        self.num_convs = len(kernel_sizes)

        # check if args are valid
        assert self.num_convs > 0, "There must be at least one convolution."
        assert (self.num_convs == len(strides)), "Incompatible args: kernel_sizes must be of the same length as strides."

        # initalize convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_maps = nn.ModuleList()  # 1x1 convs to map residual connections to suitable channel dimension
        self.dropouts = nn.ModuleList()

        # initalize activation func
        if swish:
            # use swish by default
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        if out_channels == None:
            # -> default behavior: keep channels if out_channels is not given
            out_channels = [in_channels for _ in range(self.num_convs)]
        elif type(out_channels) in {tuple, list}:
            assert (self.num_convs == len(out_channels)), "Length of out_channels must match length of kernel_sizes."  # check for validness
            # -> use passed values
        else:
            # -> invalid type
            raise TypeError("Invalid type for out_channels: Either pass a tuple or list of values or None for default behavior.")

        # calculate num of groups for norm if not given, default=16
        if norm_groups is None:
            norm_groups = 16 if in_channels % 16 == 0 else 1

        for i in range(self.num_convs-1):
            # initalize convolution
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=kernel_sizes[i][0] // 2)  # in_channels = out_channels[i-1] if i > 0 else in_channels
            self.convs.append(conv)
            
            # initalize GroupNorm
            norm = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels[i])
            self.norms.append(norm)

            self.skip_maps.append(nn.Conv2d(in_channels, out_channels[i], kernel_size=1, padding=0))

            # init dropout
            self.dropouts.append(nn.Dropout(p=dropout_probability))

            in_channels = out_channels[i]

        # add last convolution where channels are changed
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[-1], kernel_size=kernel_sizes[self.num_convs-1], stride=strides[self.num_convs-1], padding=kernel_sizes[i][0] // 2)
        self.convs.append(conv)

        # add last GroupNorm
        norm = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels[-1])
        self.norms.append(norm)
        
        self.skip_maps.append(nn.Conv2d(in_channels, out_channels[-1], kernel_size=1, padding=0))
        
        self.dropouts.append(nn.Dropout(p=dropout_probability))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # do first computation outside of loop to avoid copy operation to preserve residual connection
        x1 = self.dropouts[0](self.activation(self.norms[0](self.convs[0](X))) + self.skip_maps[0](X)) 

        for i in range(1, self.num_convs):
            x1 = self.dropouts[i](self.activation(self.norms[i](self.convs[i](x1))) + self.skip_maps[i](x1))

        return x1
