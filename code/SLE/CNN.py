import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 通过输入7个维度的传感器数据回归步长，每一步15个采样点，3轴加速度、3轴角速度、时间间隔，共7通道

        self.conv1 = nn.Sequential(
            
            nn.Conv1d(in_channels=7, out_channels=10, kernel_size=13, stride=1),  #输入7个通道，15个采样点，7*15，10个大小为1*13的卷积核
            nn.ReLU(), #激活函数relu
            nn.MaxPool1d(kernel_size=2,stride=1),  # 10*3 -> 10*2
        )
        '''
        self.conv2= nn.Sequential(

            nn.Conv1d(in_channels=10, out_channels=15, kernel_size=3, stride=1), #输入10个通道，12个点，10*12，15个1*3的卷积核
            nn.ReLU(), #激活函数relu
            nn.MaxPool1d(kernel_size=2,stride=1),  # 上面输出15*10数据，最大池化 ----> 15*9
        )
        '''
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=10*2, out_features=1, bias=True), #全连接层，将展平成一维的135个数据输入，输出回归步长结果
            nn.Sigmoid(),
        )
            
         
        
    def forward(self, input):
        x = self.conv1(input)
        #x = self.conv2(x)
        y = self.linear(x)
        return y

if __name__ == "__main__":
    cnn = CNN()
    print(cnn)


