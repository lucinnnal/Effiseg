import torch
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, num_classes=20):
        super().__init__()

        self.weight = torch.ones(num_classes)
        self.weight[0] = 2.5959737
        self.weight[1] = 6.741505
        self.weight[2] = 3.5353868
        self.weight[3] = 9.866315
        self.weight[4] = 9.690922
        self.weight[5] = 9.369371
        self.weight[6] = 10.289124 
        self.weight[7] = 9.953209
        self.weight[8] = 4.3098087
        self.weight[9] = 9.490392
        self.weight[10] = 7.674411
        self.weight[11] = 9.396925	
        self.weight[12] = 10.347794 	
        self.weight[13] = 6.3928986
        self.weight[14] = 10.226673 	
        self.weight[15] = 10.241072	
        self.weight[16] = 10.28059
        self.weight[17] = 10.396977
        self.weight[18] = 10.05567	
        self.weight[19] = 0

        self.loss = torch.nn.NLLLoss(self.weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs['logits'], dim=1), targets)