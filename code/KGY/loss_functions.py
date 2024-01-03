import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSE
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
# Huber Loss
class HuberLoss(nn.Module):
    def __init__(self, delta = 1.0):
        super().__init__()
        self.delta = delta # 임계값
        self.mse = nn.MSELoss(reduction = 'mean')
        self.l1 = nn.L1Loss(reduction = 'mean')
        
    def forward(self,yhat,y):
        residual = torch.abs(y-yhat)
        condition = (residual < self.delta).float()
        
        loss = condition*0.5*self.mse(yhat,y) + (1-condition)*self.delta*(residual-0.5*self.delta)
        return torch.mean(loss)
# pytorch의 F.smooth_l1_loss()가 Huber Loss 이용        

# Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,yhat,y):
        residual = yhat - y
        log_cosh = residual + torch.nn.functional.softplus(-2.0*residual) - torch.log(2.0)
        return torch.mean(log_cosh)
    
# Quantile loss
class QuantileLoss(nn.Module):
    def __init__(self,quantile = 0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self,yhat,y):
        residual = yhat - y
        quantile_loss = torch.max((self.quantile - 1) * residual, self.quantile*residual)
        return torch.mean(quantile_loss)

    