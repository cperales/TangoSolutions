import torch
from torch import nn

class BoardLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device

    def forward(self, x):
        loss = torch.zeros(1, device=self.device)
        # Apply threshold while keeping gradients
        binary_output = torch.where(x > 0.5, 
                                    torch.ones_like(x),
                                    torch.zeros_like(x))
        batch_size = binary_output.shape[0]
        for n in range(batch_size):
            b = binary_output[n]
            loss += self.cons_row(b) / 6.0
            loss += self.cons_col(b) / 6.0
        
        loss /= batch_size
        return loss


    def cons_col(self, board):
        loss = torch.zeros(1, device=self.device)
        for j in range(board.shape[0]):
            prev_cell = board[0, j]
            num = 1
            for cell in board[1:, j]:
                if prev_cell == cell:
                    num += 1
                    if num == 3:
                        loss += torch.ones(1, device=self.device)
                        break
                else:
                    num = 1
                prev_cell = cell

        return loss
    
    def cons_row(self, board):
        loss = torch.zeros(1, device=self.device)
        for i in range(board.shape[1]):
            prev_cell = board[i, 0]
            num = 1
            for cell in board[i, 1:]:
                if prev_cell == cell:
                    num += 1
                    if num == 3:
                        loss += torch.ones(1, device=self.device)
                        break
                else:
                    num = 1
                prev_cell = cell

        return loss
