import torch
from torch import nn

class BoardLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.one = torch.ones(torch.Size([]), device=self.device)

    def forward(self, x):
        # Apply threshold while keeping gradients
        binary_output = torch.where(x > 0.5, 
                                    torch.ones_like(x),
                                    torch.zeros_like(x))
        
        # Row constraint loss - each row should have exactly 3 suns
        row_sum = binary_output.sum(dim=2)  # Sum across each row
        loss = ((row_sum - 3) ** 2).mean()  # Should be 3 suns per row
        
        # Column constraint loss - each column should have exactly 3 suns
        col_sum = binary_output.sum(dim=1)  # Sum across each column
        loss += ((col_sum - 3) ** 2).mean()  # Should be 3 suns per column
        
        # Constraints: no more than 2 suns or moons together
        batch_size = binary_output.shape[0]
        [(self.cons_row(binary_output[n], loss, batch_size), self.cons_col)
         for n in range(batch_size)]
        return loss


    def cons_col(self, board, loss, batch_size):
        for j in range(board.shape[0]):
            prev_cell = board[0, j]
            num = 1
            for cell in board[1:, j]:
                if prev_cell == cell:
                    num += 1
                    if num == 3:
                        loss += self.one / ( 6 * batch_size)
                        break
                else:
                    num = 1
                prev_cell = cell

        return loss
    
    def cons_row(self, board, loss, batch_size):
        for i in range(board.shape[1]):
            prev_cell = board[i, 0]
            num = 1
            for cell in board[i, 1:]:
                if prev_cell == cell:
                    num += 1
                    if num == 3:
                        loss += self.one / ( 6 * batch_size)
                        break
                else:
                    num = 1
                prev_cell = cell

        return loss
