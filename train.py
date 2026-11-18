import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from game import load_solution, TangoBoard
from model import TangoCNN, TangoTransformer, EnsembleCNN
from copy import deepcopy
from loss import BoardLoss
from transformer_chatgpt import VisionTransformer as ChatGPTTransformer
from transformer_claude import ImageTransformer as ClaudeTransformer
from transformer_deepseek import ImageTransformer as DeepseekTransformer


model_dict = {'cnn': TangoCNN,
              'transformer': TangoTransformer,
              'ensemble': EnsembleCNN,
              'chatgpt': ChatGPTTransformer,
              'claude': ClaudeTransformer,
              'deepseek': DeepseekTransformer}


class TangoDataset(Dataset):
    def __init__(self, sol_filenames, num_initial_values=12):
        """
        Dataset for Tango board training
        
        Args:
            solutions: List of 6x6 numpy arrays with solution boards
            num_initial_values: Number of values to expose as input
            transform: Optional transforms to apply
        """
        self.sol_filenames = sol_filenames
        self.num_initial_values = num_initial_values
        self.solutions = dict()

    def __len__(self):
        return len(self.sol_filenames)
    
    def get_solution(self, idx):
        sol_filename = self.sol_filenames[idx]
        if not sol_filename in self.solutions.keys():
            self.solutions[sol_filename] = load_solution(sol_filename).board.astype(np.float32)
        return self.solutions[sol_filename]

    def __getitem__(self, idx):
        solution = self.get_solution(idx)
        
        # Create a mask where True means the position is exposed
        mask = np.zeros((6, 6), dtype=bool)
        indices = np.random.choice(36, self.num_initial_values, replace=False)
        mask.flat[indices] = True
        
        # Create input where:
        # Channel 0: 1 where sun is exposed, 0 elsewhere
        # Channel 1: 1 where moon is exposed, 0 elsewhere
        # Channel 2: 1 where position is hidden, 0 elsewhere
        input_tensor = np.zeros((3, 6, 6), dtype=np.float32)
        
        # Channel 0: Exposed suns
        input_tensor[0][np.logical_and(mask, solution == 1)] = 1.0
        
        # Channel 1: Exposed moons
        input_tensor[1][np.logical_and(mask, solution == 0)] = 1.0
        
        # Channel 2: Hidden positions
        input_tensor[2][~mask] = 1.0
        
        # # Target: 1 for Sun, 0 for Moon
        # target = solution.astype(np.float32)
        target = solution
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
            
        return input_tensor, target


class MinTangoDataset(Dataset):
    def __init__(self, data_folder, target_folder, index):
        """
        Dataset for Tango board training
        
        Args:
            solutions: List of 6x6 numpy arrays with solution boards
            num_initial_values: Number of values to expose as input
            transform: Optional transforms to apply
        """
        filenames = np.array([os.path.join(data_folder, f) for f in os.listdir(data_folder)])
        self.data_filenames = filenames[index]
        self.target_filenames = np.array([f.replace(data_folder, target_folder) for f in filenames])[index]
        assert len(self.data_filenames) == len(self.target_filenames)
        self.solutions = dict()

    def __len__(self):
        return len(self.data_filenames)
    
    def get_solution(self, idx):
        data_file = self.data_filenames[idx]
        target_file = self.target_filenames[idx]
        if not idx in self.solutions.keys():
            self.solutions[idx] = dict()
            self.solutions[idx]['data'] = load_solution(data_file).board.astype(np.float32)
            self.solutions[idx]['target'] = load_solution(target_file).board.astype(np.float32)
        return self.solutions[idx]

    def __getitem__(self, idx):
        solution_dict = self.get_solution(idx)
        data = solution_dict['data']
        
        # Create input where:
        # Channel 0: 1 where sun is exposed, 0 elsewhere
        # Channel 1: 1 where moon is exposed, 0 elsewhere
        # Channel 2: 1 where position is hidden, 0 elsewhere
        input_tensor = np.zeros((3, 6, 6), dtype=np.float32)
                
        # Channel 0: Exposed suns
        input_tensor[0][data == 1] = 1.0
        
        # Channel 1: Exposed moons
        input_tensor[1][data == 0] = 1.0
        
        # Channel 2: Hidden positions
        input_tensor[2][data == -1] = 1.0
        
        # # Target: 1 for Sun, 0 for Moon
        # target = solution.astype(np.float32)
        target = solution_dict['target']
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
            
        return input_tensor, target


def train_model(model, train_loader, val_loader, criterion, optimizer_lr=10**-3, num_epochs=50, device='cuda'):
    """Train the CNN model"""
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)

    # Weights for different components of the loss
    alpha = .1      # Weight for standard BCE loss
    beta = .9       # Weight for board constraint loss
    best_val_loss = torch.inf
    board_criterion = BoardLoss(device=device)
    stable_counter = 0
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)

                # Standard BCE loss
                bce_loss = criterion(outputs, targets)

                # Board loss - no consecutive 3 suns or moons
                board_constraint_loss = board_criterion(outputs)
                
                # Combined loss
                loss = alpha * bce_loss + beta * board_constraint_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_cells = 0
            total_cells = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    bce_loss = criterion(outputs, targets)

                    # Board loss - no consecutive 3 suns or moons
                    board_constraint_loss = board_criterion(outputs)
                    
                    # Combined loss
                    loss = alpha * bce_loss + beta * board_constraint_loss
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    correct_cells += (predicted == targets).sum().item()
                    total_cells += targets.numel()

                    # Calculate board accuracy
                    correct_boards = 0
                    batch_size = targets.size(0)
                    for i in range(batch_size):
                        tango_board_predicted = TangoBoard()
                        tango_board_predicted.fulfill(board=predicted[i].cpu())
                        if tango_board_predicted.check():
                            correct_boards += 1

            board_accuracy = correct_boards / batch_size
            
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct_cells / total_cells
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_accuracy)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model = deepcopy(model)
                best_model.to('cpu')
                print(f"Selected best model from epoch {epoch+1}")
            else:
                stable_counter += 1
                if stable_counter == 5:
                    stable_counter = 0
                    optimizer_lr /= 2
                    print(f"Optimizer reduced to {optimizer_lr}")
                    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - '
                  f'Val Loss: {val_epoch_loss:.4f} - Val Board Accuracy: {board_accuracy:.4f}')
    except KeyboardInterrupt:
        pass

    best_model.to(device)
    return history, best_model


def test_model(model, test_loader, device='cuda'):
    """Test the model on a separate test set"""
    model.eval()
    correct_cells = 0
    total_cells = 0
    correct_boards = 0
    total_boards = 0
    if isinstance(model, EnsembleCNN):
        counter = list()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if isinstance(model, EnsembleCNN):
                alpha = model.moe(inputs)
                decision = alpha.argmax(dim=1)
                counter.append(decision.cpu())
                outputs = model(inputs, alpha)
            else:
                outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            # Accuracy per cell
            correct_cells += (predicted == targets).sum().item()
            total_cells += targets.numel()
            
            # Accuracy for complete boards (all cells correct)
            batch_size = targets.size(0)
            for i in range(batch_size):
                tango_board_target = TangoBoard()
                tango_board_target.fulfill(board=targets[i].cpu())
                if tango_board_target.check():
                    tango_board_predicted = TangoBoard()
                    tango_board_predicted.fulfill(board=predicted[i].cpu())
                    if tango_board_predicted.check():
                        correct_boards += 1
                else:
                    if (predicted[i] == targets[i]).all().item():
                        correct_boards += 1
            total_boards += batch_size
    
    cell_accuracy = correct_cells / total_cells
    board_accuracy = correct_boards / total_boards
    
    print(f'Test Cell Accuracy: {cell_accuracy:.4f}')
    print(f'Test Board Accuracy: {board_accuracy:.4f}')

    if isinstance(model, EnsembleCNN):
        sum_counter = np.concatenate(counter)
        plt.hist(sum_counter, bins=np.arange(5) + 1)
        plt.savefig('images/hist_moe.png')
        plt.close()
    
    return cell_accuracy, board_accuracy


def visualize_predictions(model, test_loader, num_examples=5, device='cuda', initial_values=None):
    """Visualize some predictions"""
    model.eval()
    
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 15))
    if initial_values:
        fig.suptitle(f'initial_values={initial_values}\n')
    
    for i in range(min(num_examples, inputs.size(0))):
        # Plot input
        input_board = np.zeros((6, 6, 3))
        for r in range(6):
            for c in range(6):
                if inputs[i, 2, r, c] > 0.5:  # Hidden position
                    input_board[r, c] = [0.5, 0.5, 0.5]  # Gray
                elif inputs[i, 0, r, c] > 0.5:  # Sun
                    input_board[r, c] = [1.0, 1.0, 0.0]  # Yellow
                elif inputs[i, 1, r, c] > 0.5:  # Moon
                    input_board[r, c] = [0.0, 0.0, 1.0]  # Blue
        
        axes[i, 0].imshow(input_board)
        axes[i, 0].set_title('Input (Partial Board)')
        axes[i, 0].axis('off')
        
        # Plot target
        target_board = np.zeros((6, 6, 3))
        for r in range(6):
            for c in range(6):
                if targets[i, r, c] > 0.5:  # Sun
                    target_board[r, c] = [1.0, 1.0, 0.0]  # Yellow
                else:  # Moon
                    target_board[r, c] = [0.0, 0.0, 1.0]  # Blue
        
        axes[i, 1].imshow(target_board)
        axes[i, 1].set_title('Target (Solution)')
        axes[i, 1].axis('off')
        
        # Plot prediction
        pred_board = np.zeros((6, 6, 3))
        for r in range(6):
            for c in range(6):
                if predicted[i, r, c] > 0.5:  # Sun
                    pred_board[r, c] = [1.0, 1.0, 0.0]  # Yellow
                else:  # Moon
                    pred_board[r, c] = [0.0, 0.0, 1.0]  # Blue
        
        axes[i, 2].imshow(pred_board)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('images/tango_' + model.name + '_predictions.png')
    plt.show()


def main_min(data_folder='data', target_folder='target', model_type='cnn', epochs=50, resume=False):
    data_filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    target_filenames = [os.path.join(target_folder, f) for f in os.listdir(target_folder)]
    assert len(data_filenames) == len(target_filenames)
    
    print(f"{len(data_filenames)} Tango board solutions found")
    
    # Convert solutions format if needed (assuming 1 for Sun, 0 for Moon)
    # If your solutions use different values, adjust accordingly
    
    # Split data into train, validation, test sets
    index = np.arange(len(data_filenames))
    train_index, _ = train_test_split(index, test_size=0.25, random_state=42)

    train_index, val_index = train_test_split(train_index, test_size=0.25, random_state=42)
    
    print(f"Train set: {len(train_index)}, Validation set: {len(val_index)}")
    
    # Create datasets
    train_dataset = MinTangoDataset(data_folder, target_folder, train_index)
    val_dataset = MinTangoDataset(data_folder, target_folder, val_index)
    # test_dataset = TangoDataset(test_data, num_initial_values=initial_values)
    
    # Create data loaders
    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_dict[model_type]()
    model_name = 'models/tango_' + model_type + '_model.pth'
    
    if resume and os.path.isfile(model_name):
        print('Loading from', model_name)
        model.load_state_dict(torch.load(model_name))
    criterion = nn.BCELoss()
    
    # Train model
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=epochs,
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/tango_' + model_type + '_training_history.png')
    # plt.show()
    
    # # Test model
    # cell_accuracy, board_accuracy = test_model(model, test_loader, device)
    
    # # Visualize predictions
    # visualize_predictions(model, test_loader, num_examples=5, device=device)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


def main(folder='solutions', model_type='cnn', random_cells=12, epochs=50, resume=False):
    sol_filenames = [os.path.join(folder, f) for f in os.listdir(folder)]
    
    print(f"Loaded {len(sol_filenames)} Tango board solutions")
    
    # Convert solutions format if needed (assuming 1 for Sun, 0 for Moon)
    # If your solutions use different values, adjust accordingly
    
    # Split data into train, validation, test sets
    train_data, test_data = train_test_split(sol_filenames, test_size=0.25, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
    
    print(f"Train set: {len(train_data)}, Validation set: {len(val_data)}, Test set: {len(test_data)}")
    
    # Create datasets
    initial_values=random_cells
    train_dataset = TangoDataset(train_data, num_initial_values=initial_values)
    val_dataset = TangoDataset(val_data, num_initial_values=initial_values)
    # test_dataset = TangoDataset(test_data, num_initial_values=initial_values)
    
    # Create data loaders
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_dict[model_type]()
    model_name = 'models/tango_' + model_type + '_model.pth'
    
    if resume and os.path.isfile(model_name):
        print('Loading from', model_name)
        model.load_state_dict(torch.load(model_name))
    criterion = nn.BCELoss()
    
    # Train model
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=epochs,
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/tango_' + model_type + '_training_history.png')
    # plt.show()
    
    # # Test model
    # cell_accuracy, board_accuracy = test_model(model, test_loader, device)
    
    # # Visualize predictions
    # visualize_predictions(model, test_loader, num_examples=5, device=device)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


def main_test(model_type='cnn', sol_folder='solutions', random_cells=None):
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_dict[model_type]()
    model.name = model_type
    state_dict_pth = 'models/tango_' + model_type + '_model.pth'
    model.load_state_dict(torch.load(state_dict_pth))
    model.to(device)
    model.eval()

    # Create data loaders
    sol_filenames = [os.path.join(sol_folder, f) for f in os.listdir(sol_folder)]
    batch_size = 100
    # Split data into train, validation, test sets
    _, test_data = train_test_split(sol_filenames, test_size=0.25, random_state=42)

    if not random_cells:
        random_cells = 1

    for initial_values in range(random_cells, 36):
        print(f"\nInitial values {initial_values}, removing {36 - initial_values}")
        test_dataset = TangoDataset(test_data, num_initial_values=initial_values)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Test model
        cell_accuracy, board_accuracy = test_model(model, test_loader, device)

        # print(f"Cell accuracy: {cell_accuracy}, board accuracy, {board_accuracy}\n")
        
        # # Visualize predictions
        if board_accuracy >= 0.5:
            visualize_predictions(model, test_loader, num_examples=5, device=device, initial_values=initial_values)
            break


if __name__ == "__main__":
    # models = ['cnn', 'ensemble', 'deepseek', 'claude', 'chatgpt', 'transformer']
    models = ['ensemble']
    for model_type in models:
        main_min(data_folder='data', target_folder='target', model_type=model_type, epochs=50, resume=False)
        # main_test(model_type=model_type, random_cells=10)
        # main(model_type=model_type, random_cells=18, epochs=50, resume=False)
        main_test(model_type=model_type, random_cells=18)

