from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import GCN, GAT, GraphSAGE
from torch import nn, optim
import torch.nn.functional as F


import torch
from torch_geometric.data import Batch

# for each model, 
# define the model class, 
# and the training objective (regression and mean squared error), 
# and save the training loss and validation loss over epochs, and final test loss
# and use the validation set to tune the hyperparameters

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# base class
class Model():
    def __init__(self):
        pass
    def train_model(self):
        pass
    def hyperparameters_tuning(self):
        pass
    def get_loss(self, prediction, target):
        self.eval()
        with torch.no_grad():
            out = self(prediction)
            loss = self.criterion(out, target)
        self.train()
        return loss.item()

class ytd_model(Model):
    '''Predict the temperature of the next day using the temperature of the current day'''
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def train_model(self, train_loader, validation_loader, test_loader, epochs=200):
        train_loss = 0
        val_loss = 0
        test_loss = 0

        for train_batch in train_loader:
            for index in range(-8, 0):
                if 'temp_mean' in train_batch.feature_names[0][index]:
                    break
            # print(f'train_batch: {train_batch}')
            # print(f'train_batch.x: {train_batch.x.shape}')
            # print(f'train_batch.y: {train_batch.y.shape}')
            # print(f'train_batch.feature_names: {len(train_batch.feature_names)};\n   {train_batch.feature_names[0]}')
            # # for i in range(len(train_batch.feature_names)):
            # #     print(f'train_batch.feature_names[{i}]: {train_batch.feature_names[i]}; type: {type(train_batch.feature_names[i])}')
            # print(f'train_batch.feature_names[-8]: {train_batch.feature_names[0][-8]}; type: {type(train_batch.feature_names[0][-8])}')
            # print(f'train_batch.feature_names[-7]: {train_batch.feature_names[0][-7]}; type: {type(train_batch.feature_names[0][-7])}')
            # print(f'train_batch.feature_names[-6]: {train_batch.feature_names[0][-6]}; type: {type(train_batch.feature_names[0][-6])}')
            # print(f'train_batch.feature_names[-5]: {train_batch.feature_names[0][-5]}; type: {type(train_batch.feature_names[0][-5])}')
            # print(f'train_batch.feature_names[-4]: {train_batch.feature_names[0][-4]}; type: {type(train_batch.feature_names[0][-4])}')
            # print(f'train_batch.feature_names[-3]: {train_batch.feature_names[0][-3]}; type: {type(train_batch.feature_names[0][-3])}')
            # print(f'train_batch.feature_names[-2]: {train_batch.feature_names[0][-2]}; type: {type(train_batch.feature_names[0][-2])}')
            # print(f'train_batch.feature_names[-1]: {train_batch.feature_names[0][-1]}; type: {type(train_batch.feature_names[0][-1])}')
            # break

        for train_batch in train_loader:
            # print(f'train_batch: {train_batch}')
            # print(f'train_batch.x: {train_batch.x.shape}')
            # print(f'train_batch.y: {train_batch.y.shape}')
            # print(f'train_batch.feature_names: {len(train_batch.feature_names)}')
            out = train_batch.x[:,index] # -6 is the index of the temperature of the current day
            # print(f'out: {out.shape}')
            # [num_nodes * batch_size, 1]
        
            # Use PyG's built-in batch handling
            batch_size = train_batch.num_graphs
            out = out.view(batch_size, -1)  # [batch_size, num_nodes]
            y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
        
            loss = self.criterion(out, y.to(device))
            train_loss += loss.item()
        training_loss = train_loss / len(train_loader)

        for validation_batch in validation_loader:
            out = validation_batch.x[:, index] # -6 is the index of the temperature of the current day
            # [num_nodes * batch_size, 1]
        
            # Use PyG's built-in batch handling
            batch_size = validation_batch.num_graphs
            out = out.view(batch_size, -1)  # [batch_size, num_nodes]
            y = validation_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
        
            loss = self.criterion(out, y.to(device))
            val_loss += loss.item()
        validation_loss = val_loss / len(validation_loader)

        for test_batch in test_loader:
            out = test_batch.x[:, index] # -6 is the index of the temperature of the current day
            # [num_nodes * batch_size, 1]
        
            # Use PyG's built-in batch handling
            batch_size = test_batch.num_graphs
            out = out.view(batch_size, -1)  # [batch_size, num_nodes]
            y = test_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
        
            loss = self.criterion(out, y.to(device))
            test_loss += loss.item()
        test_loss = test_loss / len(test_loader)

        print(f'training_loss: {training_loss}')
        print(f'validation_loss: {validation_loss}')
        print(f'test_loss: {test_loss}')
        return {
            'training_loss': [training_loss] * epochs,
            'validation_loss': [validation_loss] * epochs,
            'test_loss': test_loss
        }

# ML Model
# Linear Regression
# Implement using scikit-learn
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 18)  # Changed output dim to 18
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        
    def forward(self, x):
        # Ensure input is a tensor and has correct shape
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.fc(x)  # Will output shape [batch_size, 18]

    def train_model(self, train_data, validation_data, test_data, epochs=200):
        training_loss = []
        validation_loss = []
        
        # Get features and targets from the data dictionary
        train_x = train_data['x'].float()
        train_y = train_data['y'].float()
        val_x = validation_data['x'].float()
        val_y = validation_data['y'].float()
        test_x = test_data['x'].float()
        test_y = test_data['y'].float()
        
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            out = self(train_x.to(device))
            loss = self.criterion(out, train_y.to(device))
            loss.backward()
            self.optimizer.step()

            training_loss.append(loss.item())
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_out = self(val_x.to(device))
                val_loss = self.criterion(val_out, val_y.to(device)).item()
            validation_loss.append(val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
                # print(f'Sample predictions: {out[:5].detach().numpy()}')
                # print(f'Sample targets: {train_y[:5].numpy()}')

        # Test loss calculation
        self.eval()
        with torch.no_grad():
            test_out = self(test_x.to(device))
            test_loss = self.criterion(test_out, test_y.to(device)).item()
            print('\nFinal Test Results:')
            print(f'Test Loss (MSE): {test_loss:.4f}')
            print(f'Test RMSE: {torch.sqrt(torch.tensor(test_loss)):.4f}°C')    
        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    def hyperparameters_tuning(self):
        pass    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

    def train_model(self, train_data, validation_data, test_data, epochs=200):
        training_loss = []
        validation_loss = []
        
        # Get features and targets from the data dictionary
        train_x = train_data['x'].float()
        train_y = train_data['y'].float()
        val_x = validation_data['x'].float()
        val_y = validation_data['y'].float()
        test_x = test_data['x'].float()
        test_y = test_data['y'].float()
        
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            out = self(train_x.to(device))
            loss = self.criterion(out, train_y.to(device))
            loss.backward()
            self.optimizer.step()

            training_loss.append(loss.item())
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_out = self(val_x.to(device))
                val_loss = self.criterion(val_out, val_y.to(device)).item()
            validation_loss.append(val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
                # print(f'Sample predictions: {out[:5].detach().numpy()}')
                # print(f'Sample targets: {train_y[:5].numpy()}')

        # Test loss calculation
        self.eval()
        with torch.no_grad():
            test_out = self(test_x.to(device))
            test_loss = self.criterion(test_out, test_y.to(device)).item()
            print('\nFinal Test Results:')
            print(f'Test Loss (MSE): {test_loss:.4f}')
            print(f'Test RMSE: {torch.sqrt(torch.tensor(test_loss)):.4f}°C')    
        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    
    def hyperparameters_tuning(self):
        pass

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForestRegression(Model):
    def __init__(self, input_dim, output_dim, n_estimators=100, max_depth=None):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.criterion = nn.MSELoss()
        
    def train_model(self, train_data, validation_data, test_data, epochs=200):
        # Get features and targets
        train_x = train_data['x'].numpy()
        train_y = train_data['y'].numpy()
        val_x = validation_data['x'].numpy()
        val_y = validation_data['y'].numpy()
        test_x = test_data['x'].numpy()
        test_y = test_data['y'].numpy()
        
        # Train the model
        self.model.fit(train_x, train_y)
        
        # Calculate predictions
        train_pred = self.model.predict(train_x)
        val_pred = self.model.predict(val_x)
        test_pred = self.model.predict(test_x)
        
        # Calculate final losses
        training_loss = np.mean((train_pred - train_y) ** 2)
        validation_loss = np.mean((val_pred - val_y) ** 2)
        test_loss = np.mean((test_pred - test_y) ** 2)
        
        print('\nRandom Forest Results:')
        print(f'Train Loss: {training_loss:.4f}, Val Loss: {validation_loss:.4f}')
        print(f'\nFinal Test Results:')
        print(f'Test Loss (MSE): {test_loss:.4f}')
        print(f'Test RMSE: {np.sqrt(test_loss):.4f}°C')
        
        # For visualization compatibility, return constant losses
        num_epochs = epochs  # Match other models
        return {
            'training_loss': [training_loss] * num_epochs,  # Constant loss over "epochs"
            'validation_loss': [validation_loss] * num_epochs,
            'test_loss': test_loss
        }
    
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return self.model.predict(X)


# NN Model
# MLP


# GNN Model
class GNN(torch.nn.Module):
    def __init__(self, use_edge_attr=False, num_attn_heads=0):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.num_attn_heads = num_attn_heads

    def train_model(self, train_loader, validation_loader, test_loader, epochs=200):
        training_loss = []
        validation_loss = []
        
        # Convert targets to float
        for data in train_loader:
            data.y = data.y.float()
        for data in validation_loader:
            data.y = data.y.float()
        for data in test_loader:
            data.y = data.y.float()
        
        # Print shapes before training
        # print("\nGNN Model Shapes:")
        # print(f"Input features shape: {train_batch.x.shape}")  # Should be [num_nodes * batch_size, num_features]
        # print(f"Edge index shape: {train_batch.edge_index.shape}")  # Should be [2, num_edges * batch_size]
        # print(f"Edge attr shape: {train_batch.edge_attr.shape}")  # Should be [num_edges * batch_size, edge_features]
        # print(f"Target shape: {train_batch.y.shape}")  # Should be [num_nodes * batch_size]
        
        for epoch in range(epochs):
            train_loss = 0
            for train_batch in train_loader:
                self.train()
                self.optimizer.zero_grad()
                out = self(train_batch.to(device))  # [num_nodes * batch_size, 1]
            
                # Use PyG's built-in batch handling
                batch_size = train_batch.num_graphs
                out = out.view(batch_size, -1)  # [batch_size, num_nodes]
                y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
            
                loss = self.criterion(out, y.to(device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            training_loss.append(train_loss / len(train_loader))
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for val_batch in validation_loader:
                    val_out = self(val_batch.to(device))
                    # Reshape validation outputs and targets
                    val_out = val_out.view(val_batch.num_graphs, -1)
                    val_y = val_batch.y.view(val_batch.num_graphs, -1)
                    # print(f'val_out: {val_out}')
                    # print(f'val_y: {val_y}')
                    # print(f'loss: {self.criterion(val_out, val_y).item()}')
                    val_loss += self.criterion(val_out, val_y).item()
                    # print(f'val_loss: {val_loss}')
                # print(f'Final avg val_loss: {val_loss/ len(validation_loader)}')
                validation_loss.append(val_loss / len(validation_loader))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}')
                # print(f'Sample predictions shape: {out.shape}')
                # print(f'Sample targets shape: {y.shape}')

        # Test evaluation
        with torch.no_grad():
            test_loss = 0
            for test_batch in test_loader:
                test_out = self(test_batch.to(device))
                test_out = test_out.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_y = test_batch.y.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_loss += self.criterion(test_out, test_y.to(device)).item()
            test_loss = test_loss / len(test_loader)

        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    # def train_model(self, train_data, validation_data, test_data):
    #     training_loss = []
    #     validation_loss = []
        
    #     # Convert targets to float
    #     for data in train_data:
    #         data.y = data.y.float()
    #     for data in validation_data:
    #         data.y = data.y.float()
    #     for data in test_data:
    #         data.y = data.y.float()
        
    #     # Create batches
    #     train_batch = Batch.from_data_list(train_data)
    #     val_batch = Batch.from_data_list(validation_data)
    #     test_batch = Batch.from_data_list(test_data)
        
    #     # Print shapes before training
    #     print("\nGNN Model Shapes:")
    #     print(f"Input features shape: {train_batch.x.shape}")  # Should be [num_nodes * batch_size, num_features]
    #     print(f"Edge index shape: {train_batch.edge_index.shape}")  # Should be [2, num_edges * batch_size]
    #     print(f"Edge attr shape: {train_batch.edge_attr.shape}")  # Should be [num_edges * batch_size, edge_features]
    #     print(f"Target shape: {train_batch.y.shape}")  # Should be [num_nodes * batch_size]
        
    #     for epoch in range(100):
    #         self.train()
    #         self.optimizer.zero_grad()
    #         out = self(train_batch)  # [num_nodes * batch_size, 1]
            
    #         # Use PyG's built-in batch handling
    #         batch_size = train_batch.num_graphs
    #         out = out.view(batch_size, -1)  # [batch_size, num_nodes]
    #         y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
            
    #         loss = self.criterion(out, y)
    #         loss.backward()
    #         self.optimizer.step()
    #         training_loss.append(loss.item())
            
    #         # Validation
    #         self.eval()
    #         with torch.no_grad():
    #             val_out = self(val_batch)
    #             # Reshape validation outputs and targets
    #             val_out = val_out.view(len(validation_data), -1)
    #             val_y = val_batch.y.view(len(validation_data), -1)
    #             val_loss = self.criterion(val_out, val_y).item()
    #         validation_loss.append(val_loss)
            
    #         # Print progress
    #         if (epoch + 1) % 10 == 0:
    #             print(f'Epoch {epoch+1:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    #             print(f'Sample predictions shape: {out.shape}')
    #             print(f'Sample targets shape: {y.shape}')

    #     # Test evaluation
    #     with torch.no_grad():
    #         test_out = self(test_batch)
    #         test_out = test_out.view(len(test_data), -1)  # [batch_size, num_nodes]
    #         test_y = test_batch.y.view(len(test_data), -1)  # [batch_size, num_nodes]
    #         test_loss = self.criterion(test_out, test_y)

    #     return {
    #         'training_loss': training_loss,
    #         'validation_loss': validation_loss,
    #         'test_loss': test_loss
    #     }
    def hyperparameters_tuning(self):
        pass
    def get_loss(self, prediction, target):
        self.eval()
        # print(f'prediction.shape: {prediction.shape}')
        # print(f'target.shape: {target.shape}')
        with torch.no_grad():
            loss = self.criterion(prediction.to(device), target.to(device))
        self.train()
        return loss.item()

# Graph Convolutional Network
# Graph Attention Network
# GraphSAGE

# GCN Model
class my_GCN(GNN):
    def __init__(self, num_features, num_outputs, hidden_size=64, num_layers=3, use_edge_attr=False, num_attn_heads=0):
        super().__init__(use_edge_attr, num_attn_heads)
        if num_attn_heads > 0:
            self.time_and_location_dim = 11
            self.weather_dim = 8
            self.window_size = (num_features - self.time_and_location_dim) // self.weather_dim
            self.feature_dim = self.time_and_location_dim + self.weather_dim
            self.num_features = self.feature_dim * self.window_size

            self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_attn_heads, batch_first=True)
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size, self.feature_dim)) # Learnable Positional Embeddings: [T, F]
            self.norm1 = nn.LayerNorm(self.num_features)
        
            self.ff = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.num_features, self.num_features)
            )
            self.norm2 = nn.LayerNorm(self.num_features)

        else:
            self.num_features = num_features
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, hidden_size))
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, 1)  # One output per node
        
        # Add back optimizer and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def forward(self, data):
        if self.num_attn_heads > 0:
            x, edge_index = data.x, data.edge_index
            # print(f'x.shape: {x.shape}')
            time_and_location = x[:, :self.time_and_location_dim]
            x = torch.cat(
                [
                    time_and_location.unsqueeze(1).expand(-1, self.window_size, -1),
                    x[:, self.time_and_location_dim:].reshape(x.shape[0], self.window_size, self.weather_dim)
                ],
                dim=-1
            )
            # print(f'x.shape after concatenate weather with time and location: {x.shape}')
            x = x + self.pos_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            # print(f'x.shape after adding positional embeddings: {x.shape}')
            attn_out, _ = self.attn(x, x, x) # x + self.attn(x, x, x)
            # print(f'attn_out.shape after attention: {attn_out.shape}')
            attn_out = attn_out.reshape(attn_out.shape[0], -1)
            # print(f'attn_out.shape after reshape: {attn_out.shape}')
            x = x.reshape(x.shape[0], -1)
            # print(f'x.shape after reshape: {x.shape}')

            x = self.norm1(x+attn_out)
            x = self.ff(x)
            x = self.norm2(x)
            # print(f'x.shape after norm and ff: {x.shape}')
        else:
            x, edge_index = data.x, data.edge_index
        if self.use_edge_attr and hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
            # Convert edge attributes to weights (inverse distance)
            edge_weights = 1 / (edge_attr + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Apply edge weights in convolutions
            for conv in self.convs:
                x = conv(x, edge_index, edge_weight=edge_weights)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        else:
            # Use uniform weights
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc(x)
        return x  # Don't squeeze - let train_model handle reshaping
    
    def hyperparameters_tuning(self):
        pass

class my_GAT(GNN):
    def __init__(self, num_features, num_outputs, hidden_size=64, num_layers=3, use_edge_attr=False, num_attn_heads=0):
        super().__init__(use_edge_attr, num_attn_heads)

        if num_attn_heads > 0:
            self.time_and_location_dim = 11
            self.weather_dim = 8
            self.window_size = (num_features - self.time_and_location_dim) // self.weather_dim
            self.feature_dim = self.time_and_location_dim + self.weather_dim
            self.num_features = self.feature_dim * self.window_size

            self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_attn_heads, batch_first=True)
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size, self.feature_dim)) # Learnable Positional Embeddings: [T, F]
            self.norm1 = nn.LayerNorm(self.num_features)
        
            self.ff = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.num_features, self.num_features)
            )
            self.norm2 = nn.LayerNorm(self.num_features)

        else:
            self.num_features = num_features
        edge_dim = 1 if use_edge_attr else None
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.num_features if i == 0 else hidden_size
            self.convs.append(GATConv(in_channels, hidden_size, heads=4, concat=False, edge_dim=edge_dim))
        
        self.fc = nn.Linear(hidden_size, 1)  # One output per node
        
        # Add back optimizer and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def forward(self, data):
        if self.num_attn_heads > 0:
            x, edge_index = data.x, data.edge_index
            # print(f'x.shape: {x.shape}')
            time_and_location = x[:, :self.time_and_location_dim]
            x = torch.cat(
                [
                    time_and_location.unsqueeze(1).expand(-1, self.window_size, -1),
                    x[:, self.time_and_location_dim:].reshape(x.shape[0], self.window_size, self.weather_dim)
                ],
                dim=-1
            )
            # print(f'x.shape after concatenate weather with time and location: {x.shape}')
            x = x + self.pos_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            # print(f'x.shape after adding positional embeddings: {x.shape}')
            attn_out, _ = self.attn(x, x, x) # x + self.attn(x, x, x)
            # print(f'attn_out.shape after attention: {attn_out.shape}')
            attn_out = attn_out.reshape(attn_out.shape[0], -1)
            # print(f'attn_out.shape after reshape: {attn_out.shape}')
            x = x.reshape(x.shape[0], -1)
            # print(f'x.shape after reshape: {x.shape}')

            x = self.norm1(x+attn_out)
            x = self.ff(x)
            x = self.norm2(x)
            # print(f'x.shape after norm and ff: {x.shape}')
        else:
            x, edge_index = data.x, data.edge_index
        
        if self.use_edge_attr and hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        else:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc(x)
        return x  # Don't squeeze - let train_model handle reshaping
    def hyperparameters_tuning(self):
        pass

class my_GraphSAGE(GNN):
    def __init__(self, num_features, num_outputs, hidden_size=64, num_layers=3, use_edge_attr=False, num_attn_heads=0):
        super().__init__(use_edge_attr, num_attn_heads)
        if num_attn_heads > 0:
            self.time_and_location_dim = 11
            self.weather_dim = 8
            self.window_size = (num_features - self.time_and_location_dim) // self.weather_dim
            self.feature_dim = self.time_and_location_dim + self.weather_dim
            self.num_features = self.feature_dim * self.window_size

            self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_attn_heads, batch_first=True)
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size, self.feature_dim)) # Learnable Positional Embeddings: [T, F]
            self.norm1 = nn.LayerNorm(self.num_features)
        
            self.ff = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.num_features, self.num_features)
            )
            self.norm2 = nn.LayerNorm(self.num_features)

        else:
            self.num_features = num_features
        self.convs = nn.ModuleList()
        # self.convs.append(SAGEConv(self.num_features, hidden_size))
        self.convs.append(SAGEConv(self.num_features, hidden_size))
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, 1)  # One output per node
        
        # Add back optimizer and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def forward(self, data):
        if self.num_attn_heads > 0:
            x, edge_index = data.x, data.edge_index
            # print(f'x.shape: {x.shape}')
            time_and_location = x[:, :self.time_and_location_dim]
            x = torch.cat(
                [
                    time_and_location.unsqueeze(1).expand(-1, self.window_size, -1),
                    x[:, self.time_and_location_dim:].reshape(x.shape[0], self.window_size, self.weather_dim)
                ],
                dim=-1
            )
            # print(f'x.shape after concatenate weather with time and location: {x.shape}')
            x = x + self.pos_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            # print(f'x.shape after adding positional embeddings: {x.shape}')
            attn_out, _ = self.attn(x, x, x) # x + self.attn(x, x, x)
            # print(f'attn_out.shape after attention: {attn_out.shape}')
            # print(f'x.shape after reshape: {x.shape}')

            attn_out = attn_out.reshape(attn_out.shape[0], -1)
            # # print(f'attn_out.shape after reshape: {attn_out.shape}')
            x = x.reshape(x.shape[0], -1)
            # # print(f'x.shape after reshape: {x.shape}')

            x = self.norm1(x+attn_out)
            x = self.ff(x)
            x = self.norm2(x)
            # print(f'x.shape before mean: {x.shape}')
            # x = x.mean(dim=1)
            # print(f'x.shape after mean: {x.shape}')
            # print(f'x.shape after norm and ff: {x.shape}')
        else:
            x, edge_index = data.x, data.edge_index

        # print(f'attn_heads: {self.num_attn_heads}')
        
        # GraphSAGE aggregates neighbor information without edge weights
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc(x)
        return x  # Don't squeeze - let train_model handle reshaping
    def hyperparameters_tuning(self):
        pass

class GNNmixin:
    def train_model(self, train_loader, validation_loader, test_loader, epochs=200):
        pass
    
    def hyperparameters_tuning(self):
        pass

    def get_loss(self, prediction, target):
        self.eval()
        with torch.no_grad():
            loss = self.criterion(prediction.to(device), target.to(device))
        self.train()
        return loss.item()

class GCN(GCN, GNNmixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train_model(self, train_loader, validation_loader, test_loader, epochs=200, use_edge_attr=False):
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        training_loss = []
        validation_loss = []
        
        # Convert targets to float
        for data in train_loader:
            data.y = data.y.float()
        for data in validation_loader:
            data.y = data.y.float()
        for data in test_loader:
            data.y = data.y.float()
        
        for epoch in range(epochs):
            train_loss = 0
            for train_batch in train_loader:
                self.train()
                self.optimizer.zero_grad()
                # out = self(train_batch.to(device))  # [num_nodes * batch_size, 1]
                if use_edge_attr:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index, edge_weight=1/(train_batch.edge_attr + 1e-10))
                else:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index)
            
                # Use PyG's built-in batch handling
                batch_size = train_batch.num_graphs
                # print(f'train_batch.x.shape: {train_batch.x.shape}')
                # print(f'original out.shape: {out.shape}')
                out = out.view(batch_size, -1)  # [batch_size, num_nodes]
                # print(f'reshaped out.shape: {out.shape}')
                # print(f'train_batch.y.shape: {train_batch.y.shape}')
                y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
                # print(f'reshaped y.shape: {y.shape}')
                loss = self.criterion(out, y.to(device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            training_loss.append(train_loss / len(train_loader))
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for val_batch in validation_loader:
                    # val_out = self(val_batch.to(device))
                    if use_edge_attr:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index, edge_weight=1/(val_batch.edge_attr + 1e-10))
                    else:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index)
                    # Reshape validation outputs and targets
                    val_out = val_out.view(val_batch.num_graphs, -1)
                    val_y = val_batch.y.view(val_batch.num_graphs, -1)
                    # print(f'val_out: {val_out}')
                    # print(f'val_y: {val_y}')
                    # print(f'loss: {self.criterion(val_out, val_y).item()}')
                    val_loss += self.criterion(val_out, val_y).item()
                    # print(f'val_loss: {val_loss}')
                # print(f'Final avg val_loss: {val_loss/ len(validation_loader)}')
                validation_loss.append(val_loss / len(validation_loader))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}')
                # print(f'Sample predictions shape: {out.shape}')
                # print(f'Sample targets shape: {y.shape}')

        # Test evaluation
        with torch.no_grad():
            test_loss = 0
            for test_batch in test_loader:
                # test_out = self(test_batch.to(device))
                if use_edge_attr:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index, edge_weight=1/(test_batch.edge_attr + 1e-10))
                else:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index)
                test_out = test_out.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_y = test_batch.y.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_loss += self.criterion(test_out, test_y.to(device)).item()
            test_loss = test_loss / len(test_loader)

        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    
class GAT(GAT, GNNmixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(self, train_loader, validation_loader, test_loader, epochs=200, use_edge_attr=False):
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        training_loss = []
        validation_loss = []
        
        # Convert targets to float
        for data in train_loader:
            data.y = data.y.float()
        for data in validation_loader:
            data.y = data.y.float()
        for data in test_loader:
            data.y = data.y.float()
        
        for epoch in range(epochs):
            train_loss = 0
            for train_batch in train_loader:
                self.train()
                self.optimizer.zero_grad()
                # out = self(train_batch.to(device))  # [num_nodes * batch_size, 1]
                if use_edge_attr:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index, edge_attr=train_batch.edge_attr)
                else:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index)
            
                # Use PyG's built-in batch handling
                batch_size = train_batch.num_graphs
                out = out.view(batch_size, -1)  # [batch_size, num_nodes]
                y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
            
                loss = self.criterion(out, y.to(device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            training_loss.append(train_loss / len(train_loader))
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for val_batch in validation_loader:
                    # val_out = self(val_batch.to(device))
                    if use_edge_attr:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index, edge_attr=val_batch.edge_attr)
                    else:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index)
                    # Reshape validation outputs and targets
                    val_out = val_out.view(val_batch.num_graphs, -1)
                    val_y = val_batch.y.view(val_batch.num_graphs, -1)
                    # print(f'val_out: {val_out}')
                    # print(f'val_y: {val_y}')
                    # print(f'loss: {self.criterion(val_out, val_y).item()}')
                    val_loss += self.criterion(val_out, val_y).item()
                    # print(f'val_loss: {val_loss}')
                # print(f'Final avg val_loss: {val_loss/ len(validation_loader)}')
                validation_loss.append(val_loss / len(validation_loader))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}')
                # print(f'Sample predictions shape: {out.shape}')
                # print(f'Sample targets shape: {y.shape}')

        # Test evaluation
        with torch.no_grad():
            test_loss = 0
            for test_batch in test_loader:
                # test_out = self(test_batch.to(device))
                if use_edge_attr:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index, edge_attr=test_batch.edge_attr)
                else:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index)
                test_out = test_out.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_y = test_batch.y.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_loss += self.criterion(test_out, test_y.to(device)).item()
            test_loss = test_loss / len(test_loader)

        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    
class GraphSAGE(GraphSAGE, GNNmixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(self, train_loader, validation_loader, test_loader, epochs=200, use_edge_attr=False):
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        training_loss = []
        validation_loss = []
        
        # Convert targets to float
        for data in train_loader:
            data.y = data.y.float()
        for data in validation_loader:
            data.y = data.y.float()
        for data in test_loader:
            data.y = data.y.float()
        
        for epoch in range(epochs):
            train_loss = 0
            for train_batch in train_loader:
                self.train()
                self.optimizer.zero_grad()
                # out = self(train_batch.to(device))  # [num_nodes * batch_size, 1]
                if use_edge_attr:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index, edge_attr=train_batch.edge_attr)
                else:
                    out = self(x=train_batch.x, edge_index=train_batch.edge_index)
            
                # Use PyG's built-in batch handling
                batch_size = train_batch.num_graphs
                out = out.view(batch_size, -1)  # [batch_size, num_nodes]
                y = train_batch.y.view(batch_size, -1)  # [batch_size, num_nodes]
            
                loss = self.criterion(out, y.to(device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            training_loss.append(train_loss / len(train_loader))
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for val_batch in validation_loader:
                    # val_out = self(val_batch.to(device))
                    if use_edge_attr:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index, edge_attr=val_batch.edge_attr)
                    else:
                        val_out = self(x=val_batch.x, edge_index=val_batch.edge_index)
                    # Reshape validation outputs and targets
                    val_out = val_out.view(val_batch.num_graphs, -1)
                    val_y = val_batch.y.view(val_batch.num_graphs, -1)
                    # print(f'val_out: {val_out}')
                    # print(f'val_y: {val_y}')
                    # print(f'loss: {self.criterion(val_out, val_y).item()}')
                    val_loss += self.criterion(val_out, val_y).item()
                    # print(f'val_loss: {val_loss}')
                # print(f'Final avg val_loss: {val_loss/ len(validation_loader)}')
                validation_loss.append(val_loss / len(validation_loader))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {training_loss[-1]:.4f}, Val Loss: {validation_loss[-1]:.4f}')
                # print(f'Sample predictions shape: {out.shape}')
                # print(f'Sample targets shape: {y.shape}')

        # Test evaluation
        with torch.no_grad():
            test_loss = 0
            for test_batch in test_loader:
                # test_out = self(test_batch.to(device))
                if use_edge_attr:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index, edge_attr=test_batch.edge_attr)
                else:
                    test_out = self(x=test_batch.x, edge_index=test_batch.edge_index)
                test_out = test_out.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_y = test_batch.y.view(test_batch.num_graphs, -1)  # [batch_size, num_nodes]
                test_loss += self.criterion(test_out, test_y.to(device)).item()
            test_loss = test_loss / len(test_loader)

        return {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'test_loss': test_loss
        }
    