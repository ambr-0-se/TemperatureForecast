# config.py
# parameters for the weather prediction dataset
import os
from models import my_GCN, my_GAT, my_GraphSAGE
from models import GCN, GAT, GraphSAGE
from models import LinearRegression, RandomForestRegression, MLP
from models import ytd_model

PROJECT_ROOT = os.path.dirname(__file__)

# data preprocessing config
neighbors_per_station = [1] # number of neighbors (k-nearest neighbors)
time_series = False
temporal_window = 1

# training config
train_times = 3
epochs = 100
batch_size = 256 # [1, 4, 32, 64, 256]

# models config


baseline_configs = [
    {'name': 'ytd_model', 'model': ytd_model}
]

gnn_model_configs = [
    {'name': 'my_GCN', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 1}},
    {'name': 'my_GCN_with_edge_attr', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 1}},
    {'name': 'my_GAT', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 1}},
    {'name': 'my_GAT_with_edge_attr', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 2}},
    {'name': 'GraphSAGE', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.2}, 'use_edge_attr': False}
]

ML_model_configs = [
    {'name': 'LinearRegression', 'model': LinearRegression, 'hyperparameters': {}},
    {'name': 'MLP', 'model': MLP, 'hyperparameters': {'hidden_size': 128, 'num_layers': 3}},
    {'name': 'RandomForest', 'model': RandomForestRegression, 'hyperparameters': {'n_estimators': 100, 'max_depth': None}}
]

# gnn_model_configs = [
#     {'name': 'my_GCN_1layer', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 1}},
#     {'name': 'my_GCN_with_edge_attr_1layer', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 1}},
#     {'name': 'my_GAT_1layer', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 1}},
#     {'name': 'my_GAT_with_edge_attr_1layer', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 1}},
#     {'name': 'GraphSAGE_1layer', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 1, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'my_GCN_2layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 2}},
#     {'name': 'my_GCN_with_edge_attr_2layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 2}},
#     {'name': 'my_GAT_2layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 2}},
#     {'name': 'my_GAT_with_edge_attr_2layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 2}},
#     {'name': 'GraphSAGE_2layers', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'my_GCN_3layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GCN_with_edge_attr_3layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GAT_3layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GAT_with_edge_attr_3layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'GraphSAGE_3layers', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'my_GCN_4layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 4}},
#     {'name': 'my_GCN_with_edge_attr_4layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 4}},
#     {'name': 'my_GAT_4layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 4}},
#     {'name': 'my_GAT_with_edge_attr_4layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 4}},
#     {'name': 'GraphSAGE_4layers', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 4, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'my_GCN_5layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 5}},
#     {'name': 'my_GCN_with_edge_attr_5layers', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 5}},
#     {'name': 'my_GAT_5layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 5}},
#     {'name': 'my_GAT_with_edge_attr_5layers', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 5}},
#     {'name': 'GraphSAGE_5layers', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 5, 'dropout': 0.2}, 'use_edge_attr': False},
# ]
# gnn_model_configs = [
#     {'name': 'my_GCN', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GCN_with_edge_attr', 'model': my_GCN, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GAT', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GAT_with_edge_attr', 'model': my_GAT, 'hyperparameters': {'use_edge_attr': True, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'my_GraphSAGE', 'model': my_GraphSAGE, 'hyperparameters': {'use_edge_attr': False, 'hidden_size': 128, 'num_layers': 3}},
#     {'name': 'GCN', 'model': GCN, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'GCN_with_edge_attr', 'model': GCN, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': True},
#     {'name': 'GAT', 'model': GAT, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': False},
#     {'name': 'GAT_with_edge_attr', 'model': GAT, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': True},
#     {'name': 'GraphSAGE', 'model': GraphSAGE, 'hyperparameters': {'hidden_channels': 128, 'num_layers': 3, 'dropout': 0.2}, 'use_edge_attr': False}
# ]