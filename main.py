"""
Weather Station Temperature Prediction

This module implements a comparative study of different machine learning models
for predicting temperatures across a network of weather stations.

Models compared:
- Graph Neural Networks (GCN, GAT, GraphSAGE)
- Traditional ML (Linear Regression, Random Forest)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time

from datetime import datetime

from create_region_graph import load_and_process_data, create_graph_data, plot_network_graph
from create_graph_features import get_graph_features, create_graph_dataset, load_weather_data, get_station_names
# from models import GCN, GAT, GraphSAGE, LinearRegression, MLP,RandomForestRegression
from visualization import visualize_results
import config


# Constants
# PROJECT_ROOT = 'cs886_project'
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.dirname(__file__)
# load current time and date
current_time = datetime.now().strftime("%d%B_%H%M")

PATHS = {
    'station': f'{PROJECT_ROOT}/data/all_stations.csv',
    'features': f'{PROJECT_ROOT}/features/',
    'results': f'{PROJECT_ROOT}/results/{current_time}/',
    'weather_data': f'{PROJECT_ROOT}/data/weather_prediction_dataset.csv',
    'graph_features': f'{PROJECT_ROOT}/features/graph_features.pkl'
}
# Create results directory if it doesn't exist
os.makedirs(PATHS['results'], exist_ok=True)

def save_config():
    """Save the configuration to a txt file."""
    config_path = os.path.join(PATHS['results'], 'config.txt')
    with open(config_path, 'w') as f:
        f.write('Data preprocessing config:\n')
        f.write(f"  Neighbors per station: {config.neighbors_per_station}\n")
        f.write(f"  Time series: {config.time_series}\n")
        f.write(f"  Temporal window: {config.temporal_window}\n")
        f.write('\nTraining config:\n')
        f.write(f"  Train times: {config.train_times}\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f'  Batch size: {config.batch_size}\n')
        f.write('\nModels config:\n')
        for model_config in config.gnn_model_configs:
            f.write(f"  _{model_config['name']}:\n")
            f.write(f"      Model: {model_config['model']}\n")
            for key, value in model_config['hyperparameters'].items():
                f.write(f"      {key}: {value}\n")
        f.write('\n')
        for model_config in config.ML_model_configs:
            f.write(f"  _{model_config['name']}:\n")
            f.write(f"      Model: {model_config['model']}\n")
            for key, value in model_config['hyperparameters'].items():
                f.write(f"      {key}: {value}\n")
        f.write('\n')

def split_data(indices, time_series=False, temporal_window=1):
    """Split data indices into train, validation, and test sets.
    
    Args:
        indices: List of data indices
        time_series: Whether to use time series prediction
        temporal_window: Number of previous timesteps to consider
    
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    np.random.seed(42)
    if not time_series:
        # Remove initial temporal_window indices needed for prediction
        valid_indices = indices[temporal_window:]
        # np.random.shuffle(valid_indices)
        
        # Split into 80% train, 10% validation, 10% test
        train_size = int(0.8 * len(valid_indices))
        val_size = int(0.9 * len(valid_indices))

        print(f'train indices: {valid_indices[0]}, {valid_indices[1]}, ..., {valid_indices[train_size-2]}, {valid_indices[train_size-1]}')
        print(f'val indices: {valid_indices[train_size]}, {valid_indices[train_size+1]}, ..., {valid_indices[val_size-2]}, {valid_indices[val_size-1]}')
        print(f'test indices: {valid_indices[val_size]}, {valid_indices[val_size+1]}, ..., {valid_indices[len(valid_indices)-2]}, {valid_indices[len(valid_indices)-1]}')
        
        return (
            valid_indices[:train_size],
            valid_indices[train_size:val_size],
            valid_indices[val_size:]
        )

def prepare_tabular_features(graph_features):
    """Convert graph features to tabular format for traditional ML models.
    
    Args:
        graph_features: Dictionary containing graph data for each set
    
    Returns:
        dict: Tabular features and targets for each set
    """
    feature_sets = {}
    
    for set_name in ['train', 'val', 'test']:
        features, targets = [], []
        
        for data in graph_features[set_name].dataset:
            # Flatten and concatenate node and edge features
            node_features = data.x.flatten()
            edge_features = data.edge_attr.flatten()
            combined_features = torch.cat([node_features, edge_features])
            
            features.append(combined_features)
            targets.append(data.y)
        
        # Convert lists to tensors
        feature_sets[set_name] = {
            'x': torch.stack(features) if features else torch.tensor([]),
            'y': torch.stack(targets) if targets else torch.tensor([])
        }
    
    _print_feature_shapes(feature_sets)
    return feature_sets

def _print_feature_shapes(feature_sets):
    """Helper function to print feature dimensions."""
    print("\nTabular Data Shapes:")
    for set_name, data in feature_sets.items():
        print(f"{set_name.capitalize()} features: {data['x'].shape}, "
              f"targets: {data['y'].shape}")

def train_and_evaluate_models(graph_features, gnn_dims, tabular_features, tabular_dims, train_times=3):
    """Train and evaluate all models.
    
    Args:
        graph_features: Graph-structured data for GNN models
        gnn_dims: Input/output dimensions for GNN models
        tabular_features: Tabular data for traditional ML models
        tabular_dims: Input/output dimensions for ML models
    
    Returns:
        dict: Training and evaluation results for all models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {'GNN': {}, 'ML': {}}
    # Check if edge_attr is available
    # Check number of edges
    # for batch in graph_features['train']:
    #     print(f'Batch edge_attr: {batch.edge_attr}')
    #     print(f'Batch edge_index: {batch.edge_index}')
    #     print(f'Number of edges: {batch.edge_index.shape[1]}')
    #     break

    # Train GNN models
    # a list of dictionaries with the model name and the model class (and hyperparameters)
    # gnn_models = [
    #     {'name': 'GCN', 'model': GCN, 'hyperparameters': {'use_edge_attr': False}},
    #     {'name': 'GCN_with_edge_attr', 'model': GCN, 'hyperparameters': {'use_edge_attr': True}},
    #     {'name': 'GAT', 'model': GAT, 'hyperparameters': {'use_edge_attr': False}},
    #     {'name': 'GAT_with_edge_attr', 'model': GAT, 'hyperparameters': {'use_edge_attr': True}},
    #     {'name': 'GraphSAGE', 'model': GraphSAGE, 'hyperparameters': {'use_edge_attr': False}}
    # ]

    baseline_models = config.baseline_configs
    print(f'baseline_models (using ytd\'s mean temperature): {baseline_models}')
    for model_config in baseline_models:
        model = model_config['model']()
        temp_results = model.train_model(
            graph_features['train'],
            graph_features['val'],
            graph_features['test'],
            epochs=config.epochs
        )
        results['GNN'][model_config['name']] = {
            'training_loss': {
                'mean': np.array(temp_results['training_loss']),
                'std': np.zeros(config.epochs)
            },
            'validation_loss': {
                'mean': np.array(temp_results['validation_loss']),
                'std': np.zeros(config.epochs)
            },
            'test_loss': {
                'mean': temp_results['test_loss'],
                'std': 0
            },
            'time (in sec)': 0
        }

    gnn_models = config.gnn_model_configs
    for model_config in gnn_models:
        # train the model for train_times, and get the mean and std of the results
        temp_results = []
        start_time = time.time()
        # if model_config['name'] not in ['GCN', 'GCN_with_edge_attr', 'GAT', 'GAT_with_edge_attr', 'GraphSAGE']:
        if 'my' in model_config['name']:
            for i in range(train_times):
                model = model_config['model'](
                    num_features=gnn_dims['input of each node'],
                    num_outputs=gnn_dims['output of each node'],
                    **model_config['hyperparameters']
                ).to(device)
                print(f'\nTraining custom GNN model: {model_config["name"]} for the {i+1}th time')
                temp_results.append(model.train_model(
                    graph_features['train'],
                    graph_features['val'],
                    graph_features['test'],
                    epochs=config.epochs
                ))
        else:
            for i in range(train_times):
                model = model_config['model'](
                    in_channels=gnn_dims['input of each node'],
                    out_channels=gnn_dims['output of each node'],
                    **model_config['hyperparameters']
                ).to(device)
                print(f'\nTraining predefined GNN model: {model_config["name"]} for the {i+1}th time')
                temp_results.append(model.train_model(
                    graph_features['train'],
                    graph_features['val'],
                    graph_features['test'],
                    epochs=config.epochs,
                    use_edge_attr=model_config['use_edge_attr']
                ))
        end_time = time.time()
        avg_time = (end_time - start_time) / train_times
        
        results['GNN'][model_config['name']] = {
            'training_loss': {
                'mean': np.mean([temp_result['training_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['training_loss']for temp_result in temp_results], axis=0)
            },
            'validation_loss': {
                'mean': np.mean([temp_result['validation_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['validation_loss']for temp_result in temp_results], axis=0)
            },
            'test_loss': {
                'mean': np.mean([temp_result['test_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['test_loss']for temp_result in temp_results], axis=0)
            },
            'time (in sec)': avg_time
        }
    
    # Train traditional ML models
    # ml_models = [
    #     {'name': 'LinearRegression', 'model': LinearRegression, 'hyperparameters': {}},
    #     {'name': 'RandomForest', 'model': RandomForestRegression, 'hyperparameters': {'n_estimators': 100,'max_depth': None}}
    # ]
    ml_models = config.ML_model_configs
    for model_config in ml_models:
        temp_results = []
        start_time = time.time()
        for i in range(train_times):
            if model_config['name'] == 'RandomForest':
                model = model_config['model'](
                    tabular_dims['input'],
                    tabular_dims['output'],
                    **model_config['hyperparameters']
                )
            else:
                model = model_config['model'](
                    tabular_dims['input'],
                    tabular_dims['output'],
                    **model_config['hyperparameters']
                ).to(device)

            print(f'\nTraining ML model: {model_config["name"]} for the {i+1}th time')
            temp_results.append(model.train_model(
                tabular_features['train'],
                tabular_features['val'],
                tabular_features['test'],
                epochs=config.epochs
            ))
        end_time = time.time()
        avg_time = (end_time - start_time) / train_times
        results['ML'][model_config['name']] = {
            'training_loss': {
                'mean': np.mean([temp_result['training_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['training_loss']for temp_result in temp_results], axis=0)
            },
            'validation_loss': {
                'mean': np.mean([temp_result['validation_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['validation_loss']for temp_result in temp_results], axis=0)
            },
            'test_loss': {
                'mean': np.mean([temp_result['test_loss']for temp_result in temp_results], axis=0),
                'std': np.std([temp_result['test_loss']for temp_result in temp_results], axis=0)
            },
            'time (in sec)': avg_time
        }
    return results

def save_results(results):
    """Save train and dev losses over epochs into a file for each model, save test losses (mean and std) and avg training time to a CSV file."""
    os.makedirs(os.path.join(PATHS['results'], f'train_loss/'), exist_ok=True)
    os.makedirs(os.path.join(PATHS['results'], f'val_loss/'), exist_ok=True)
    for model_type in results.keys():
        for model_name in results[model_type].keys():
            train_losses_path = os.path.join(PATHS['results'], f'train_loss/{model_name}.csv')
            pd.DataFrame([results[model_type][model_name]['training_loss']['mean'], results[model_type][model_name]['training_loss']['std']]).to_csv(train_losses_path)
            val_losses_path = os.path.join(PATHS['results'], f'val_loss/{model_name}.csv')
            pd.DataFrame([results[model_type][model_name]['validation_loss']['mean'], results[model_type][model_name]['validation_loss']['std']]).to_csv(val_losses_path)
    # for model_type, results in results.items():
    #     for model_name, metrics in results.items():
    #         if metrics and 'training_loss' in metrics:
    #             train_losses_path = os.path.join(PATHS['results'], f'train_loss/{model_name}.csv')
    #             pd.DataFrame([metrics['training_loss']['mean'], metrics['training_loss']['std']]).to_csv(train_losses_path)
    #         if metrics and 'validation_loss' in metrics:
    #             val_losses_path = os.path.join(PATHS['results'], f'val_loss/{model_name}.csv')
    #             pd.DataFrame([metrics['validation_loss']['mean'], metrics['validation_loss']['std']]).to_csv(val_losses_path)
    
    test_losses_path = os.path.join(PATHS['results'], 'test_losses.csv')

    # for model_name, metrics in results.items():
    #     print(f'model_name: {model_name}; metrics: {metrics}')

    test_losses = {
        f"{model_name}": [results[model_type][model_name]['test_loss']['mean'], results[model_type][model_name]['test_loss']['std'], results[model_type][model_name]['time (in sec)']] 
        for model_type in results.keys()
        for model_name in results[model_type].keys()
    }
    test_losses_df = pd.DataFrame.from_dict(test_losses, orient='index', columns=['mean', 'std', 'time (in sec)'])
    test_losses_df.to_csv(test_losses_path)

    time_results_path = os.path.join(PATHS['results'], 'time_results.csv')
    time_results = {
        f"{model_name}": [results[model_type][model_name]['time (in sec)']]
        for model_type in results.keys()
        for model_name in results[model_type].keys()
    }
    time_results_df = pd.DataFrame.from_dict(time_results, orient='index', columns=['time (in sec)'])
    time_results_df.to_csv(time_results_path)

def analyze_graph_connectivity(graph_features):
    """Analyze connectivity and edge attributes for each node in the datasets.
    
    Args:
        graph_features: Dictionary containing train, val, and test graph data
    """
    print("\n=== Analyzing Graph Connectivity ===")
    
    # Load station names
    weather_data = load_weather_data(file_path=PATHS['weather_data'])
    station_names = get_station_names(weather_data)
    print(f"Number of stations in metadata: {len(station_names)}")
    
    for dataset_name, dataloader in graph_features.items():
        if not dataloader:
            print(f"\n{dataset_name.upper()} set is empty!")
            continue
            
        print(f"\n{dataset_name.upper()} Set Analysis:")
        print(f"Number of timesteps: {len(dataloader)}")
        
        # Get number of nodes from first graph
        num_nodes = dataloader.dataset[0].x.shape[0]
        print(f"Number of nodes per graph: {num_nodes}")
        
        # Initialize node statistics
        node_stats = {i: {
            'name': station_names[i] if i < len(station_names) else f"Unknown_{i}",
            'has_edge_attr': False,
            'total_outgoing': 0,
            'total_incoming': 0,
            'avg_out_degree': 0,
            'avg_in_degree': 0,
            'timesteps_with_edges': 0
        } for i in range(num_nodes)}
        
        # Print warning if there's a mismatch
        if num_nodes != len(station_names):
            print(f"\nWARNING: Mismatch between number of stations ({len(station_names)}) "
                  f"and number of nodes in graph ({num_nodes})")
        
        # Analyze each timestep
        for t, data in enumerate(dataloader.dataset):
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            
            # Count edges for each node
            for i in range(edge_index.shape[1]):
                source = edge_index[0, i].item()
                target = edge_index[1, i].item()
                
                node_stats[source]['total_outgoing'] += 1
                node_stats[target]['total_incoming'] += 1
                
                if edge_attr is not None:
                    node_stats[source]['has_edge_attr'] = True
                    node_stats[target]['has_edge_attr'] = True
            
            # Track timesteps with edges
            for node in range(num_nodes):
                if node in edge_index[0] or node in edge_index[1]:
                    node_stats[node]['timesteps_with_edges'] += 1
        
        # Calculate averages
        for node in node_stats:
            node_stats[node]['avg_out_degree'] = node_stats[node]['total_outgoing'] / len(dataloader)
            node_stats[node]['avg_in_degree'] = node_stats[node]['total_incoming'] / len(dataloader)
        
        # Print detailed analysis with station names
        print("\nPer-Node Statistics:")
        print("-" * 120)
        print(f"{'Node':<6} {'Station':<15} {'Has Attr':<10} {'Total Out':<10} {'Total In':<10} "
              f"{'Avg Out':<10} {'Avg In':<10} {'Active Timesteps':<15}")
        print("-" * 120)
        
        nodes_without_edges = []
        highly_connected = []
        
        for node, stats in node_stats.items():
            total_edges = stats['total_outgoing'] + stats['total_incoming']
            print(f"{node:<6} {stats['name']:<15} {str(stats['has_edge_attr']):<10} "
                  f"{stats['total_outgoing']:<10} {stats['total_incoming']:<10} "
                  f"{stats['avg_out_degree']:<10.2f} {stats['avg_in_degree']:<10.2f} "
                  f"{stats['timesteps_with_edges']:<15}")
            
            if total_edges == 0:
                nodes_without_edges.append((node, stats['name']))
            elif total_edges > len(dataloader) * 6:  # More than average
                highly_connected.append((node, stats['name'], total_edges))
        
        # Print summary
        print(f"\nSummary for {dataset_name} set:")
        print(f"- Total timesteps: {len(dataloader)}")
        print(f"- Total nodes: {num_nodes}")
        
        if nodes_without_edges:
            print("\nNodes without any edges:")
            for node, name in nodes_without_edges:
                print(f"- Node {node} ({name})")
        # if highly_connected:
        #     print("\nHighly connected nodes:")
        #     for node, name, edges in sorted(highly_connected, key=lambda x: x[2], reverse=True):
        #         print(f"- Node {node} ({name}): {edges} total edges "
        #               f"({edges/len(dataloader):.2f} edges per timestep)")
        
        # # Edge attribute statistics if available
        # if hasattr(dataloader.dataset[0], 'edge_attr'):
        #     all_edge_attr = torch.cat([d.edge_attr for d in dataloader.dataset])
        #     print("\nEdge Attribute Statistics:")
        #     print(f"- Min: {all_edge_attr.min().item():.2f}")
        #     print(f"- Max: {all_edge_attr.max().item():.2f}")
        #     print(f"- Mean: {all_edge_attr.mean().item():.2f}")
        #     print(f"- Std: {all_edge_attr.std().item():.2f}")

def main(neighbors_per_station, time_series, temporal_window, train_times, batch_size):
    """Run the weather prediction experiment."""
    
    print("\n=== Starting Weather Prediction Experiment ===")
    
    # 1. Create and visualize weather station network
    print("\n1. Setting up Weather Station Network...")
    station_data = load_and_process_data(PATHS['station'])
    print(f"- Loaded {len(station_data)} weather stations")
    
    node_coor, edge_index, edge_attr = create_graph_data(
        station_data,
        neighbors_per_station=neighbors_per_station,
        path=PATHS['features']
    )
    print("- Created graph structure")
    print("- Saved network features")
    
    plot_network_graph(station_data, edge_index, path=PATHS['features'])
    print("- Generated network visualization")

    # 2. Prepare dataset
    print("\n2. Preparing Dataset...")
    # config = {
    #     'time_series': time_series,
    #     'temporal_window': temporal_window
    # }
    weather_data = load_weather_data(file_path=PATHS['weather_data'])
    dataset_indices = weather_data.index.tolist()
    train_indices, val_indices, test_indices = split_data(
        dataset_indices, 
        time_series, 
        temporal_window
    ) # TODO: Random split -> Time-based split
    print(f"- Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # 3. Generate features
    print("\n3. Generating Features...")
    _ = create_graph_dataset(
        weather_data, 
        node_coor,
        edge_index, 
        edge_attr, 
        threshold=0.8, 
        graph_features_path=PATHS['graph_features']
    )

    graph_features = get_graph_features(
        train_indices, val_indices, test_indices,
        graph_features_path=PATHS['graph_features'],
        time_series=time_series,
        k=temporal_window,
        batch_size=batch_size
    ) # return a dictionary with the train, val, and test graph features wrapped in DataLoader
    print("- Created graph features")
        
    # Analyze graph connectivity
    analyze_graph_connectivity(graph_features)
    # print(f'graph_features["train"].dataset[0].y.shape: {graph_features["train"].dataset[0].y.shape}')
    gnn_dimensions = {
        'input of each node': graph_features['train'].dataset[0].x.shape[1],
        'output of each node': 1,
        'output of each graph': graph_features['train'].dataset[0].y.shape[0]
    }
    print(f'GNN dimensions: {gnn_dimensions}')
    print(f"- GNN dimensions:")
    
    for train_batch in graph_features['train']:
        print(f'Sample train batch shape: {train_batch.x.shape}')
        print(f'Sample train batch edge_index shape: {train_batch.edge_index.shape}')
        print(f'Sample train batch edge_attr shape: {train_batch.edge_attr.shape}')
        print(f'Sample train batch y shape: {train_batch.y.shape}')
        break

    for key, value in gnn_dimensions.items():
        print(f"{key}: {value}")

    tabular_features = prepare_tabular_features(graph_features)
    tabular_dimensions = {
        'input': tabular_features['train']['x'].shape[1],
        'output': (tabular_features['train']['y'].shape[1] 
                  if len(tabular_features['train']['y'].shape) > 1 else 1)
    }
    print(f"- Tabular dimensions:")
    for key, value in tabular_dimensions.items():
        print(f"{key}: {value}")

    # 4. Train and evaluate models
    print("\n4. Training and Evaluating Models...")
    results = train_and_evaluate_models(
        graph_features, gnn_dimensions,
        tabular_features, tabular_dimensions,
        train_times
    )
    print("- Completed model training and evaluation")

    # 5. Generate visualizations and save results
    print("\n5. Saving Results...")
    print("- Generating visualizations")
    visualize_results(results, PATHS['results'])
    
    print("- Saving test losses")
    save_results(results) # train losses and val losses over epochs for each model, and all test losses in a csv file
    
    print("\n=== Experiment Completed ===")

if __name__ == "__main__":
    try:
        # print(f'type of temporal_window: {type(config.temporal_window)}')
        save_config()
        if isinstance(config.neighbors_per_station, list):
            for neighbors_per_station in config.neighbors_per_station:
                PATHS['results'] = f'{PROJECT_ROOT}/results/{current_time}/{neighbors_per_station}nn/'
                os.makedirs(PATHS['results'], exist_ok=True)
                main(neighbors_per_station, config.time_series, config.temporal_window, config.train_times, config.batch_size)
        elif isinstance(config.temporal_window, list):
            for temporal_window in config.temporal_window:
                PATHS['results'] = f'{PROJECT_ROOT}/results/{current_time}/{temporal_window}d/'
                os.makedirs(PATHS['results'], exist_ok=True)
                main(config.neighbors_per_station, config.time_series, temporal_window, config.train_times, config.batch_size)
        elif isinstance(config.batch_size, list):
            for batch_size in config.batch_size:
                PATHS['results'] = f'{PROJECT_ROOT}/results/{current_time}/bs{batch_size}/'
                os.makedirs(PATHS['results'], exist_ok=True)
                main(config.neighbors_per_station, config.time_series, config.temporal_window, config.train_times, batch_size)
        else:
            main(config.neighbors_per_station, config.time_series, config.temporal_window, config.train_times, config.batch_size)
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()