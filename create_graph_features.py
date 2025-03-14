"""
Utility functions for creating graph features from weather data.
"""
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import pickle
import config

PROJECT_ROOT = config.PROJECT_ROOT

def load_graph_data():
    """Load the graph structure data from files"""
    print("\nLoading graph data...")
    
    # Load data
    edge_index = np.load(f'{PROJECT_ROOT}/features/edge_index.npy')
    edge_attr = np.load(f'{PROJECT_ROOT}/features/edge_attr.npy') 
    node_coor = np.load(f'{PROJECT_ROOT}/features/node_coor.npy')
    
    print("\nInitial data shapes:")
    print(f"Edge Index: {edge_index.shape}")
    print(f"Edge Attributes: {edge_attr.shape}")
    print(f"Node Coordinates: {node_coor.shape}")
    
    # Verify edge_index format
    print("\nEdge index format check:")
    print(f"Edge index dtype: {edge_index.dtype}")
    print(f"Min node index: {edge_index.min()}")
    print(f"Max node index: {edge_index.max()}")
    print(f"Number of nodes: {len(node_coor)}")
    
    # Check edge consistency
    print("\nEdge consistency check:")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Number of edge attributes: {len(edge_attr)}")
    
    # Check edge distribution
    for node in range(len(node_coor)):
        out_edges = np.sum(edge_index[0] == node)
        in_edges = np.sum(edge_index[1] == node)
        print(f"Node {node}: {out_edges} outgoing, {in_edges} incoming edges")
    
    # Verify edge attributes
    print("\nEdge attribute check:")
    print(f"Min distance: {edge_attr.min()}")
    print(f"Max distance: {edge_attr.max()}")
    print(f"Mean distance: {edge_attr.mean()}")
    
    return edge_index, edge_attr, node_coor

def load_weather_data(file_path='weather_prediction_dataset-main/cs886_project/data/weather_prediction_dataset.csv'):
    """Load the weather prediction dataset"""
    return pd.read_csv(file_path)

def get_station_names(weather_data):
    """Extract unique station names from dataset columns in order of appearance"""
    station_names = []
    for col in weather_data.columns:
        if '_' in col:
            station_name = 'DE_BILT' if col.startswith('DE_BILT') else col.split('_')[0]
            if station_name not in station_names:  # Preserve order
                station_names.append(station_name)
    return station_names  # Return list instead of set to maintain order

def get_unified_features(row, station, selected_features, feature_means):
    """Get unified feature vector for a station"""
    features = []
    for feature in selected_features:
        col_name = f"{station}_{feature}"
        features.append(row[col_name] if col_name in row.index 
                       else feature_means[feature])
    return features

def create_graph_dataset(weather_data, edge_index, edge_attr, threshold=0.8, graph_features_path=f'{PROJECT_ROOT}/features/graph_features.pkl'):
    """Create graph dataset from weather data and graph structure.
    
    Args:
        weather_data: DataFrame containing weather data
        edge_index: Graph edge connections
        edge_attr: Edge attributes
        station_names: List of station names in order
        threshold: Minimum percentage (0.0-1.0) of stations that must have a feature (default: 0.8)
        graph_features_path: Path to save graph features
    """
    # Get feature counts
    station_names = get_station_names(weather_data)
    all_features = set()

    for station in station_names:
        station_features = [col[len(station)+1:] for col in weather_data.columns 
                          if col.startswith(station)]
        all_features.update(station_features)

    # Calculate minimum count based on percentage threshold
    min_station_count = int(len(station_names) * threshold)
    print(f"\nFeature Selection:")
    print(f"Total stations: {len(station_names)}")
    print(f"Threshold: {threshold*100:.1f}% ({min_station_count} stations)")

    feature_counts = {}
    selected_features = []
    for feature in all_features:
        count = sum(1 for station in station_names 
                   if any(col.startswith(station) and col.endswith(feature) 
                         for col in weather_data.columns))
        feature_counts[feature] = count
        
        # Select feature if it appears in enough stations
        if count >= min_station_count:
            selected_features.append(feature)
    
    print(f"Selected {len(selected_features)} features that appear in at least {threshold*100:.1f}% of stations")
    print("Selected features:", selected_features)
            
    # Calculate feature means
    feature_means = {}
    for feature in selected_features:
        feature_values = []
        for station in station_names:
            col_name = f"{station}_{feature}"
            if col_name in weather_data.columns:
                feature_values.extend(weather_data[col_name].values)
        feature_means[feature] = np.mean(feature_values)
    
    data_list = []
    
    # Don't sort station names - use them in original order
    for idx in range(len(weather_data)):
        node_features_list = []
        for station in station_names:  # Use original order
            unified_features = get_unified_features(weather_data.iloc[idx], station, 
                                                 selected_features, feature_means)
            node_features_list.append(unified_features)
        
        node_features = torch.tensor(node_features_list, dtype=torch.float)
        data = Data(
            x=node_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            feature_names=selected_features,
            station_names=station_names
        )
        data_list.append(data)
    
    save_graph_features(data_list, graph_features_path)
    return data_list

def save_graph_features(data_list, filename=f'{PROJECT_ROOT}/features/graph_features.pkl'):
    """Save the dataset to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(data_list, f)

def load_graph_features(filename=f'{PROJECT_ROOT}/features/graph_features.pkl'):
    """Load the dataset from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_dataset_stats(data_list):
    """Print statistics about the dataset"""
    print(f"\nDataset contains {len(data_list)} graph objects")
    print(f"Each graph has {data_list[0].num_nodes} nodes and {data_list[0].num_edges} edges")
    print(f"Node feature dimensions: {data_list[0].x.shape}")
    print(f"Feature names: {data_list[0].feature_names}")
    print(f"Station names: {data_list[0].station_names}")
    print(data_list[0])

def get_x(data_list, index, k):
    # return the stack of concatenated graph features of the [index-k:index] of each station
    # Get features from index-k to index for each station
    station_features = []
    for station in data_list[index].station_names:
        # Get station index
        station_idx = data_list[index].station_names.index(station)
        
        # Collect features for this station across time steps
        station_time_features = []
        for t in range(index-k, index):
            station_time_features.append(data_list[t].x[station_idx])
            
        # Concatenate features across time steps
        station_features.append(torch.cat(station_time_features))
        
    # Stack features for all stations
    return torch.stack(station_features)

def get_feature_names(data_list, index, k):
    # return the concatenated feature_names of the [index-k:index] in a list
    return [f'{feature_name}_{idx}' for idx in range(index-k, index) for feature_name in data_list[idx].feature_names]

def get_y(data_list, index):
    # Get temperature means for all 18 regions
    temp_means = []
    for station in data_list[index].station_names:
        station_idx = data_list[index].station_names.index(station)
        feature_idx = data_list[index].feature_names.index('temp_mean')
        temp_means.append(data_list[index].x[station_idx][feature_idx])
    
    # Return as a tensor of shape [18] (one value per region)
    return torch.tensor(temp_means, dtype=torch.float)

def get_y_names(data_list, index):
    # return the name of y (temp_mean_station_name_index)
    return [f'temp_mean_{station}_{index}' for station in data_list[index].station_names]

def get_graph_features(train_indexes, val_indexes, test_indexes, graph_features_path, time_series, k, batch_size=1):
    data_list = load_graph_features(graph_features_path)
    
    # Get station names from both sources
    station_df = pd.read_csv(f'{PROJECT_ROOT}/data/all_stations.csv')
    graph_station_names = station_df['Name'].str.strip().tolist()
    weather_station_names = data_list[0].station_names
    
    # Create mapping between different name formats
    name_mapping = {
        'BASEL BINNINGEN': 'BASEL',
        'DE BILT': 'DE_BILT',
        'DRESDEN WAHNSDORF': 'DRESDEN',
        'OSLO BLINDERN': 'OSLO',
        'ROMA CIAMPINO': 'ROMA',
        'LJUBLJANA BEZIGRAD': 'LJUBLJANA'
    }
    
    # Verify node order matches
    print("\nVerifying node attribute mapping:")
    for i, (graph_name, weather_name) in enumerate(zip(graph_station_names, weather_station_names)):
        graph_name = name_mapping.get(graph_name, graph_name)
        if graph_name != weather_name:
            print(f"WARNING: Mismatch at index {i}: Graph={graph_name}, Weather={weather_name}")
    
    train_data = []
    val_data = []
    test_data = []

    # Create data objects for training set
    for index in train_indexes:
        x = get_x(data_list, index, k)
        edge_index = data_list[index-k].edge_index
        edge_attr = data_list[index-k].edge_attr
        y = get_y(data_list, index)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            feature_names=get_feature_names(data_list, index, k),
            station_names=data_list[index-k].station_names,
            y_names=get_y_names(data_list, index)
        )
        train_data.append(data)

    # Create data objects for validation set
    for index in val_indexes:
        x = get_x(data_list, index, k)
        edge_index = data_list[index-k].edge_index
        edge_attr = data_list[index-k].edge_attr
        y = get_y(data_list, index)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            feature_names=get_feature_names(data_list, index, k),
            station_names=data_list[index-k].station_names,
            y_names=get_y_names(data_list, index)
        )
        val_data.append(data)

    # Create data objects for test set
    for index in test_indexes:
        x = get_x(data_list, index, k)
        edge_index = data_list[index-k].edge_index
        edge_attr = data_list[index-k].edge_attr
        y = get_y(data_list, index)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            feature_names=get_feature_names(data_list, index, k),
            station_names=data_list[index-k].station_names,
            y_names=get_y_names(data_list, index)
        )
        test_data.append(data)

    # Print debug info for first training data object
    if train_data:
        print(f'First training data object:')
        print(f'x shape: {train_data[0].x.shape}')
        print(f'edge_index shape: {train_data[0].edge_index.shape}')
        print(f'edge_attr shape: {train_data[0].edge_attr.shape}')
        print(f'y shape: {train_data[0].y.shape}')

    # Save the data lists
    with open(f'{PROJECT_ROOT}/features/train_graph_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{PROJECT_ROOT}/features/val_graph_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{PROJECT_ROOT}/features/test_graph_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    # Create a DataLoader for each set
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def main():
    # Load data
    edge_index, edge_attr, node_coor = load_graph_data()
    weather_data = load_weather_data(file_path=f'{PROJECT_ROOT}/data/weather_prediction_dataset.csv')
    
    # Process features
    station_names = get_station_names(weather_data)
    print("\nStation names from weather data:")
    print(station_names)
    
    # Load original station data to get node order
    station_df = pd.read_csv(f'{PROJECT_ROOT}/data/all_stations.csv')
    print("\nStation names from graph construction (in order):")
    print(station_df['Name'].str.strip().tolist())
    
    print("\nNumber of nodes in graph:", len(node_coor))
    print("Number of stations in weather data:", len(station_names))
    
    # Create and save dataset
    data_list = create_graph_dataset(weather_data, edge_index, edge_attr, station_names)
    save_graph_features(data_list)
    
    # Load and verify
    loaded_data = load_graph_features()
    print_dataset_stats(loaded_data)

if __name__ == "__main__":
    main()