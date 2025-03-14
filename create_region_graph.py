"""
Utilities for creating and visualizing the weather station network graph.
Create and save the node coordinates, edge index, and edge attributes.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.neighbors import NearestNeighbors
import config

PROJECT_ROOT = config.PROJECT_ROOT

def to_decimal_coords(old_coordinates):
    """Convert coordinates from degrees:minutes:seconds to decimal degrees."""
    components = old_coordinates.split(':')
    value = float(components[0]) + (float(components[1])/60.0) + (float(components[2])/3600.0)
    return value

def create_basemap():
    """Create a basemap for plotting the European region."""
    return Basemap(
        llcrnrlon=-11., llcrnrlat=35.,
        urcrnrlon=25., urcrnrlat=62.,
        rsphere=(6378137.00, 6356752.3142),
        resolution='l', projection='merc',
        lat_0=45., lon_0=0., lat_ts=20.
    )

def load_and_process_data(filepath):
    """Load and process weather station data.
    
    Args:
        filepath: Path to the CSV file containing station data
    
    Returns:
        DataFrame with processed station information
    """
    df = pd.read_csv(filepath)
    df['decimal_lat'] = df['lat'].apply(to_decimal_coords)
    df['decimal_lon'] = df['lon'].apply(to_decimal_coords)
    df['Name'] = df['Name'].str.strip()
    df['Name'] = df['Name'].str.split().str.get(0)
    # df.sort_values(by=['Name'], inplace=True)
    return df

def calculate_distances(coordinates, k):
    """Calculate k nearest neighbors for each station using Euclidean distance.
    
    Args:
        coordinates: Array of station coordinates (lat, lon)
        k: Number of nearest neighbors to find (including self)
    
    Returns:
        distances: Array of shape (n_stations, k) containing distances to k nearest neighbors
        indices: Array of shape (n_stations, k) containing indices of k nearest neighbors
    """
    # print(f"  Input coordinates shape: {coordinates.shape}")
    
    # Use NearestNeighbors with specific k instead of all stations
    nn_model = NearestNeighbors(
        n_neighbors=k+1,  # +1 because first neighbor is self
        algorithm='ball_tree',  # More efficient for geographic coordinates
        metric='haversine'  # Better for geographic distances
    )
    
    # Convert to radians for haversine distance
    coords_rad = np.radians(coordinates)
    nn_model.fit(coords_rad)
    
    # Find k nearest neighbors
    distances, indices = nn_model.kneighbors(coords_rad)
    
    # Convert distances from radians to kilometers (Earth's radius ≈ 6371 km)
    distances = distances * 6371.0
    
    # print(f"  Output distances shape: {distances.shape}")
    # print(f"  Output indices shape: {indices.shape}")
    
    return distances, indices

def create_edges(distances, indices, neighbors_per_station):
    """Create graph edges by connecting each station to its nearest neighbors.
    
    Args:
        distances: Array of distances to nearest neighbors
        indices: Array of indices of nearest neighbors
        neighbors_per_station: Number of nearest neighbors to connect
    
    Returns:
        edge_index: Array of edge connections [2, num_edges]
        edge_attr: Array of edge distances [num_edges, 1]
    """
    edge_index = []
    edge_attr = []
    
    print(f"  Creating edges for {len(distances)} stations with {neighbors_per_station} neighbors each")
    
    for i in range(len(distances)):
        # Skip first neighbor (self) and take next k neighbors
        for j, dist in zip(indices[i][1:neighbors_per_station+1], 
                         distances[i][1:neighbors_per_station+1]):
            edge_index.append([i, j])
            edge_attr.append([dist])
    
    edge_index = np.array(edge_index).T  # Shape: [2, num_edges]
    edge_attr = np.array(edge_attr)      # Shape: [num_edges, 1]
    
    print(f"  Created edge_index with shape: {edge_index.shape}")
    print(f"  Created edge_attr with shape: {edge_attr.shape}")
    
    return edge_index, edge_attr

def analyze_connectivity(station_df, edge_index, edge_attr):
    """Analyze the connectivity of each station in the network.
    
    Args:
        station_df: DataFrame containing station information
        edge_index: Array of edge connections [2, num_edges]
        edge_attr: Array of edge distances [num_edges, 1]
    """
    print("\nAnalyzing Network Connectivity:")
    num_stations = len(station_df)
    
    # Initialize counters for each station
    incoming_edges = {i: [] for i in range(num_stations)}
    outgoing_edges = {i: [] for i in range(num_stations)}
    
    # Count incoming and outgoing edges for each station
    for edge_idx in range(edge_index.shape[1]):
        source = edge_index[0, edge_idx]
        target = edge_index[1, edge_idx]
        distance = edge_attr[edge_idx, 0]
        
        outgoing_edges[source].append((target, distance))
        incoming_edges[target].append((source, distance))
    
    # Print connectivity information
    print("\nConnectivity Details:")
    print(f"{'Station':<15} {'Out':<5} {'In':<5} {'Total':<6} {'Avg Dist (km)':<12}")
    print("-" * 50)
    
    isolated_stations = []
    highly_connected = []
    
    for i in range(num_stations):
        out_count = len(outgoing_edges[i])
        in_count = len(incoming_edges[i])
        total_edges = out_count + in_count
        
        # Calculate average distance for connected edges
        all_distances = [d for _, d in outgoing_edges[i]] + [d for _, d in incoming_edges[i]]
        avg_distance = np.mean(all_distances) if all_distances else 0
        
        print(f"{station_df.iloc[i]['Name']:<15} {out_count:<5} {in_count:<5} "
              f"{total_edges:<6} {avg_distance:>11.2f}")
        
        # Track special cases
        if total_edges == 0:
            isolated_stations.append(station_df.iloc[i]['Name'])
        elif total_edges > 6:  # More than average connections
            highly_connected.append((station_df.iloc[i]['Name'], total_edges))
    
    # Print summary
    print("\nNetwork Summary:")
    print(f"- Total number of stations: {num_stations}")
    print(f"- Total number of edges: {edge_index.shape[1]}")
    print(f"- Average edges per station: {edge_index.shape[1]/num_stations:.2f}")
    
    if isolated_stations:
        print("\nIsolated Stations (no connections):")
        for station in isolated_stations:
            print(f"- {station}")


def create_graph_data(station_df, neighbors_per_station=3, path=None):
    """Create graph data from station dataframe."""
    print("\nCreating Graph Data:")
    print(f"- Number of stations: {len(station_df)}")
    print(f"- Neighbors per station: {neighbors_per_station}")
    
    # Extract coordinates
    node_coor = np.array(station_df[['decimal_lat', 'decimal_lon']])
    print(f"- Extracted station coordinates with shape {node_coor.shape}")

    
    # Calculate nearest neighbors
    print("- Calculating nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=neighbors_per_station + 2, algorithm='ball_tree').fit(node_coor)
    distances, indices = nbrs.kneighbors(node_coor)
    
    # Create edges
    print("- Creating graph edges...")
    print(f"  Creating edges for {len(station_df)} stations with {neighbors_per_station} neighbors each")
    edge_index = []
    edge_attr = []
    
    for i in range(len(station_df)):
        # Skip first index (self) and take next k neighbors
        for j in indices[i, 1:neighbors_per_station + 1]:
            edge_index.append([i, j])
            edge_attr.append([distances[i, indices[i] == j][0]])
    
    edge_index = np.array(edge_index).T
    edge_index = np.flip(edge_index, axis=0).copy()
    edge_attr = np.array(edge_attr)
    
    print(f"  Created edge_index with shape: {edge_index.shape}")
    print(f"  Created edge_attr with shape: {edge_attr.shape}")
    
    # Analyze network connectivity
    analyze_connectivity(station_df, edge_index, edge_attr)
    
    # Save graph features if path is provided
    if path:
        print("\n- Saving graph features...")
        np.save(f'{path}/node_coor.npy', node_coor)
        np.save(f'{path}/edge_index.npy', edge_index)
        np.save(f'{path}/edge_attr.npy', edge_attr)
        print("  Features saved successfully")
    
    return node_coor, edge_index, edge_attr

def plot_network_graph(df, edge_index, path=None):
    """Visualize the weather station network."""
    print("\nCreating Network Visualization:")
    try:
        if path:
            os.makedirs(path, exist_ok=True)
            print(f"- Using visualization path: {path}")
            
        print("- Setting up plot...")
        plt.figure(figsize=(12, 8))
        
        # Create and setup basemap
        print("- Creating basemap...")
        graph_map = create_basemap()
        graph_map.drawcoastlines()
        graph_map.drawcountries()
        graph_map.fillcontinents(color='lightgray', lake_color='lightblue')
        graph_map.drawmapboundary(fill_color='lightblue')
        
        # Plot stations
        print("- Plotting stations...")
        x, y = graph_map(df['decimal_lon'].values, df['decimal_lat'].values)
        graph_map.scatter(x, y, c='red', s=50, zorder=5)
        print(f"  Plotted {len(df)} stations")
        
        # Plot edges with arrows
        if edge_index is not None:
            print(f"- Adding edges (total: {edge_index.shape[1]})...")
            for i in range(edge_index.shape[1]):
                node1, node2 = edge_index[0, i], edge_index[1, i]
                
                lon1, lat1 = df.iloc[node1][['decimal_lon', 'decimal_lat']]
                lon2, lat2 = df.iloc[node2][['decimal_lon', 'decimal_lat']]
                
                x1, y1 = graph_map(lon1, lat1)
                x2, y2 = graph_map(lon2, lat2)
                
                # Draw edge with arrow
                pos = 0.8
                dx = x2 - x1
                dy = y2 - y1
                arrow_x = x1 + pos * dx
                arrow_y = y1 + pos * dy
                
                plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.2, zorder=4)
                plt.arrow(arrow_x, arrow_y, dx*(1-pos)*0.1, dy*(1-pos)*0.1,
                         head_width=20000, head_length=20000, fc='b', ec='b',
                         alpha=0.2, zorder=4)

        # Add station labels
        print("- Adding station labels...")
        for name, xpt, ypt in zip(df['Name'].tolist(), x, y):
            plt.text(xpt+85000, ypt+35000, name,
                    backgroundcolor='white', fontsize=8)

        plt.title('Weather Station Network Graph (Directed)')
        
        if path:
            print("- Saving visualization...")
            plt.savefig(f'{path}/region_graph.png', dpi=300, bbox_inches='tight')
            print("  Saved to region_graph.png")
        plt.close()
        print("- Visualization completed")
        
    except Exception as e:
        print(f"Error in plot_network_graph: {e}")
        plt.close()

def main():
    """Create and visualize the weather station network."""
    print("\n=== Starting Network Graph Creation ===")
    try:
        # Load and process station data
        print("\n1. Loading Station Data...")
        station_df = load_and_process_data(f'{PROJECT_ROOT}/data/all_stations.csv')
        print(f"- Loaded {len(station_df)} stations")
        
        # Create graph data and visualization
        print("\n2. Creating Graph Structure...")
        node_coor, edge_index, edge_attr = create_graph_data(
            station_df, 
            neighbors_per_station=3,
            path=f'{PROJECT_ROOT}/features/'
        )
        
        if edge_index is not None:
            print("\n3. Creating Visualization...")
            plot_network_graph(
                station_df, 
                edge_index,
                path=f'{PROJECT_ROOT}/features/'
            )
        
        # Clean up memory
        gc.collect()
        plt.close('all')
        print("\n=== Network Graph Creation Completed ===")
        
    except Exception as e:
        print(f"\nError in main: {e}")
        plt.close('all')

if __name__ == "__main__":
    main()
