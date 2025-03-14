import requests
import pandas as pd
import time

def decimal_to_dms(decimal_coord):
    """Convert decimal degrees to degrees:minutes:seconds format"""
    # Get the sign
    sign = '+' if decimal_coord >= 0 else '-'
    decimal_coord = abs(decimal_coord)
    
    # Get degrees
    degrees = int(decimal_coord)
    
    # Get minutes and seconds
    minutes_decimal = (decimal_coord - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = int((minutes_decimal - minutes) * 60)
    
    # Format with leading zeros
    # For latitude: ±DD:MM:SS
    # For longitude: ±DDD:MM:SS
    if len(str(degrees)) <= 2:  # Latitude
        return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:02d}"
    else:  # Longitude
        return f"{sign}{degrees:03d}:{minutes:02d}:{seconds:02d}"

def get_station_info():
    # Dictionary of known station information
    # Format: 'name': [ID, Name, Country, Lat, Lon, Height]
    # Note: Some IDs and heights might need verification
    stations = {
        'BASEL': [239, 'BASEL BINNINGEN', 'CH', 47.5333, 7.5833, 316],
        'BUDAPEST': [64, 'BUDAPEST', 'HU', 47.5111, 19.0208, 153],
        'DE_BILT': [260, 'DE BILT', 'NL', 52.1017, 5.1778, 2],  # KNMI station
        'DRESDEN': [43, 'DRESDEN WAHNSDORF', 'DE', 51.1167, 13.6833, 246],
        'DUSSELDORF': [91, 'DUSSELDORF', 'DE', 51.2277, 6.7735, 37],  # Airport elevation
        'HEATHROW': [1860, 'HEATHROW', 'GB', 51.4789, -0.4494, 25],
        'KASSEL': [480, 'KASSEL', 'DE', 51.2978, 9.4437, 231],
        'MAASTRICHT': [168, 'MAASTRICHT', 'NL', 50.9053, 5.7617, 114],
        'MALMO': [5175, 'MALMO', 'SE', 55.6100, 13.0800, 20],
        'MONTELIMAR': [786, 'MONTELIMAR', 'FR', 44.5811, 4.7331, 73],
        'MÜNCHEN': [52, 'MUENCHEN', 'DE', 48.1642, 11.5431, 515],
        'OSLO': [193, 'OSLO BLINDERN', 'NO', 59.9423, 10.7200, 94],
        'PERPIGNAN': [36, 'PERPIGNAN', 'FR', 42.7372, 2.8706, 42],
        'ROMA': [176, 'ROMA CIAMPINO', 'IT', 41.7833, 12.5833, 105],
        'SONNBLICK': [15, 'SONNBLICK', 'AT', 47.0500, 12.9500, 3106],
        'STOCKHOLM': [10, 'STOCKHOLM', 'SE', 59.3500, 18.0500, 44],
        'TOURS': [2190, 'TOURS', 'FR', 47.4439, 0.7278, 108],
        'LJUBLJANA': [228, 'LJUBLJANA BEZIGRAD', 'SI', 46.0656, 14.5125, 299]
    }
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(stations, orient='index',
                               columns=['ID', 'Name', 'country', 'lat', 'lon', 'height'])
    
    # Reset index and format columns
    df = df.reset_index(drop=True)
    
    # Convert decimal coordinates to DMS format
    df['lat'] = df['lat'].apply(decimal_to_dms)
    df['lon'] = df['lon'].apply(decimal_to_dms)
    
    # Pad the Name column to match format
    df['Name'] = df['Name'].str.ljust(40)
    
    return df

def main():
    # Generate the dataset
    df = get_station_info()
    
    # Save to CSV
    output_path = 'weather_prediction_dataset-main/cs886_project/data/all_stations.csv'
    df.to_csv(output_path, index=False)
    
    print("\nStation data saved to:", output_path)
    print("\nDataset preview:")
    print(df)

if __name__ == "__main__":
    main()