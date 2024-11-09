import pandas as pd
import numpy as np
from prophet import Prophet
import geopandas as gpd
import matplotlib

matplotlib.use('TkAgg')  # Use the Tkinter backend for matplotlib
import matplotlib.pyplot as plt

# File paths
COMPETITION_DATA_PATH = r"C:\path\to\DSMLC Final Competition 2024 Dataset.xlsx"
SUPPLEMENTARY_DATA_PATH = r"C:\path\to\owid-co2-data.csv"
GEOJSON_FILE_PATH = r'C:\path\to\world-administrative-boundaries.geojson'
YEAR = 2050  # Year you wish to visualize, NOTE: data avialable for 2024-2050


def remove_outliers_std(df, column, num_std=3):
    """
    Remove outliers from a DataFrame based on a specified number of standard deviations.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - column (str): Column to remove outliers from.
    - num_std (int): Number of standard deviations to use for cutoff.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    mean = df[column].mean()
    std_dev = df[column].std()
    cutoff = std_dev * num_std
    lower, upper = mean - cutoff, mean + cutoff
    return df[(df[column] >= lower) & (df[column] <= upper)]


def PrepareData():
    """
    Prepares and combines competition and supplementary data for use with the Prophet model.

    Returns:
    - pd.DataFrame: Combined DataFrame in a format compatible with Prophet.
    """
    # Load data
    competition_data = pd.read_excel(COMPETITION_DATA_PATH)
    supplementary_data = pd.read_csv(SUPPLEMENTARY_DATA_PATH)

    # Filter and rename columns for consistency
    filtered_competition_data = competition_data[
        ["Country", "Code", "Year", "9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)"]]
    filtered_supplementary_data = supplementary_data[["country", "year", "iso_code", "co2_per_gdp"]].rename(
        columns={"country": "Country", "iso_code": "Code", "year": "Year",
                 "co2_per_gdp": "9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)"}
    )

    # Split data by year and combine
    before_2000 = filtered_supplementary_data[filtered_supplementary_data['Year'] < 2000]
    from_2000_onwards = filtered_competition_data[filtered_competition_data['Year'] >= 2000]
    combined_data = pd.concat([before_2000, from_2000_onwards], ignore_index=True).sort_values(
        by=['Country', 'Year']).reset_index(drop=True)

    # Clean data
    combined_data.replace(0, np.nan, inplace=True)
    combined_data.dropna(subset=['9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)', 'Code'], inplace=True)
    combined_data = remove_outliers_std(combined_data, '9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)')

    # Prepare data for Prophet
    combined_data["ds"] = pd.to_datetime(combined_data['Year'], format='%Y')
    combined_data.rename(columns={"9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)": 'y'}, inplace=True)

    # Optionally save cleaned data
    # combined_data.to_csv(r'C:\path\to\cleaned_dataset.csv', index=False)

    return combined_data


def forcaster(df):
    """
    Generate forecasts for CO₂ emissions using Prophet for each country code.

    Parameters:
    - df (pd.DataFrame): DataFrame in Prophet format with columns 'ds', 'y', and 'Code'.

    Returns:
    - pd.DataFrame: Combined forecasts for all countries.
    """
    unique_codes = df['Code'].unique()
    forecasts = {}

    for code in unique_codes:
        # Filter and forecast for each country
        code_data = df[df['Code'] == code][['ds', 'y']]
        m = Prophet(yearly_seasonality=False)
        m.fit(code_data)

        future = m.make_future_dataframe(periods=40, freq='YS')
        forecast = m.predict(future)
        forecasts[code] = forecast

    all_forecasts_df = pd.concat([df.assign(country=code) for code, df in forecasts.items()], ignore_index=True)
    all_forecasts_df.to_csv(r'C:\path\to\forecasts.csv', index=False)

    return all_forecasts_df


def plot_map_for_year(ax, data, year):
    """
    Plot a map showing CO₂ emissions for a specified year.

    Parameters:
    - ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
    - data (pd.DataFrame): DataFrame containing geographical and emissions data.
    - year (int): Year for which the data will be visualized.
    """
    yearly_data = data[data['ds'].dt.year == year]
    if not yearly_data.empty:
        ax.clear()
        yearly_data.plot(ax=ax, column='trend', cmap="OrRd", legend=True,
                         legend_kwds={'label': "CO2 Emissions Per GDP", 'orientation': "horizontal"})
        ax.axis('off')
        ax.set_title(f'CO2 Emissions by Country in {year}', fontdict={'fontsize': 20, 'fontweight': '3'})
    plt.draw()


def MakeFigure(geojson_file_path, co2Data):
    """
    Generate a figure showing CO₂ emissions on a world map for a specified year.

    Parameters:
    - geojson_file_path (str): Path to GeoJSON file for map boundaries.
    - co2Data (pd.DataFrame): DataFrame containing CO₂ emissions data.

    Displays:
    - Choropleth map for the initial year.
    """
    gdf = gpd.read_file(geojson_file_path)
    co2Data = co2Data[['country', 'ds', 'trend']]
    merged_data = gdf.merge(co2Data, left_on='iso3', right_on='country')

    fig, ax = plt.subplots(1, figsize=(15, 10))
    plot_map_for_year(ax, merged_data, YEAR)
    plt.show()


# Execute data processing, forecasting, and visualization
CleanData = PrepareData()
Predictions = forcaster(CleanData)
MakeFigure(GEOJSON_FILE_PATH, Predictions)