## This code is going to be used to process the data for the competition
import pandas as pd
import numpy as np
from prophet import Prophet
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend for matplotlib
import matplotlib.pyplot as plt

# Set file paths and initial display year at the top for easy access and modification
#### PLEASE NOTE THAT THESE PATHS ARE UNIQUE TO EACH COMPUTER
# To use this data please download them from the following links and modify your own path
#Competition Data can be found at: https://drive.google.com/drive/folders/1HkOQLoChqjIN82Tjn1aMnIMunjyCEkTp
#Supplementary Data can be found at: https://github.com/owid/co2-data
COMPETITION_DATA_PATH = r"C:\Users\felix\PycharmProjects\pythonProject\.venv\Scripts\DSMLC Final Competition 2024 Dataset.xlsx"
SUPPLEMENTARY_DATA_PATH = r"C:\Users\felix\PycharmProjects\pythonProject\.venv\Scripts\owid-co2-data.csv"
##GEOJSON Spatial data can be found at (please select GeoJSON file format): https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/?location=2,0.17578,-0.17578&basemap=jawg.light&dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6IndvcmxkLWFkbWluaXN0cmF0aXZlLWJvdW5kYXJpZXMiLCJvcHRpb25zIjp7fX0sImNoYXJ0cyI6W3siYWxpZ25Nb250aCI6dHJ1ZSwidHlwZSI6ImNvbHVtbiIsImZ1bmMiOiJDT1VOVCIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNGRjUxNUEifV0sInhBeGlzIjoic3RhdHVzIiwibWF4cG9pbnRzIjo1MCwic29ydCI6IiJ9XSwidGltZXNjYWxlIjoiIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D
GEOJSON_FILE_PATH = r'C:\Users\felix\PycharmProjects\pythonProject\.venv\Scripts\world-administrative-boundaries.geojson'
INITIAL_YEAR = 2024  # Year to visualize on the map. Data is avialable for years 1950-2050


# A basic code which removes outliers within 3 standard deviations from the data
def remove_outliers_std(df, column, num_std=3):
    mean = df[column].mean()
    std_dev = df[column].std()
    cutoff = std_dev * num_std
    lower = mean - cutoff
    upper =  mean+cutoff
    return df[(df[column] >= lower) & (df[column] <= upper)]


def PrepareData ():
    '''
    Prepares and combinnes competition and supplmentary data to be forcasted for Prophet
    :return: Dataframe in Prophet format
    '''

    # Load both of the data sets
    competition_data = pd.read_excel(
        r"C:\Users\felix\PycharmProjects\pythonProject\.venv\Scripts\DSMLC Final Competition 2024 Dataset.xlsx")
    supplementary_data = pd.read_csv(r"C:\Users\felix\PycharmProjects\pythonProject\.venv\Scripts\owid-co2-data.csv")

    # Extract the wanted data
    filtered_competition_data = competition_data[
        ["Country", "Code", "Year", "9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)"]]
    filtered_supplementary_data = supplementary_data[["country", "year", "iso_code", "co2_per_gdp"]]

    # Refromat supplementray data so that both data sets are in the same format
    filtered_supplementary_data = filtered_supplementary_data[["country", "iso_code", "year", "co2_per_gdp"]]
    filtered_supplementary_data = filtered_supplementary_data.rename(
        columns={"country": "Country", "iso_code": "Code", "year": "Year",
                 "co2_per_gdp": "9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)"})

    #Comp data only includes data from 2000 onwards, use suppplementary data from years before 2000s
    before_2000 = filtered_supplementary_data[filtered_supplementary_data['Year'] < 2000]
    from_2000_onwards = filtered_competition_data[filtered_competition_data['Year'] >= 2000]

    # Combine the data sets
    combined_data = pd.concat([before_2000,from_2000_onwards], ignore_index=True)

    # Sort the combined dataset alphabetically by 'Country' and then by 'Year' so data is in original format
    combined_data = combined_data.sort_values(by=['Country', 'Year'])

    # Reset the index of the sorted DataFrame
    combined_data.reset_index(drop=True, inplace=True)

    ## Drop Null and 0's and outliers
    combined_data.replace(0,np.nan,inplace=True)
    combined_data = combined_data.dropna(subset=['9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)'])
    combined_data = combined_data.dropna(subset=['Code'])
    combined_data =  remove_outliers_std(combined_data,'9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)')



    #Now that data is all in one file, prepare it to be passed to prophet by reformatting data
    combined_data["ds"] = pd.to_datetime(combined_data['Year'], format='%Y')
    combined_data.rename(columns={"9.4.1 - Annual CO₂ emissions per GDP (kg per international-$)" : 'y'}, inplace=True)



    ## This line is used to save the final file to ones computer, it is not neccesary but can be usefull,
    ##Uncomment the line to save file to computer

    #combined_data.to_csv(r'C:\Users\felix\Downloads\cleaned_dataset.csv', index=False)

    return combined_data


def forcaster(df):
    # Get a list of unique codes from the data and preform analysis for each code
    unique_codes = df['Code'].unique()
    forecasts = {}

    for code in unique_codes:
        # Filter data for the current code
        code_data = df[df['Code'] == code][['ds', 'y']]

        # Fit the model on the training dataset
        m = Prophet(yearly_seasonality=False)
        m.fit(code_data)  # only include the 'ds' and 'y' columns

        # Create a dataframe for future predictions that includes the test data years
        future = m.make_future_dataframe(periods=40, freq='YS')

        # Predict on the future dates
        forecast = m.predict(future)

        # Store the forecast
        forecasts[code] = forecast

    all_forecasts_df = pd.concat([df.assign(country=code) for code, df in forecasts.items()],
                                 ignore_index=True)

    # Write the combined DataFrame to a CSV file
    all_forecasts_df.to_csv(r'C:\Users\felix\Downloads\forecasts.csv', index=False)

    return all_forecasts_df


# Plot the map for a given year using geopandas
def plot_map_for_year(ax, data, year):
    # Filter the data for the selected year
    yearly_data = data[data['ds'].dt.year == year]  # Ensure we compare just the year part
    if not yearly_data.empty:
        ax.clear()  # Clear existing data before re-plotting
        # Use the 'legend_kwds' argument to customize the legend
        yearly_data.plot(ax=ax, column='trend', cmap="OrRd", legend=True,
                         legend_kwds={'label': "CO2 Emissions Per GDP",
                                      'orientation': "horizontal"})
        ax.axis('off')
        ax.set_title(f'CO2 Emissions by Country in {year}', fontdict={'fontsize': 20, 'fontweight': '3'})
    plt.draw()


# Main function to create the figure using geopandas and matplotlib
def MakeFigure(geojson_file_path, co2Data):
    # Load GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(geojson_file_path)
    co2Data = co2Data[['country', 'ds', 'trend']]

    # Merge the GeoDataFrame with co2Data on 'iso3' which is assumed to be the common key
    merged_data = gdf.merge(co2Data, left_on='iso3', right_on='country')

    # Create the figure and axis objects for the choropleth map
    fig, ax = plt.subplots(1, figsize=(15, 10))

    # Show the map for the initial year specified by INITIAL_YEAR
    plot_map_for_year(ax, merged_data, INITIAL_YEAR)

    # Show the plot
    plt.show()

# Execute the data preparation and forecasting, then generate the map
CleanData = PrepareData()
Predictions = forcaster(CleanData)
MakeFigure(GEOJSON_FILE_PATH, Predictions)