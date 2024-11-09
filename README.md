# Machine-Learning-Project
This project is designed to preprocess, analyze, and forecast CO₂ emissions data for the DSMLC Final Competition 2024. It integrates multiple data sources, removes outliers, and uses the Prophet model for time series forecasting. Additionally, the project generates a choropleth map to visualize CO₂ emissions per GDP by country over time.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Installation](#installation)
- [File Structure](#file-structure)

## Project Overview
This project performs the following tasks:
1. **Data Preparation**: Merges and cleans competition and supplementary datasets to create a unified dataset suitable for analysis.
2. **Outlier Removal**: Identifies and removes outliers in CO₂ emissions data based on a specified standard deviation threshold.
3. **Forecasting**: Uses the Prophet library to predict future CO₂ emissions trends for each country.
4. **Data Visualization**: Generates a choropleth map to visualize CO₂ emissions per GDP by country for a specific year.

## Data Sources
The project relies on three main data sources:
1. **Competition Data**: [Google Drive](https://drive.google.com/drive/folders/1HkOQLoChqjIN82Tjn1aMnIMunjyCEkTp)
2. **Supplementary CO₂ Data**: [GitHub - OWID CO₂ Data](https://github.com/owid/co2-data)
3. **GeoJSON Spatial Data**: [Public Open DataSoft](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/?location=2,0.17578,-0.17578&basemap=jawg.light)

## Requirements
- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `prophet`
  - `geopandas`
  - `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install the equired libraries:
   ```bash
   pip install -r requirments.txt
3. Download the datasets from the provided links and replace the strings in the code with your local path to the datasets

## File Structure 
DSMlCompetitionCode.py: Main script that performs data preprocessing, forecasting, and visualization.

requirements.txt: Lists all the required libraries for this project.

README.md: Project documentation.
