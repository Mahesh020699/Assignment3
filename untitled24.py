import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err


def clustering_dataset():
    
    
    """
    Perform clustering on the dataset and visualize the results.

    Reads GDP, forest, and population datasets from CSV files.
    Performs scatter matrix plot of the merged dataset.
    Prints silhouette scores for different numbers of clusters.
    Normalizes the selected columns for clustering.
    Performs K-means clustering and assigns cluster labels.
    Visualizes the clusters in a scatter plot.

    Returns:
    None
    """    
    
    
    df = pd.read_csv("GDP.csv", skiprows=4)
    df = df.drop(['Country Code', 'Indicator Code'], axis=1)
    
    df2 = pd.read_csv("forest.csv", skiprows=4)
    df2 = df2.drop(['Country Code', 'Indicator Code'], axis=1)
    
    df3 = pd.read_csv("population.csv", skiprows=4)
    df3 = df3.drop(['Country Code', 'Indicator Code'], axis=1)
    
    df1_year = df[["Country Name", "2019"]].copy()
    df3_year = df3[["Country Name", "2019"]].copy()
    
    df_xx = pd.merge(df1_year, df3_year, on="Country Name", how="outer")
    
    df_xx = df_xx.dropna()
    
    pd.plotting.scatter_matrix(df_xx, figsize=(9.0, 9.0))
    plt.tight_layout()
    plt.show()
    
    print("n score")
    
    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_xx[['2019_x', '2019_y']])
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        print(ncluster, skmet.silhouette_score(df_xx[['2019_x', '2019_y']], labels))
    
    selected_columns = ['2019_x', '2019_y']
    
    df_xx_norm = df_xx[selected_columns].copy()
    scaler = StandardScaler()
    df_xx_norm[selected_columns] = scaler.fit_transform(df_xx_norm[selected_columns])
    
    ncluster = 3
    
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_xx_norm)
    df_xx["xx_cluster"] = kmeans.labels_
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_xx_norm["2019_x"], df_xx_norm["2019_y"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 40, "k", marker="d")
    plt.xlabel("GDP", fontsize=12, fontweight="bold")
    plt.ylabel("population", fontsize=12, fontweight="bold")
    plt.title("Clustering for countries", fontsize=16, fontweight="bold")
    plt.savefig("Clustering_dataset.png", dpi=300)
    plt.show()

 #df_gdp, df_gdp_trans = read_file("gdp_.csv")
 #df_population, df_population_trans = read_file("population_.csv")    


def read_file(file_name):
    
    
    """
   Read a CSV file and process the data.

   Args:
       file_name (str): The name of the CSV file.

   Returns:
       tuple: A tuple containing the original DataFrame and the processed DataFrame.
   """
    
    
    # Read the specified file into a DataFrame
    data_frame = pd.read_csv(file_name)

    # Remove unnecessary columns from the DataFrame
    modified_df = data_frame.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])

    # Replace any missing values with 0
    modified_df = modified_df.replace(np.nan, 0)

    # Rename the 'Country Name' column to 'Year'
    modified_df = modified_df.rename(columns={'Country Name': 'Year'})

    # Transpose the DataFrame
    transposed_df = np.transpose(modified_df)

    # Set the first row as the header
    header = transposed_df.iloc[0].values.tolist()
    transposed_df.columns = header

    # Reset the index and remove the first row
    transposed_df = transposed_df.reset_index().rename(columns={"index": "year"}).iloc[1:]

    # Remove any rows with missing values
    transposed_df = transposed_df.dropna()

    # Extract the year from the 'year' column and convert it to integer
    transposed_df["year"] = transposed_df["year"].str[:4].astype(int)

    # Convert specific columns to numeric
    transposed_df["United States"] = pd.to_numeric(transposed_df["United States"])
    transposed_df["Ireland"] = pd.to_numeric(transposed_df["Ireland"])

    return data_frame, transposed_df


def curve_fun(t, scale, growth):
    """
    Calculate a curve based on a list of values with a specified scale and growth rate.
    
    Parameters
    ----------
    t : list
        List of values.
    scale : float
        Scale of the curve.
    growth : float
        Growth rate of the curve.
    
    Returns
    -------
    c : float
        Result of the curve calculation.
    """
    c = scale * np.exp(growth * (t-1960))
  
    return c


def plot_fit(data_frame, country, indicator, title):
    """
    Plot the data and fitted curve for a specific country and indicator.

    Args:
        data_frame (pd.DataFrame): DataFrame containing the data.
        country (str): Name of the country.
        indicator (str): Name of the indicator.
        title (str): Title for the plot.

    Returns:
        None
    """

    # Fit the curve to the data
    params, cov = opt.curve_fit(curve_fun, data_frame["year"], data_frame[country], p0=[4e8, 0.1])
    sigma = np.sqrt(np.diag(cov))

    # Calculate error ranges
    low, up = err.err_ranges(data_frame["year"], curve_fun, params, sigma)

    # Add fitted values to the DataFrame
    data_frame["fit_value"] = curve_fun(data_frame["year"], *params)

    # Create the plot
    plt.figure()
    plt.title(title, fontweight='bold')

    # Plot the data and fitted curve
    plt.plot(data_frame["year"], data_frame[country], label="data")
    plt.plot(data_frame["year"], data_frame["fit_value"], c="red", label="fit")
    plt.fill_between(data_frame["year"], low, up, alpha=0.3)

    # Configure plot settings
    plt.legend()
    plt.xlim(1990, 2019)
    plt.xlabel("Year", fontweight='bold', fontsize=12)
    plt.ylabel(indicator, fontweight='bold', fontsize=12)
    plt.savefig("GDP_ireland.png", dpi=300)
    plt.show()

def plot_prediction(dframe, country, indicator, title):
    """
    Plot the actual data and predicted values for a specific country and indicator.

    Args:
        dframe (pd.DataFrame): DataFrame containing the data.
        country (str): Name of the country.
        indicator (str): Name of the indicator.
        title (str): Title for the plot.

    Returns:
        None
    """

    # Fit the curve to the data
    params, cov = opt.curve_fit(curve_fun, dframe["year"], dframe[country], p0=[4e8, 0.1])
    sigma = np.sqrt(np.diag(cov))

    # Calculate error ranges
    low, up = err.err_ranges(dframe["year"], curve_fun, params, sigma)

    # Add predicted values to the DataFrame
    dframe["fit_value"] = curve_fun(dframe["year"], *params)

    # Create the plot
    plt.figure()
    plt.title(title, fontweight='bold')

    # Generate predicted values for future years
    pred_years = np.arange(1960, 2060)
    pred_values = curve_fun(pred_years, *params)

    # Plot the actual data and predicted values
    plt.plot(dframe["year"], dframe[country], label="data")
    plt.plot(pred_years, pred_values, label="predicted values")
    plt.legend()
    plt.xlim(1960, 2060)
    plt.xlabel("Year", fontweight='bold', fontsize=12)
    plt.ylabel(indicator, fontweight='bold', fontsize=12)
    plt.savefig("GDP_prediction.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    
    df_gdp, df_gdp_trans = read_file("gdp_.csv")
    df_population, df_population_trans = read_file("population_.csv")
    
    clustering_dataset()
    
    #plotting the plots and prediction plots for gdp dataframe
    plot_fit(df_gdp_trans , "Ireland", "GDP", "GDP of Ireland")
    plot_prediction(df_gdp_trans , "Ireland", "GDP", "GDP Prediction for 2060-Ireland")
    
    #plotting the plots and prediction plots for population dataframe
    plot_fit(df_population_trans, "Ireland", "population", "Population Growth of Ireland")
    plot_prediction(df_population_trans , "Ireland", "population", "Population Growth prediction for 2060-Ireland")
