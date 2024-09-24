import pandas as pd
import numpy as np
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from sklearn.preprocessing import MaxAbsScaler

def data_generator(start_date, periods, channels, spend_scalar, adstock_alphas, saturation_lamdas, betas, freq="W"):
    '''
    Generates a synthetic dataset for a MMM with trend, seasonality, and channel-specific contributions.

    Args:
        start_date (str or pd.Timestamp): The start date for the generated time series data.
        periods (int): The number of time periods (e.g., days, weeks) to generate data for.
        channels (list of str): A list of channel names for which the model will generate spend and sales data.
        spend_scalar (list of float): Scalars that adjust the raw spend for each channel to a desired scale.
        adstock_alphas (list of float): The adstock decay factors for each channel, determining how much past spend influences the current period.
        saturation_lamdas (list of float): Lambda values for the logistic saturation function, controlling the saturation effect on each channel.
        betas (list of float): The coefficients for each channel, representing the contribution of each channel's impact on sales.

    Returns:
        pd.DataFrame: A DataFrame containing the generated time series data, including demand, sales, and channel-specific metrics.
    '''
    
    # 0. Create time dimension
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    df = pd.DataFrame({'date': date_range})
    
    # 1. Add trend component with some growth
    df["trend"]= (np.linspace(start=0.0, stop=20, num=periods) + 5) ** (1 / 8) - 1
    
    # 2. Add seasonal component with oscillation around 0
    df["seasonality"] = df["seasonality"] = 0.1 * np.sin(2 * np.pi * df.index / 52)
    
    # 3. Multiply trend and seasonality to create overall demand with noise
    df["demand"] = df["trend"] * (1 + df["seasonality"]) + np.random.normal(loc=0, scale=0.10, size=periods)
    df["demand"] = df["demand"] * 1000
    
    # 4. Create proxy for demand, which is able to follow demand but has some noise added
    df["demand_proxy"] = np.abs(df["demand"]* np.random.normal(loc=1, scale=0.10, size=periods))
    
    # 5. Initialize sales based on demand
    df["sales"] = df["demand"]
    
    # 6. Loop through each channel and add channel-specific contribution
    for i, channel in enumerate(channels):
        
        # Create raw channel spend, following demand with some random noise added
        df[f"{channel}_spend_raw"] = df["demand"] * spend_scalar[i]
        df[f"{channel}_spend_raw"] = np.abs(df[f"{channel}_spend_raw"] * np.random.normal(loc=1, scale=0.30, size=periods))
               
        # Scale channel spend
        channel_transformer = MaxAbsScaler().fit(df[f"{channel}_spend_raw"].values.reshape(-1, 1))
        df[f"{channel}_spend"] = channel_transformer .transform(df[f"{channel}_spend_raw"].values.reshape(-1, 1))
        
        # Apply adstock transformation
        df[f"{channel}_adstock"] = geometric_adstock(
            x=df[f"{channel}_spend"].to_numpy(),
            alpha=adstock_alphas[i],
            l_max=8, normalize=True
        ).eval().flatten()
        
        # Apply saturation transformation
        df[f"{channel}_saturated"] = logistic_saturation(
            x=df[f"{channel}_adstock"].to_numpy(),
            lam=saturation_lamdas[i]
        ).eval()
        
        # Calculate contribution to sales
        df[f"{channel}_sales"] = df[f"{channel}_saturated"] * betas[i]
        
        # Add the channel-specific contribution to sales
        df["sales"] += df[f"{channel}_sales"]
    
    return df