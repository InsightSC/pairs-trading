import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts

def get_stationarity(df):
    """
    Performs ADF test of stationarity.
    INPUTS: dataframe spread of pair of stocks
    """
    adf_results = {}

    for column in df:
        if column != 'Date':
            adf_results[column] = ts.adfuller(df[column])[1]
    
    return adf_results

def get_pnl(exposure, spread, positions):
    """
    Returns matrix of cumulative sum of pnl% of the executed trades for each pair of stocks
    INPUTS: total exposures; spread of the stocks; the [1,0,-1] position matrix
    """
    returns = spread.drop('Date', axis=1) - spread.drop('Date', axis=1).shift(1)
    strategy_returns = (positions.drop('Date', axis=1).shift(1)*returns/exposure.drop('Date', axis=1)).cumsum()
    strategy_returns.insert(0, 'Date', spread['Date'])

    return strategy_returns.dropna()

def get_entry_exit_points(df, threshold):
    """
    Returns [1,0,-1] positions matrix based on entry/exit signals
    Trading signal: entry when z score crosses threshold and exit the position when it reaches the mean (i.e. 0)
    INPUTS: z score dataframe; threshold value (likely between 1 and 2)
    """
    long_positions = pd.DataFrame(df['Date'])
    short_positions = pd.DataFrame(df['Date'])
    positions = pd.DataFrame(df['Date'])

    for column in df:
        if column != 'Date':
            long_positions[column] = np.where(df[column] < -threshold, 1, 0) # entry when long
            long_positions[column] = np.where(df[column] >= 0, 0, long_positions[column]) # exit when long

            short_positions[column] = np.where(df[column] > threshold, -1, 0) # entry when short
            short_positions[column] = np.where(df[column] <= 0, 0, short_positions[column]) # exit when short

            positions[column] = long_positions[column] + short_positions[column]

    return positions

def get_z_score(df_spread):
    """
    Returns mean reversion values, defined as std dev amount of dispersion of the spread from the mean.
    INPUTS: dataframe containing spreads of pairs of stocks.
    """
    df_z_score = pd.DataFrame(df_spread['Date'])
    
    for column in df_spread:
        if column != 'Date':
            df_z_score[column] = (df_spread[column] - df_spread[column].rolling(window=30).mean())/(df_spread[column].rolling(window=30).std())
    
    return df_z_score.dropna()

def get_total_exposure(df, stocks):
    """
    Returns dataframe containing sum of pairs of stocks -> to be used in pnl% calculation.
    INPUTS: merged dataframe, name of stocks in form of list
    """
    df_sum = pd.DataFrame(df['Date']) 

    for stock1 in stocks:
        for stock2 in stocks:
            if stock1 != stock2 and stock2 + ' - ' + stock1 not in df_sum.columns:
                df_sum[stock1 + ' - ' + stock2] = df['Stock ' + stock1] + df['Stock ' + stock2]

    return df_sum

def get_df_spread(df, stocks):
    """
    Returns dataframe containing spread of pairs of stocks, to be used in calculation of mean reversion values & pnl.
    INPUTS: merged dataframe; name of stocks in form of list
    """
    df_spread = pd.DataFrame(df['Date']) 

    for stock1 in stocks:
        for stock2 in stocks:
            if stock1 != stock2 and stock2 + ' - ' + stock1 not in df_spread.columns:
                df_spread[stock1 + ' - ' + stock2] = df['Stock ' + stock1] - df['Stock ' + stock2]

    return df_spread

def get_df_merged(data_frames):
    """
    Returns merged dataframe of the 8 stocks, drops NaN values for stock H by filling forward.
    INPUTS: the read csv dataframes in form of list
    """
    df_merged = reduce(lambda left,right: pd.merge(left,right,on=['Date'], how='left'), data_frames)
    df_merged = df_merged.dropna(how='all').fillna(method='ffill')

    return df_merged

def main():
    df1 = pd.read_csv('Stock A.csv')
    df2 = pd.read_csv('Stock B.csv')
    df3 = pd.read_csv('Stock C.csv')
    df4 = pd.read_csv('Stock D.csv')
    df5 = pd.read_csv('Stock E.csv')
    df6 = pd.read_csv('Stock F.csv')
    df7 = pd.read_csv('Stock G.csv')
    df8 = pd.read_csv('Stock H.csv')
    data_frames = [df1, df2, df3, df4, df5, df6, df7, df8]
    stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    """
    PART 1: MEAN REVERSION ANALYSIS
    - 1. Merge the 8 read csv dataframes into one df_merged
    - 2. Get the spread between stock prices, defined as long/short equal number of each security
    - 3. Get the z score of this spread which is itself a measure of dispersion from the mean
    - 4. Output a matrix displaying the mean reversion values for each pair of stocks (saved as mean_reversion_values.csv)
    - *5. Analyse z score to find pairs that are most mean reverting
    - *6. Perform stationarity test to confirm pairs that are most mean reverting
    * Note: decided to use z-score as measure of mean reversion but found it difficult to rank and find top 3 pairs from this (gave up especially after stumbling onto ADF test)
            so, what I did is to output the mean reversion values, plot the graphs and identify those most mean reverting (saved as mean_reversion_values (with charts).xlsx)
            then I stumbled onto ADF test and concept of stationarity: saw that ADF test over spread values coincides with this qualitative observation
                i.e. those with lowest p-value for stationary under ADF test are those that 'look' most mean reverting
            From this I started to write the ADF code from scratch (as should not be using external libraries) but thought would not be able to finish on time...
            So ended up having this final step as way to collapse my qualitative picks to top 3 (not ideal of course...)
    """
    df_merged = get_df_merged(data_frames)
    df_spread = get_df_spread(df_merged, stocks)
    df_z_score = get_z_score(df_spread)
    df_z_score.to_csv('mean_reversion_values.csv') #Outputs a matrix displaying the mean reversion values for each pair of stocks

    print("Analysis of z-values shows best pairs are those pairing with stock H: (A - H), (B - H), ... , (F-H), (G-H)")

    adf_results = get_stationarity(df_spread)
    best_three_pairs_adf = sorted(adf_results, key = adf_results.get, reverse = False)[:3]
    print("ADF test collapses this to best three pairs: " + str(best_three_pairs_adf)) # ADF test suggests best pairs are (G,H), (B,H) and (C,H)

    """
    PART 2: TRADING STRATEGY DESIGN
    - Enter the pair trade when the spread is 2sigma above its mean and close when it reverts back to its mean
    - Position sizing: remain market neutral, each pair as single portfolio - further enhancement would have wgt1, wgt2 in method get_df_spread (spread = wgt1 * p1 - wgt2 * p2)
                       and weights dependent on how stationary / mean-reverting time series is
    """
    threshold = 2 #choose +-2sigma as trading signal for entry
    df_positions = get_entry_exit_points(df_z_score, threshold) #[1,0,-1] position matrix based on this rule

    """
    PART 3: IMPLEMENT TRADING STRATEGY
    - We implement the trading strategy across all 8 stocks and confirm that pnl% is one of largest for the more mean reverting stocks, i.e. those identified in PART 1
    - Trade execution: executed trades can be seen in positions matrix outputed as positions.csv
    """
    df_positions.to_csv('positions.csv') #outputs the [1,0,-1] position matrix
    
    df_exposure = get_total_exposure(df_merged, stocks)
    df_pnl = get_pnl(df_exposure, df_spread, df_positions)
 
    """
    PART 4: EVALUATION AND PRESENTATION
    - Three plots of cummulative P&L%
    - Plots: 1) Mean reversion analysis - best pairs being (, H):
                    -> Shows strategy performs strongly for the pairs identified in part 1, namely those that pair with stock H:
                                        Pair    P&L%
                                        G-H	    +37%
                                        C-H	    +36%
                                        E-H	    +33%
                                        A-H	    +33%
                                        B-H	    +29%
                                        D-H	    +28%
                                        F-H	    +21%
                        vs. for other pairs avg P&L% of +2% (min -8%, max +8%)
             2) Best three pairs ADF test (with all pairs):
                    -> Shows in bold the top three pairs ('G - H', 'B - H', 'C - H') chosen with ADF test
             3) Best five pairs - ADF test
                    -> Shows that only the top 3 in ADF test have significant p-values to demonstrate mean reversion
    """
    ax = df_pnl.plot(x = 'Date', y = 'A - B', label = '_nolegend_', title = 'Mean reversion analysis - best pairs being (, H)')
    for column in df_pnl.columns:
        if column != 'Date':
            if column in ['A - H', 'B - H', 'C - H', 'D - H', 'E - H', 'F - H', 'G - H']:
                df_pnl[column].plot(ax=ax, label = column, linewidth=3)
            else:
                df_pnl[column].plot(ax=ax, label = '_nolegend_')

    ax1 = df_pnl.plot(x = 'Date', y = 'A - B', label = '_nolegend_', title = 'Best three pairs post ADF test')
    for column in df_pnl.columns:
        if column != 'Date':
            if column in ['G - H', 'B - H', 'C - H']:
                df_pnl[column].plot(ax=ax1, label = column, linewidth=3)
            else:
                df_pnl[column].plot(ax=ax1, label = '_nolegend_')

    best_eight_pairs_adf = sorted(adf_results, key = adf_results.get, reverse = False)[:4]
    ax2 = df_pnl.plot(x = 'Date', y = 'A - B', label = '_nolegend_', title = 'Best five pairs - ADF test')
    for column in best_eight_pairs_adf:
        if column != 'Date':
            if column in ['A - H', 'B - H', 'C - H', 'D - H', 'E - H', 'F - H', 'G - H']:
                df_pnl[column].plot(ax=ax2, label = column, linewidth=3)
            else:
                df_pnl[column].plot(ax=ax2, label = '_nolegend_')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
