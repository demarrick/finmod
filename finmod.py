import discord
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import asyncio
import os
token = os.getenv('token')

intents = discord.Intents.default()
intents.members = True 
intents.messages=True
intents.message_content=True
intents.voice_states=True
client= discord.Client(intents=intents)
general_chat=1179370783726899272
# Dictionary to store message timestamps
message_timestamps = {}
today=date.today()
tomorrow = today + timedelta(days=1)
start_date="2023-11-17"
initial_wealth= 106092663.0
portfolio_list=["^SP500TR", "GLD", "MSTR", "OMC", "VZ", "APTV", "BBY", "KMB", "CPB", "PXD", "APA", "PNC", "SYF", "HUM", "ABBV", "DE", "CMI", "TXN","ON"]
weights = [0.45, 0.05, 0.10, 0.025, 0.025, 0.025, 0.025,0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
deficit = 1 - sum(weights)
weights[-1] += deficit
deadline = date.fromisoformat("2023-12-20")
if today < deadline:
    dataframes={}
    for element in portfolio_list :
        dataframes["panda_"+element]=yf.download(element, start=start_date, end=tomorrow)
    for element in dataframes:
        interim=dataframes[element]
        print(len(interim['Adj Close']))
else:
    print("Round 2 deadline reached.")
return_dataframes={}
for element in dataframes:
    current_df = dataframes[element]
    returns = current_df['Adj Close'].pct_change()
    current_df['Return']=returns
    return_dataframes[element] = current_df
compound_return_list=[]
for element in return_dataframes:
    compound_return = (return_dataframes[element]['Return']+1).cumprod().iloc[-1] -1
    compound_return_list=compound_return_list+[compound_return]
wportfolio=pd.DataFrame()
wportfolio['Asset']=portfolio_list
wportfolio['Weights']=weights
positions=[]
for element in weights:
    positions=positions+[initial_wealth*element]
wportfolio['Position']=positions
wportfolio['Position']=wportfolio['Position'].round()
wportfolio['Compound Return']= compound_return_list
total_cret = sum(wportfolio['Compound Return'] * wportfolio['Weights'])
wportfolio['Position t=2']=wportfolio['Position']+(wportfolio['Position']*wportfolio['Compound Return'])
wportfolio['Position t=2'] = wportfolio['Position t=2'].round()
wportfolio['Contribution to Performance']=(wportfolio['Compound Return']*wportfolio['Weights'])/total_cret
total_profit=wportfolio['Position t=2'].sum()
daily_risk_free_rate=yf.download("^IRX", start=start_date, end=tomorrow)
combined_returns = pd.DataFrame()
panda_store = []
for i in return_dataframes:
    dummy = pd.DataFrame(return_dataframes[i]['Return'])
    panda_store.append(dummy)
combined_returns = pd.concat(panda_store, axis=1)
combined_returns.columns = portfolio_list
weighted_returns = combined_returns.mul(weights, axis=1)
daily_portfolio_returns = weighted_returns.sum(axis=1)
daily_portfolio_returns.name = 'Portfolio'
final_combined_returns=pd.concat([combined_returns, daily_portfolio_returns], axis=1)
cumulative_returns_assets = (1 + combined_returns).cumprod() - 1
cumulative_portfolio_returns = (1 + daily_portfolio_returns).cumprod() - 1
cumulative_portfolio_returns.name = 'Cumulative Portfolio'
final_combined_cumulative_returns = pd.concat([cumulative_returns_assets, cumulative_portfolio_returns], axis=1)
final_combined_returns_date=final_combined_returns.reset_index()
daily_risk_free_rate_date=daily_risk_free_rate.reset_index()
interim_portfolio_ret=final_combined_returns_date[['Date', '^SP500TR','Portfolio']]
interim_portfolio_ret_adjusted=interim_portfolio_ret[interim_portfolio_ret['Date']!="2023-11-20"]
interim_risk_free_rates=daily_risk_free_rate_date[['Date', 'Adj Close']]
performance_reference_panda=interim_portfolio_ret_adjusted.merge(interim_risk_free_rates, on='Date', how='inner')
performance_reference_panda_renamed=performance_reference_panda.rename(columns={'Adj Close':'Risk Free Rate', 'Portfolio':'Portfolio Return', '^SP500TR':'Benchmark Return'})
performance_reference_panda_renamed['Risk Free Rate']=(performance_reference_panda_renamed['Risk Free Rate']/100)/252
excess_return=[]
bench_excess_return=[]
for index, row in performance_reference_panda_renamed.iterrows():
    daily_excess_return = row['Portfolio Return'] - row['Risk Free Rate']
    daily_excess_return_bench=row['Benchmark Return'] - row['Risk Free Rate']
    excess_return=excess_return+[daily_excess_return]
    bench_excess_return=bench_excess_return+[daily_excess_return_bench]
performance_reference_panda_renamed['Portfolio Excess Rerturn']=excess_return
performance_reference_panda_renamed['Benchmark Excess Rerturn']=bench_excess_return
Expected_return=performance_reference_panda_renamed['Portfolio Excess Rerturn'].mean()
Expected_return_volatility=performance_reference_panda_renamed['Portfolio Excess Rerturn'].std()
Daily_Sharpe_ratio=Expected_return/Expected_return_volatility
Annualized_Sharpe_ratio=Daily_Sharpe_ratio* (252 ** 0.5)
Expected_return_bench=performance_reference_panda_renamed['Benchmark Excess Rerturn'].mean()
Expected_return_volatility_bench=performance_reference_panda_renamed['Benchmark Excess Rerturn'].std()
Daily_Sharpe_ratio_bench=Expected_return_bench/Expected_return_volatility_bench
Annualized_Sharpe_ratio_bench=Daily_Sharpe_ratio_bench* (252 ** 0.5)
benchmark_cret=final_combined_cumulative_returns['^SP500TR'].iloc[-1]
portfolio_cret=final_combined_cumulative_returns['Cumulative Portfolio'].iloc[-1]
wealth_at_start= 100000000
round1_portfolio_list=["^SP500TR", "GLD", "MSTR", "F", "AAL", "AES", "OGN", "CFG", "SYF", "GM", "MOS", "CMA", 
           "UAL", "PSX", "VLO", "ALB"]
round1_weights = [0.450000178001926, 0.10000004702132, 0.050000004634652, 0.0929776394718518, 0.0545008236783605, 0.0513232224692125, 0.03991401760977,
           0.0282252122318226, 0.0275176124153853, 0.0274124125084215, 0.0234756113067814, 0.0172452084208263, 0.0172288078648165, 0.00992439939381731, 0.00753959995852304, 0.00271520301251306]
deficit2 = 1 - sum(round1_weights)
round1_weights[-1] += deficit2
buyin_start="2023-10-20"
dataframes2={}
for element in round1_portfolio_list :
    dataframes2["panda_"+element]=yf.download(element, start=buyin_start, end="2023-11-18")
for element in dataframes2:
    interim=dataframes2[element]
return_dataframes2={}
for element in dataframes2:
    current_df2 = dataframes2[element]
    returns2 = current_df2['Adj Close'].pct_change()
    current_df2['Return']=returns2
    return_dataframes2[element] = current_df2
compound_return_list2=[]
for element in return_dataframes2:
    compound_return2 = (return_dataframes2[element]['Return']+1).cumprod().iloc[-1] -1
    compound_return_list2=compound_return_list2+[compound_return2]
round1_wportfolio=pd.DataFrame()
round1_wportfolio['Asset']=round1_portfolio_list
round1_wportfolio['Weights']=round1_weights
round1_positions=[]
for element in round1_weights:
    round1_positions=round1_positions+[wealth_at_start*element]
round1_wportfolio['Position']=round1_positions
round1_wportfolio['Position']=round1_wportfolio['Position'].round()
round1_wportfolio['Compound Return']= compound_return_list2
round1_total_cret = sum(round1_wportfolio['Compound Return'] * round1_wportfolio['Weights'])
round1_wportfolio['Position t=2']=round1_wportfolio['Position']+(round1_wportfolio['Position']*round1_wportfolio['Compound Return'])
round1_wportfolio['Position t=2'] = round1_wportfolio['Position t=2'].round()
round1_wportfolio['Contribution to Performance']=(round1_wportfolio['Compound Return']*round1_wportfolio['Weights'])/round1_total_cret
panda_store2 = []
for i in return_dataframes2:
    dummy = pd.DataFrame(return_dataframes2[i]['Return'])
    panda_store2.append(dummy)
round1_combined_returns = pd.concat(panda_store2, axis=1)
round1_combined_returns.columns = round1_portfolio_list
round1_weighted_returns = round1_combined_returns.mul(round1_weights, axis=1)
round1_daily_portfolio_returns = round1_weighted_returns.sum(axis=1)
round1_daily_portfolio_returns.name = 'Portfolio'
round1_final_combined_returns=pd.concat([round1_combined_returns, round1_daily_portfolio_returns], axis=1)
round1_cumulative_returns_assets = (1 + round1_combined_returns).cumprod() - 1
round1_cumulative_portfolio_returns = (1 + round1_daily_portfolio_returns).cumprod() - 1
round1_cumulative_portfolio_returns.name = 'Cumulative Portfolio'
round1_final_combined_cumulative_returns = pd.concat([round1_cumulative_returns_assets, round1_cumulative_portfolio_returns], axis=1)
round1_final_combined_cumulative_returns_adjusted=round1_final_combined_cumulative_returns.reset_index()
round1_final_combined_cumulative_returns_adjusted
interim_panda=round1_final_combined_cumulative_returns_adjusted[['Date','Cumulative Portfolio']].copy()
interim_panda['Benchmark Returns'] = interim_panda['Date'].map(round1_combined_returns['^SP500TR'])
interim_panda_renamed=interim_panda.rename(columns={'Cumulative Portfolio':'Cumulative Portfolio Returns'})
interim_panda_renamed['Portfolio Returns'] = round1_daily_portfolio_returns.values
round1_cumulative_benchmark_returns = (1 + round1_combined_returns['^SP500TR']).cumprod() - 1
interim_panda_renamed['Cumulative Benchmark Returns'] = interim_panda['Date'].map(round1_cumulative_benchmark_returns)
interim_panda_renamed['Benchmark Returns'] = interim_panda_renamed['Date'].map(round1_combined_returns['^SP500TR'])
column_order = [col for col in interim_panda_renamed.columns if col != 'Cumulative Portfolio Returns']
column_order.append('Cumulative Portfolio Returns')
interim_panda_renamed = interim_panda_renamed[column_order]
interim_panda2=performance_reference_panda_renamed.copy()
interim_panda2_adjusted=interim_panda2[['Date','Benchmark Return','Portfolio Return']].copy()
interim_panda2_adjusted['Cumulative Benchmark Returns']=(1+interim_panda2_adjusted['Benchmark Return']).cumprod()-1 
interim_panda2_adjusted['Cumulative Portfolio Returns']=(1+interim_panda2_adjusted['Portfolio Return']).cumprod()-1 
interim_panda2_adjusted_renamed=interim_panda2_adjusted.rename(columns={'Benchmark Return':'Benchmark Returns'})
interim_panda2_adjusted_renamed_final=interim_panda2_adjusted_renamed.rename(columns={'Portfolio Return':'Portfolio Returns'})
concatenated_df = pd.concat([interim_panda_renamed, interim_panda2_adjusted_renamed_final], ignore_index=True)
# Consider the lines of code below to get the combined performance from the strat of the challenge untill the end of round 2
concatenated_df_adjusted=concatenated_df.drop(['Cumulative Benchmark Returns','Cumulative Portfolio Returns'], axis=1)
concatenated_df_adjusted['Cumulative Benchmark Returns']=(1+concatenated_df_adjusted['Benchmark Returns']).cumprod()-1
concatenated_df_adjusted['Cumulative Portfolio Returns']=(1+concatenated_df_adjusted['Portfolio Returns']).cumprod()-1
overall_bench=concatenated_df_adjusted['Cumulative Benchmark Returns'].iloc[-1]
overall_port=concatenated_df_adjusted['Cumulative Portfolio Returns'].iloc[-1]
profit=total_profit-wealth_at_start
profit2=total_profit-initial_wealth

def printOverall():
    '''prints the benchmark vs Portfolio c.ret over the full competition '''
    plt.figure(figsize=(15,5))
    sns.set_style("whitegrid")
    sns.lineplot(data=concatenated_df_adjusted, x='Date', y='Cumulative Benchmark Returns', label='Benchmark')
    sns.lineplot(data=concatenated_df_adjusted, x='Date', y='Cumulative Portfolio Returns', label='Portfolio')
    plt.legend()
    plt.ylabel('Cumulative Returns')
    plt.title('Overall Portfolio Performance against Benchmark.')
    plt.savefig('plot2.png')

def printbenchvsPort():
    '''prints the benchmark vs Portfolio c.ret over the second round'''
    plt.figure(figsize=(15,5))
    sns.set_style("whitegrid")
    bench=final_combined_cumulative_returns['^SP500TR']
    strat=final_combined_cumulative_returns['Cumulative Portfolio']
    listadi=[bench, strat]
    plt.plot(bench, label='Benchmark')
    plt.plot(strat, label='Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title("Portoflio and Benchmark cumulative Returns over time.")
    plt.legend()
    plt.savefig('plot.png')

def bar():
    '''print a barchart'''
    sns.set_context('paper')
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    wportfolio_sorted=wportfolio.sort_values('Compound Return')
    sns.barplot(ax=axes[0], data=wportfolio_sorted, x='Asset', y='Compound Return')
    axes[0].set_title('Round 2 Portfolio Compound Returns at '+str(today))
    axes[0].tick_params(axis='x')  
    sns.barplot(ax=axes[1], data=wportfolio_sorted, x='Asset', y='Contribution to Performance')
    axes[1].set_title('Contribution of each Asset to Round 2 Portfolio Total Performance at  '+str(today))
    axes[1].tick_params(axis='x')  
    plt.tight_layout()
    plt.savefig('plot3.png')

@client.event
async def on_message(message):
    if message.author != client.user:
        print(f'Author is not the Bot.')
        if '/round2' in message.content:
            printbenchvsPort()
            channel = client.get_channel(general_chat)
            await channel.send(f'From {start_date} until {today}:')
            await channel.send(f'Benchmark Cumulative return:   {benchmark_cret}')
            await channel.send(f'Portfolio Cumulative return:   {portfolio_cret}')
            await channel.send(f'Profit earned: {profit2}')
            await channel.send(file=discord.File('plot.png'))
            plt.clf()
        elif '/overall' in message.content:
            printOverall()
            channel = client.get_channel(general_chat)
            await channel.send(f'From {buyin_start} until {today}:')
            await channel.send(f'Benchmark Cumulative return:   {overall_bench}')
            await channel.send(f'Portfolio Cumulative return:   {overall_port}')
            await channel.send(f'Profit earned: {profit}')
            await channel.send(file=discord.File('plot2.png'))
            plt.clf()
        elif '/contribution' in message.content:
            bar()
            channel = client.get_channel(general_chat)
            await channel.send(f'At {today}:')
            await channel.send(file=discord.File('plot3.png'))
            plt.clf()

        # Store the current timestamp for this message
        message_timestamps[message.id] = message.created_at

        # Set a time limit (in seconds) for messages to disappear
        time_limit_seconds = 24 * 3600  # 24 hours

        # Wait for the specified time and then delete the message
        await asyncio.sleep(time_limit_seconds)

        # Delete the message after the time limit
        await message.delete()

print(token)
client.run(token)
              
