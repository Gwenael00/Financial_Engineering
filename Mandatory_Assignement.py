    # -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:22:20 2023

@author: gwena
"""

import os
os.chdir("D:\Master\WS23_24\Financial Engineering\Python")
import numpy as np
from numpy import exp, sqrt, log
import pandas as pd
import matplotlib.pyplot as plt
from BSM_functions import BSM_IV, BSM_vega
from SVJ_functions import funcOptionPrices
from integration_functions import GL_integration
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

np.random.seed(1)

def BS_delta(S, K, delta, r, sigma, T):

    d1 = (log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    res = norm.cdf(d1)
    return res

###################################################################
####################### Problem 1 #################################
###################################################################

Googl_Data = pd.read_excel('GOOGL_options.xlsx')


S = 126.45
r = 0.0515
delta = 0
optionType = 'call'

#Maturity = Googl_Data.iloc[0,1:] #From Excel
Maturity = [14, 31, 93, 183, 365] #From PDF
Strikes = Googl_Data.iloc[2:15,1:] 
Prices = Googl_Data.iloc[16:30,1:]

#1 - For each option, compute the Black-Scholes implied volatility
sigma0 = 0.50  # initial guess
IV = np.zeros((13,5))
vega = np.zeros((13,5))
BSM_delta = np.zeros((13,5))
for row in range(len(Prices)):
    for column in range(len(Prices.columns)):
        tau = Maturity[column] / 365
        K = Strikes.iloc[row,column]
        price = Prices.iloc[row,column]
        IV[row,column] = BSM_IV(S, K, delta, r, sigma0, tau, optionType, price, 0.00001, 1000, "N")
        #For 1.4
        sigma = IV[row,column]
        vega[row,column] = BSM_vega(S, K, delta, r, sigma, tau)
        BSM_delta[row,column] = BS_delta(S, K, delta, r, sigma, tau) 
      
#2 - For each option maturity, plot the implied volatility smile; i.e. implied volatility as a
#function of log(K=Ft), where Ft = Ster(T􀀀t) is the forward price

F_t = np.zeros(len(Maturity))
mmnes = np.zeros((13,5))
for t in range(len(Maturity)):
    F_t = S*exp(r * Maturity[t]/365)  
    K = Strikes.iloc[:,t] 
    mmnes = log(K / F_t)
    plt.plot(mmnes, IV[:,t]) #all in one plot
    #plt.show() #for one plot each

#3 Characterize the implied volatility patterns. Do they match what we discussed in class?    
    
# Answer: It does make sense (Convew with skew) !!! Write more detailed

#4 - For each option maturity, plot also the Black-Scholes delta and vega as a function of
#log(K=Ft)


for t in range(len(Maturity)):
    F_t = S*exp(r * Maturity[t]/365)  
    K = Strikes.iloc[:,t] 
    mmnes = log(K / F_t)
    #plt.plot(mmnes, vega[:,t])
    plt.plot(mmnes, BSM_delta[:,t])
    #plt.show()

###################################################################
####################### Problem 2 #################################
###################################################################

integrationPoints = 10000
upperBound = 30000
[uv, wgtv] = GL_integration(upperBound, integrationPoints)
settings = {"uv": uv, "wgt": wgtv}
### Model and initial parameters
settings["model"] = "SV"
settings["tauv"] = np.array(Maturity)/365
settings["strike"] = Strikes.to_numpy()

def heston_implied_price(params):
    kappa, theta, sigma, rho, v0 = params
    # Define Heston model's parameters
    kappa = max(kappa, 0.001)  
    sigma = max(sigma, 0.001)  
    rho = max(min(rho, 0.999), -0.999)  
    v0 = max(v0, 0.001) 
    
    parameters = {"S": S, "r": r, "delta": delta, "kappa": kappa, "theta": theta, "rho": rho, "sigma": sigma,"v": v0}
    Prices_hat = funcOptionPrices(parameters, settings)
    return Prices_hat, parameters


initial_params=[2, 0.04, 1.0, -0.02, 0.04]

def objective(params):
    A,B = heston_implied_price(params)
    sum_squared_diff = (sum(sum(((Prices.to_numpy()-A)/vega)**2)))
    return sum_squared_diff

result = minimize(objective, initial_params)
print(result.x)

###################################################################
####################### Problem 3 #################################
###################################################################

"""
Use 100,000 paths to generate the conditional distribution of log(ST ). Plot the condi-
tional distribution vs. the normal distribution with same mean and variance. Plot also
the simulated conditional distribution of vT . Report the mean, median, skewness and
kurtosis of the simulated distributions of log(ST ) and vT . Comment on the results.
"""
kappa = 3
theta = 0.1
rho = 0.2
sigma = 0.5
v0 = 0.1
delta_t = 1/252

W_t = np.random.normal(0,sqrt(delta_t),size=(63,100000))
Z_t = np.random.normal(0,sqrt(delta_t),size=(63,100000))
S_t = np.zeros((64,100000))
V_t = np.zeros((64,100000))
Z_t = rho*W_t + sqrt(1-rho**2)*Z_t
S_t[0] = S
V_t[0] = v0

for t in range(63):
    V_t[t+1] = V_t[t] + kappa * (theta-V_t[t]) * delta_t + sigma * sqrt(V_t[t])*Z_t[t]
    S_t[t+1] = S_t[t] +  (r * delta_t + sqrt(V_t[t+1]) * W_t[t]) * S_t[t]

lg_S_T = log(S_t[63])
mean = np.nanmean(lg_S_T)
std_dev = np.nanstd(lg_S_T)

# Create a range of values for the x-axis using numpy
x = np.linspace(min(lg_S_T), max(lg_S_T), 10000)

# Plot th histogram of your list
plt.hist(lg_S_T, bins=1000, density=True, alpha=0.6, color='g', label='Histogram')
# Plot the normal distribution curve with the same mean and standard deviation
plt.plot(x, norm.pdf(x, mean, std_dev), 'b', label='Normal Distribution')
plt.legend()
plt.show()

mean = np.nanmean(V_t[63])
std_dev = np.nanstd(V_t[63])

# Create a range of values for the x-axis using numpy
x = np.linspace(min(V_t[63]), max(V_t[63]), 10000)

# Plot th histogram of your list
plt.hist(V_t[63], bins=1000, density=True, alpha=0.6, color='g', label='Histogram')
# Plot the normal distribution curve with the same mean and standard deviation
plt.plot(x, norm.pdf(x, mean, std_dev), 'b', label='Normal Distribution')
plt.legend()
plt.show()

"""
Price the American option with the LSM algorithm of Longsta and Schwartz (2001)
using 100,000 paths with antithetic variates (i.e., 50,000 paths and 50,000 antithetic
paths),
"""
W_t = np.random.normal(0,sqrt(delta_t),size=(63,50000))
Z_t = np.random.normal(0,sqrt(delta_t),size=(63,50000))
S_t_1 = np.zeros((64,50000))
V_t_1 = np.zeros((64,50000))
S_t_2 = np.zeros((64,50000))
V_t_2 = np.zeros((64,50000))
Z_t = rho*W_t + sqrt(1-rho**2)*Z_t
S_t_1[0] = S
S_t_2[0] = S
V_t_1[0] = v0
V_t_2[0] = v0

for t in range(63):
    V_t_1[t+1] = V_t_1[t] + kappa * (theta-V_t_1[t]) * delta_t + sigma * sqrt(V_t_1[t])*Z_t[t]
    V_t_2[t+1] = V_t_2[t] + kappa * (theta-V_t_2[t]) * delta_t + sigma * sqrt(V_t_2[t])*(-Z_t[t])
    S_t_1[t+1] = S_t_1[t] +  (r * delta_t + sqrt(V_t_1[t+1]) * W_t[t]) * S_t_1[t]
    S_t_2[t+1] = S_t_2[t] +  (r * delta_t + sqrt(V_t_2[t+1]) * (-W_t[t])) * S_t_2[t]
    
S_t = np.concatenate((S_t_1,S_t_2), axis = 1)
V_t = np.concatenate((V_t_1,V_t_2), axis = 1)

df_S = pd.DataFrame(S_t).T
df_V = pd.DataFrame(V_t).T

df=exp(-r*delta_t)
Exercise = np.maximum(130-df_S,0)


def LSM_Algorithm(j):
    Value= np.array(np.zeros((len(df_S),len(df_S.T))))
    Value[:,-1] = np.maximum(130-df_S.iloc[:,-1],0)
    
    for i in reversed(range(1,len(df_S.T)-1)):
        ITM = df_S[130-df_S[i]>0]
        ITM = ITM.reset_index()
        OTM = df_S[130-df_S[i]<=0]
        OTM = OTM.reset_index()
        v = df_V.iloc[ITM['index'],i]
        if j == 0:
            X=np.array([ITM[i]/130,(ITM[i]/130)**2,v, v**2]).T
        elif j == 1:
            X=np.array([ITM[i]/130,(ITM[i]/130)**2,(ITM[i]/130)**3,v, v**2,v**3]).T  
        elif j == 2:
            X=np.array([ITM[i]/130,(ITM[i]/130)**2,(ITM[i]/130)**3,v, v**2,v**3,v*ITM[i].to_numpy()/130]).T
        elif j == 3:
            X=np.array([ITM[i]/130,(ITM[i]/130)**2,(ITM[i]/130)**3,v, v**2,v**3,v*ITM[i].to_numpy()/130,v**2*ITM[i].to_numpy()/130,v*(ITM[i].to_numpy()/130)**2]).T
        else:
            X=np.array([ITM[i]/130,(ITM[i]/130)**2,(ITM[i]/130)**3,(ITM[i]/130)**4,v, v**2,v**3,v**4, v*ITM[i].to_numpy()/130,v**2*ITM[i].to_numpy()/130,v*(ITM[i].to_numpy()/130)**2]).T
        X = np.nan_to_num(X)
        Y=Value[ITM['index'],i+1]*df
        Y = np.nan_to_num(Y)
        LinReg = LinearRegression().fit(X,Y)
        Continuation = LinReg.predict(X)
        Value[ITM['index'],i] = np.where(Continuation>Exercise.iloc[ITM['index'],i],Value[ITM['index'],i+1]*df, Exercise.iloc[ITM['index'],i])
        Value[OTM['index'],i] = Value[OTM['index'],i+1]*df
    
    P0 = np.mean(Value[:,1]*df)
    se = np.std(Value[:,1])/sqrt(len(Value))
    print('The AM Put value is {} with standard error {}. This means that the 95% confidence interval is from {} to {}'.format(P0,se, P0+norm.ppf(0.025)*se, P0-norm.ppf(0.025)*se))

for t in range(5):
    LSM_Algorithm(j=t)
    


me = np.mean(np.maximum(130-df_S.iloc[:,-1],0)*exp(-r*(3/12)))
se = np.std(np.maximum(130-df_S.iloc[:,-1],0)*exp(-r*(3/12)))/sqrt(len(df_S.iloc[:,-1]))

print ('Price of a Euro Put is {} with standard error {}'.format(me, se))
