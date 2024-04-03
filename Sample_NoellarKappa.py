# Linear Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats




def patients():
    df = pd.read_csv('ML_HW_Data_Patients.csv')

    # # Check if values have NaN values
    print(df.isnull().any())

    # remove outliers from Y(systolic)
    sns.displot(df['Systolic'])
    plt.show()
    # df['Systolic'] = np.abs(stats.zscore(df['Systolic']))

    newSystolic = df[(df['Systolic'] < 113)].index
    df.drop(newSystolic, inplace=True)
    newSystolic2 = df[(df['Systolic'] > 131)].index
    df.drop(newSystolic2, inplace=True)

    # # Standardize X
    df['Age'] = scale(df['Age'].values)
    df['Height'] = scale(df['Height'].values)
    df['Weight'] = scale(df['Weight'].values)

    # # Remove Outliers (X)
    sns.displot(df['Age'])
    plt.show()
    df['Age'] = np.abs(stats.zscore(df['Age']))

    sns.displot(df['Height'])
    plt.show()
    df['Height'] = np.abs(stats.zscore(df['Height']))

    sns.displot(df['Weight'])
    plt.show()
    df['Weight'] = np.abs(stats.zscore(df['Weight']))

    # Standardize
    df['Age'] = scale(df['Age'].values)
    df['Height'] = scale(df['Height'].values)
    df['Weight'] = scale(df['Weight'].values)

    # create Dummy Variables
    # Gender
    df['Gender'] = pd.get_dummies(df['Gender'], drop_first=True)

    # # Smoker
    df['Smoker'] = pd.get_dummies(df['Smoker'], drop_first=True)

    #
    # # Location
    Location = np.array(pd.get_dummies(df['Location'], drop_first=True))
    df['Location'] = Location[:, 1:]
    #
    # # SelfAssessedHealthStatus
    SelfAssessedHealthStatus = np.array(pd.get_dummies(df['SelfAssessedHealthStatus'], drop_first=True))
    df['SelfAssessedHealthStatus'] = SelfAssessedHealthStatus[:, 1:]

    # Variables
    X = df[['Age', 'Height', 'Weight', 'Gender', 'Smoker', 'Location', 'SelfAssessedHealthStatus']].values
    Y = df['Systolic'].values
    
    # # build linear regression model

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    ypred = regressor.predict(X_test)

    # # regression coefficents
    coeff_df = pd.DataFrame(regressor.coef_,  columns=['Coefficent'])
    print(coeff_df)

    print('Mean Squared Error:', mean_squared_error(y_test, ypred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, ypred)))
    print('R2_score:', r2_score(y_test, ypred))

    # #Outlier records>>>residuals
    #
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': ypred})

    # Plot Actual Vs Predicted
    sns.boxplot(df1['Actual'])
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df1['Actual'], df1['Predicted'] )
    plt.show()

    # #identify one or few useless features




patients()
