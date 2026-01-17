import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

#Load the Cleaned_Data csv file
data  = pd.read_csv('data/Cleaned_Data.csv')

data = pd.DataFrame(data)

#Data Overview
print(data.head(10))

#Generate heatmap to check correlation between variables

#Calculate the Correlation Matrix using
correlation_matrix = data.corr()
print('Correlation Matrix\n', correlation_matrix)

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

#Create a figure with 2 Subplots 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#Plot the Average Annual Temperature
sns.lineplot(x = 'Year', y = 'Annual_Average', data=data,  marker = 'o', color = 'b', label = 'Line', ax=axes[0])
# Adding labels and title
axes[0].set_title('Annual Anomaly Temperature 1880 - 2024')
axes[0].set_xlabel('Years')
axes[0].set_ylabel('Annual Temperature')
axes[0].grid(True)

#Plot the 5 Year Average Temperature
sns.lineplot(x = 'Year', y = '5Year_Avg', data=data, marker = 'o', color = 'r', label = 'Line', ax=axes[1])
# Adding labels and title
axes[1].set_title('5 Year Average 1880 - 2024')
axes[1].set_xlabel('Years')
axes[1].set_ylabel('5 Year Average')
axes[1].grid(True)

plt.tight_layout()
plt.show()

#Plot Seasons

#First Convert current dataset into Wide Format for usability
data_wide = pd.melt(
data, value_vars=['SUMMER', 'AUTUMN', 'WINTER', 'SPRING'], var_name='Season', value_name='Values'
)

print(data_wide.head(10))

#Vizualise
plt.figure(figsize=(15, 10))
sns.boxplot(x='Season', y= 'Values', data=data_wide, hue='Season')
plt.title('Size Distribution by Quality')
plt.xlabel('Seasons')  # x-axis title
plt.ylabel('Global Anomaly Temperature')
plt.show()

#######################

#Create New dataset in long format to add Month as a categorical variable and use the Months Jan - Dec
#as its values. Whilst retaining the integrity of the other variables. Will use for Tableau

# Melt the dataframe to long format
data_long = pd.melt(data, id_vars=['Year', 'Annual_Average', '5Year_Avg', 'SUMMER', 'AUTUMN', 'WINTER', 'SPRING', 'D-N'],
                  value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                  var_name='Month', value_name='MonthlyTemp')

# Optionally, reorder the columns for better readability
data_long = data_long[['Year', 'Month', 'MonthlyTemp', 'SUMMER', 'AUTUMN', 'WINTER', 'SPRING', 'D-N', 'Annual_Average', '5Year_Avg']]

# Write the Wide Format DataFrame to a CSV file
data_long.to_csv('data/Data_Long.csv', index=False)

#Features and target variable
X = data_long[['MonthlyTemp', 'SUMMER', 'AUTUMN', 'WINTER', 'SPRING']]
y = data_long['Annual_Average']

#Feature Importance - Use RandomForestRegressor() to filter and determine strong predictors for model training.
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importances = model.feature_importances_

for col, imp in zip(X.columns, importances):
    print("\n", col, imp)

#Drop monthly temperature


