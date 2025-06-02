import pandas as pd
from scipy.stats import entropy

pd.set_option('display.max_columns',15)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from tabulate import tabulate
from datetime import datetime
import re

def information_gain(x, y):
    igResult = []
    columnName = x.columns
    count = 0
    for element in columnName:
        columnValue = x[columnName[count]]
        entropy_before = entropy(columnValue.value_counts(normalize=True))
        y.name = 'split'
        columnValue.name = 'members'
        grouped_distrib = columnValue.groupby(y).value_counts(normalize=True).reset_index(name='count').pivot_table(
            index='split', columns='members', values='count').fillna(0)
        entropy_after = entropy(grouped_distrib, axis=1)
        entropy_after *= y.value_counts(sort=False, normalize=True)
        ig = entropy_before - entropy_after.sum()
        igResult.append(ig)
        count += 1
    InformartionGain = pd.Series(igResult)
    InformartionGain.index = columnName
    return InformartionGain.sort_values(ascending=False)

def convert_date(date_str):
    # Remove ordinal suffixes ('st', 'nd', 'rd', 'th') from the day
    date_str = date_str.strip()

    # Remove ordinal suffixes ('st', 'nd', 'rd', 'th') from the day
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    # Convert to datetime object
    return datetime.strptime(date_str, "%B %d, %Y")

def clean_year(value):
    # Check if the value contains a year range (e.g., "2015/2016" or "2009-2010")
    if isinstance(value, str) and ('/' in value or '-' in value):
        # Check if the first part is a valid 4-digit year
        first_part = value.split('/')[0].split('-')[0]
        if first_part.isdigit() and len(first_part) == 4:
            return int(first_part)
        else:
            # If not a valid year (e.g., "08/09"), return 0
            return 0
    # Check if the value is a valid 4-digit year
    elif isinstance(value, str) and value.isdigit() and len(value) == 4:
        return int(value)
    else:
        # If the value is not a valid year, return 0
        return 0

def convert_to_kg(weight_str):
    if isinstance(weight_str, str):
        weight_str = weight_str.lower().strip()
        if "kg" in weight_str:
            return float(weight_str.replace("kg", "").strip())
        elif "lbs" in weight_str:
            lbs_value = float(weight_str.replace("lbs", "").strip())
            return lbs_value * 0.453592  # Convert pounds to kg
    try:
        # If there is no unit, just return the numeric value without conversion
        return float(weight_str)
    except ValueError:
        return None  # If it's not a valid number, return None


def adjust_range_values(altitude):
    altitude = str(altitude).strip().lower()

    # Remove unwanted units and characters, including words and patterns like '..n..', 'dl', 'fee', etc.
    unwanted_chars = ['m', 'msnm', 'sn', 'eters', 'etros', 'metros', '~', '.s.l', 's', 'de', 'p', '公尺', 't', 'al',
                      'f.', 'fee', 'hru', 'dl', '..n..', '.']
    for char in unwanted_chars:
        altitude = altitude.replace(char, '')

    altitude = ' '.join(altitude.split())  # Normalize spaces

    # Handle ranges with 'a' as a separator (e.g., "1200 a 1400")
    if ' a ' in altitude:
        altitude = altitude.replace(' a ', '-')

    # Handle ranges with space or dash separators
    if ' ' in altitude or '-' in altitude:
        # Replace space with dash for consistent handling
        altitude = altitude.replace(' ', '-')
        parts = [float(x) for x in altitude.split('-') if
                 x.strip().replace('.', '').isdigit()]  # Skip non-numeric parts
        if len(parts) == 2:
            return round(sum(parts) / 2)  # Return the average of the range, rounded

    # If it's a single value, return it, rounding if necessary
    if altitude.replace('.', '').isdigit():  # Check if the remaining string is numeric
        return round(float(altitude))

    return None  # Return None if the value can't be converted

def decode_column(encoded_column, column_name, encoders):
    return encoders[column_name].inverse_transform(encoded_column)


#--import Data--
file_path = 'DataSet/arabica_data_cleaned.csv'
df = pd.read_csv(file_path)

#--data cleaning--
df = df[df['Variety'].isin(['Bourbon', 'Caturra', 'Typica'])]

# Fill missing values with appropriate values
# For numerical columns, you can use mean/median

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
    #print(col)

# For categorical columns, you can use the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
    #print(df[col].mode()[0])

#drop unnecessary column
df.drop(columns=['ID','Species'], inplace=True)

#Check for duplicates and reset index
df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

#--data transformation--
#convert measurement ft to m
count = 0
temp_unit_measeurement = df['unit_of_measurement'].tolist()
temp_altitude_low_meters = df['altitude_low_meters'].tolist()
temp_altitude_high_meters = df['altitude_high_meters'].tolist()
temp_altitude_mean_meters = df['altitude_mean_meters'].tolist()

for row in df['unit_of_measurement']:
    if row == 'ft':
        temp_altitude_low_meters[count] = temp_altitude_low_meters[count]*0.3048
        temp_altitude_high_meters[count] = temp_altitude_high_meters[count]*0.3048
        temp_altitude_mean_meters[count] = temp_altitude_mean_meters[count]*0.3048
    count = count+1

df['altitude_low_meters'] = pd.DataFrame(temp_altitude_low_meters)
df['altitude_high_meters'] = pd.DataFrame(temp_altitude_high_meters)
df['altitude_mean_meters'] = pd.DataFrame(temp_altitude_mean_meters)

df.drop(columns=['unit_of_measurement'], inplace=True) #unit_of_measurement unnecessary now


#split numeric and categorical
numeric_column = ['Altitude','Number.of.Bags', 'Bag.Weight', 'Aroma','Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance',
                  'Uniformity','Clean.Cup', 'Sweetness','Cupper.Points', 'Total.Cup.Points','Moisture', 'Category.One.Defects',
                  'Quakers','Category.Two.Defects','altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters']
date_column = ['Harvest.Year', 'Grading.Date', 'Expiration']

numeric_featrues = df[numeric_column]
date_column = df[date_column]
categorical_featrues = df.drop(columns=numeric_column)
categorical_featrues = categorical_featrues.drop(columns=date_column)

#Transform Categorical from word to number
#print(tabulate(df, headers='keys'))
#print(tabulate(df, headers='keys'))
encoders = {}
for column in categorical_featrues:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])

#print(tabulate(df, headers='keys'))


#for decode
df_temporary = df.copy()
for column in categorical_featrues:
    df_temporary[column] = decode_column(df_temporary[column], column, encoders)
print(tabulate(df_temporary, headers='keys'))



#handle date type
df_temp = date_column['Grading.Date']
attribute_list = df_temp.tolist()
replace = []
count = 0
for loop in range(len(attribute_list)):
    date_text = str(attribute_list[count])
    date_obj = convert_date(date_text)
    replace.insert(loop, date_obj)
    count = count + 1

#print(df.info())
df['Grading.Date'] = pd.DataFrame(replace)
df['Grading.Date_year'] = df['Grading.Date'].dt.year
df['Grading.Date_month'] = df['Grading.Date'].dt.month
df['Grading.Date_day'] = df['Grading.Date'].dt.day


df_temp = df['Expiration']
attribute_list = df_temp.tolist()
replace = []
count = 0
for loop in range(len(attribute_list)):
    date_text = str(attribute_list[count])
    date_obj = convert_date(date_text)
    replace.insert(loop, date_obj)
    count = count + 1

df['Expiration'] = pd.DataFrame(replace)
df['Expiration_year'] = df['Expiration'].dt.year
df['Expiration_month'] = df['Expiration'].dt.month
df['Expiration_day'] = df['Expiration'].dt.day

df = df.drop(columns=['Grading.Date', 'Expiration'])

#handle harvest year
df['Harvest.Year'] = df['Harvest.Year'].apply(clean_year)

#handle bag weiht
df['Bag.Weight'] = df['Bag.Weight'].apply(convert_to_kg)

#handle allttitude
df['Altitude'] = df['Altitude'].apply(adjust_range_values)

scaler = MinMaxScaler()
df[numeric_column] = scaler.fit_transform(df[numeric_column])



#print("=============== Data Hasil Pre-Processing ==================")
#print(tabulate(df, headers='keys'))
#print(df.info())
#print("\n")


# Assume 'Variety' is the target variable for classification
X = df.drop(columns=['Variety'])
y = df['Variety']

infgain = information_gain(X,y)
infGainIndex = infgain.index
temp_inf_gain_index = []

rank = 10
for i in range(rank):
    temp_inf_gain_index.append(infGainIndex[i])
infGainIndex = temp_inf_gain_index
#print("===================== Information Gain =========================")
#print(infgain)
#print("\n")

#print(X.info())
X_important = X[infGainIndex]
#print(X_important.info())
X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.015, random_state=42)

ASM_function = ['entropy', 'gini']
nEstimator = 1000



model_RDF = RandomForestClassifier(criterion=ASM_function[0], n_estimators=nEstimator)
model_RDF.fit(X_train, y_train)
cv_result = cross_val_score(model_RDF, X_train, y_train, cv=10, scoring='accuracy')
print("Akurasi Random Forest : ", cv_result.mean())

print("Hasil Pengujian Data Tunggal :")
predictions = model_RDF.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("\n")











