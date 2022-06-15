from datetime import datetime
import numpy as np
import os


def days_per_month(month_number, is_leap_year):
    # February
    if month_number == 2:
        if is_leap_year:
            return 29
        else:
            return 28
    # based on the Knuckle mnemonic
    elif month_number <= 7:
        if month_number % 2 != 0:
            return 31
        else:
            return 30
    else:
        if month_number % 2 == 0:
            return 31
        else:
            return 30



def extract_features_date(date):
    # extract the features of the date
    years = date.astype(object).year
    months = date.astype(object).month
    days = date.astype(object).day
    hours = date.astype(object).hour
    minutes= date.astype(object).minute
    seconds = date.astype(object).second
    day_of_week = date.astype(datetime).isoweekday()
    millennium = round(np.floor(years / 1000))
    century = round(np.floor(years / 100))
    decade = round(np.floor(years / 10))
    last_digit_year = years % 10
    is_leap_year = years % 4 == 0
    # 0 because Sunday is 0
    is_business_day = day_of_week <= 5 
    quarter = round(np.ceil(months / 3))
    days_in_month = days_per_month(months, is_leap_year)
    day_of_year = date.astype(object).timetuple().tm_yday
    is_month_start = days == 1
    is_quarter_start = days == 1 and months % 3 == 0
    is_year_start = days == 1 and months == 1
    is_year_end = days == 31 and months == 12
    # end of the year is the end of the last quarter, 31 March, 30 June, 30 September
    is_quarter_end = is_year_end or (months == 3 and days == 31) or ((months == 6 or 9) and days == 30)
    is_month_end = days == days_in_month
    
    encoding = [years, months, days, hours, minutes, seconds, day_of_week, millennium, century, decade, last_digit_year, is_leap_year, is_business_day, quarter, days_in_month, day_of_year, is_month_start,
            is_quarter_start, is_year_start, is_year_end, is_quarter_end, is_month_end]
    
    return encoding


# load data created using data_creation.py
data_location = os.path.join(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'), 'data_thesis')
data = np.load(data_location + "/np_dates.npy")

# vectorize the date
dates = np.array(list(map(extract_features_date, data)))

#Create Data folder if not exist
data_folder_check  = os.path.isdir('Data')
if not data_folder_check:
    os.makedirs('Data')
    print('Created Data folder as it was not present')

#Save the data to a file
np.save('Data/vectorized_dates.npy', dates)