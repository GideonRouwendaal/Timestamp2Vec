import numpy as np
import os
from tqdm import tqdm

# define start and end date
START_DATE = "0001-01-01"
END_DATE = "9999-12-31"


def sample_hour_min_sec(date):
    # sample a random number of hours and add this to <date>. This can be 23 max (since 24 would be a new day)
    date += np.timedelta64(np.random.randint(0, 24),'h')
    # sample a random number of minutes and add this to <date>. This can be 59 max (since 60 would be a new hour)
    date += np.timedelta64(np.random.randint(0, 60),'m')
    # sample a random number of seconds and add this to <date>. This can be 59 max (since 60 would be a new minute)
    date += np.timedelta64(np.random.randint(0, 60),'s')
    return date


def create_time_data(start, end, n_samples=10):
    """"
    Creates a numpy array of Datetime objects from <start> to <end> with a daily interval
    For each date, a random number of hours, seconds and milliseconds is sampled
    If the difference between start and end is small, make sure that <n_samples> is high enough
    """
    # interval is in days
    # first create a numpy array of days only
    dates = np.arange(start, end, dtype='datetime64[D]')
    # change type to include ms
    dates = dates.astype('datetime64[ms]')
    # create an empty array to store the dates with sampled hours, seconds and milliseconds in
    dates_with_h_m_s = []
    # go over each date, sample <n_samples> and add them to dates_with_h_m_s
    for i in tqdm(range(len(dates))):
        for _ in range(n_samples):
            # first make a copy of the date
            new_date = np.copy(dates[i])
            # sample an hour, minute and second
            new_date = sample_hour_min_sec(new_date)
            dates_with_h_m_s.append(new_date)
    # convert to np array
    dates_with_h_m_s = np.asarray(dates_with_h_m_s, dtype='datetime64[ms]')
    
    return dates_with_h_m_s

time_data = create_time_data(START_DATE, END_DATE)

#Create Data folder if not exist
data_folder_check  = os.path.isdir('Data')
if not data_folder_check:
    os.makedirs('Data')
    print('Created Data folder as it was not present')

#Save the data to a file
np.save('Data/np_datetime_data.npy', time_data)
print("Saved Data to Folder")