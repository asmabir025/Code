####################################################
#
#            CSCI 203 Final Project
#                  Fall 2022
#
# AKM Sadman Mahmud
#
####################################################
# Import Libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
#
####################################################
# CONSTANTS
# DONE - Do not modify.
# Indices for accessing data in the csv files
STATE_INDEX = 1  # state name in both csv files
YEAR_INDEX_HOSP = 2  # year in hospitalizatons.csv
HOSP_INDEX = 3  # hospitalizations in hospitalizatons.csv
YEAR_INDEX_HEAT = 4  # year in heat_index.csv
HEAT_INDEX = 5  # number of days when the heat index >90 degree F in heat_index.csv
# States included in hospitalizations.csv
STATES = [
  'Alaska', 'Arizona', 'California', 'Colorado', 'Connecticut', 'Florida',
  'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
  'Massachusetts', 'Michigan', 'Minnesota', 'Missouri', 'New Hampshire',
  'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'Oregon',
  'Pennsylvania', 'Rhode Island', 'South Carolina', 'Tennessee', 'Utah',
  'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin'
]
# A list of the options given to the user
OPTIONS = [
  "Print hospitalization data", "Print heat index data",
  "Plot data as a function of year",
  "Plot hospitalization data as a function of heat index data", "Quit"
]

####################################################
# For testing
# This allows selection of the smaller data files.
DEBUG = False


#
####################################################
# TO DO - Complete the function for reading the data
####################################################
def read_data_file(filename, year_index, state, data_index):
  '''  Reads in the data from a csv file for a given state
  and creates two lists: years and data
  
  Parameters:
    - filename (str): name of the csv data file
    - state (str): name of the state
    - year_index (int): index for the year in the csv file
    - data_index (int): index for the data in the csv file

  Returns:
    - years (list): a list of integers, years that the data was recorded.
    - data (list): a list of integers, the data
  '''
  years = []
  data = []
  with open(filename, newline='') as csvfile:  # creates a file object
    csvreader = csv.reader(csvfile, delimiter=',')  # set for reading csv file
    next(csvreader)  # skips over header
    for row in csvreader:  # reads one row at a time
      # TO DO ***** - Replace the next three lines with your code
      if row[year_index] != '' and row[data_index] != '':
        if row[STATE_INDEX] == state and int(row[year_index]) < 2016:
          years.append(int(row[year_index]))
          data.append(int(row[data_index]))

      #print(row)
      #print("row is data type: ", type(row), "\n")

  return years, data


####################################################
# TO DO - Complete the Data class
####################################################
class Data:

  def __init__(self, filename, year_index, state, data_name, data_index):
    ''' The constructor for the Data class '''
    self.state = state  # A string, the name of the state
    self.data_name = data_name  # A string, the type of data
    # Use the read_data_file to generate the lists of the years and the data
    self.years, self.data = read_data_file(filename, year_index, state,
                                           data_index)

  def __repr__(self):
    ''' A representation of the Data class '''
    border = 30 * '-' + "\n"
    s = border
    s += f'{"Data for " + self.state:^30}' + "\n"
    s += f'{"Year":^10}{self.data_name:^20}' + "\n"
    s += border
    for index in range(len(self.years)):
      s += f'{self.years[index]:^10}{self.data[index]:^20}' + "\n"
    return s

  def group_yearly_nums(self):
    ''' Organizes the data by year
        
    Returns:
    - d (dict): 
      - the keys are the years
      - the values are lists of integers
      the data for the year that is the key
    '''
    d = {}
    # TO DO ***** - Add code here - *****
    for i in range(len(self.years)):
      if self.years[i] in d.keys():
        d[self.years[i]] += [self.data[i]]
      else:
        d[self.years[i]] = [self.data[i]]
    return d


####################################################
# TO DO - Complete the Hospitalizations class
####################################################
class Hospitalizations(Data):

  def __init__(self, filename, state):
    ''' The constructor of the Hospitalizations class'''
    super().__init__(filename, YEAR_INDEX_HOSP, state, "Hospitalizations",
                     HOSP_INDEX)
    self.d_hosp = self.make_yearly_sum()

  def __repr__(self):
    ''' A representation of the Hospitalizations class'''
    # TO DO ***** - Add code here - *****
    ''' A representation of the Data class '''
    border = 35 * '-' + "\n"
    s = border
    s += 10 * ' ' + f'{"Number of Heat-Related":^5}' + '\n'
    s += f'{"Year":^10}{"Hospitalizations":^25}' + '\n'
    s += f'{"for "+ self.state:^47}' + "\n"
    s += border
    for key in self.d_hosp:
      s += f'{key:^10}{self.d_hosp[key]:^25}' + "\n"
    return s

  def make_yearly_sum(self):
    ''' Returns a dictionary with the yearly sums
    of heat-related hospitalizations
        
    Returns:
      - d_new (dict): 
        - the keys are the years
        - the values are integers, the sum of the
              heat-related hospitalizations for the year
    '''
    d_new = {}
    # TO DO ***** - Add code here - *****
    for i in range(len(self.years)):
      if self.years[i] in d_new.keys():
        d_new[self.years[i]] += self.data[i]
      else:
        d_new[self.years[i]] = self.data[i]
    return d_new


####################################################
# TO DO Complete the HeatIndex class
####################################################
class HeatIndex(Data):

  def __init__(self, filename, state):
    ''' The constructor of the HeatIndex class'''
    # TO DO ***** - Add code here - *****
    super().__init__(filename, YEAR_INDEX_HEAT, state, "Heat Index",
                     HEAT_INDEX)
    self.d_heat = self.make_yearly_av()

  def __repr__(self):
    ''' A representation of the HeatIndex class'''
    # TO DO ***** - Add code here - *****
    border = 35 * '-' + "\n"
    s = border
    s += 12 * ' ' + f'{"Number of Days with a":^10}' + '\n'
    s += f'{"Year":^10}{"Heat Index > 90 degrees F":^7}' + '\n'
    s += f'{"for "+ self.state:^47}' + "\n"
    s += border
    for key in self.d_heat:
      s += f'{key:^10}{self.d_heat[key]:^25}' + "\n"
    return s

  def make_yearly_av(self):
    ''' Returns a dictionary with the yearly averages
    of the number of number of days when the heat index 
    was over 90 degrees F
        
    Returns:
      - d_new (dict): 
          - key (int): the year
          - value (float) the average of the days 
          when the heat index was over 90 degrees F
          for a year
    '''
    d_new = {}
    # TO DO ***** - Add code here - *****
    count = {}
    for i in range(len(self.years)):
      if self.years[i] in d_new.keys():
        d_new[self.years[i]] += self.data[i]
        count[self.years[i]] += 1
      else:
        d_new[self.years[i]] = self.data[i]
        count[self.years[i]] = 1

    for key in d_new:

      d_new[key] = round(d_new[key] / count[key], 2)

    return d_new


####################################################
# TO DO Complete the helper functions for plotting
####################################################
def get_years_nums_from_d(d):
  ''' Returns the lists of the keys and values of d
    
  Parameter:
  - d(dict): 
    - the keys are the years when the data was obtained
    - the values are integers or floats, the data
            
  Returns:
  - years (a list of integers), the keys of d
  - nums (a list of floats or integers), the values of d
  '''
  # TO DO ***** - Add code here - *****
  years = []
  nums = []
  for key in d:
    years.append(key)
    nums.append(d[key])

  return years, nums


def get_hosp_heat_from_d(d_heat_plot, d_hosp_plot):
  ''' Returns lists of the values of the dictionaries,
  d_heat_plot and d_hosp_plot for years
  when data is available for both dictionaries
    
  Parameters:
    - d_heat_plot (dict): 
      - keys (int): the years for the heat index data
      - values (float): the yearly averages of the number of days
      when the heat index exceeeded 90 degrees F
    - d_hosp_plot (dict):
      - keys (int): the years for the hospitalization data
      - values (float): the yearly sums of the 
      heat-related hospitalizations
  Returns:
    - heat_nums (list): a list of floats, the values for d_heat_plot
    for years included in d_hosp_plot
    - hosp_nums (list): a list of integers, the values for
    for d_hosp_plot
  '''
  heat_nums = []
  hosp_nums = []
  # TO DO ***** - Add code here - *****
  years = []
  for key in d_hosp_plot:
    years.append(key)
    hosp_nums.append(d_hosp_plot[key])

  for year in years:
    if year in d_heat_plot.keys():
      heat_nums.append(d_heat_plot[year])

  return heat_nums, hosp_nums


def get_stats(x, y):
  ''' Returns the correlation coefficient and
  predicted y values for given x values 
  using a linear regression.
    
  Parameters:
    - x (list): the independent variable values
    - y (list): the dependent variable values
        
  Returns:
    - correlation (float): the correlation coefficient
    rounded to three decimal places
    - y_predicted (list): a list of values 
    calculated from slope*x + intercept
  '''
  # Determine the correlation coefficient
  correlation = np.corrcoef(x, y)[0][1]
  # Perform a linear regression
  fit = np.polyfit(x, y, 1)
  slope = fit[0]
  intercept = fit[1]
  # Determine predicted values for the dependent variable
  y_predicted = [slope * z + intercept for z in x]
  return round(correlation, 3), y_predicted


####################################################
# Plotting functions
####################################################
def plot_data_by_year(state, heat_years, heat_nums, hosp_years, hosp_nums):
  ''' DONE FOR YOU
  Makes two plots:
  - the yearly sum of heat-related hospitalizations as a function of years
  - the yearly average of the number of days 
  with a heat index over 90 degrees Fahrenheit as a function of years.

  Parameters:
  - heat_years (a list of integers) the years 
  for which the heat index data will be plotted
  - heat_nums (a list of floats) the yearly average numbers of days 
  when the heat index exceeded 90 degrees Fahrenheit
  - hosp_years (a list of integers) the years 
  for which the hospitalization data will be plotted
  - hosp_nums (a list of integers) the yearly sums 
  of heat-related hospitalizations
  '''
  # Add first set of axes
  fig, ax1 = plt.subplots()
  # Find min and max values for the x axis
  xmin = min(hosp_years)
  xmax = max(hosp_years)
  ax1.set_xlim([xmin, xmax])
  # Add labels
  ax1.set_xlabel('Year')
  ax1.set_ylabel('Sum of hospitalizations', color='red')
  ax1.set_title('Data for ' + state)
  # Plot hospitalization data
  plot_1 = ax1.plot(
    hosp_years, hosp_nums, color='red',
    label='Heat-related Hospitalizations')  #add lines between data
  plot_2 = ax1.plot(hosp_years, hosp_nums, 'o', color='red')  #add data markers
  ax1.tick_params(axis='y', labelcolor='red')
  # Add second set of axes
  ax2 = ax1.twinx()  #overlay the two trend
  # Add heat index data
  ax2.set_ylabel('Average number of days', color='blue')
  plot_3 = ax2.plot(heat_years, heat_nums, 'o',
                    color='blue')  #add data markers
  plot_4 = ax2.plot(heat_years,
                    heat_nums,
                    color='blue',
                    label='Heat Index >90F')  #add lines between data
  ax2.tick_params(axis='y', labelcolor='blue')
  # Add legends
  lns = plot_1 + plot_4
  labels = [l.get_label() for l in lns]
  plt.legend(lns, labels, loc=0)
  # Save to file
  plt.savefig("DataByYear.png")


def plot_heat_hosp(state, heat_nums, hosp_nums):
  ''' Plot plot the yearly sum of heat-related hospitalizations 
  as a function of the yearly average of the number of days
  with a heat index over 90 degrees Fahrenheit
    
  Parameters:
  - heat_nums (list): a list of floats, the yearly average values
  of the heat index that includes data for only the years 
  that data is available for hospitalizations
  - hosp_nums (list), a list of integers, the yearly sum 
  of heat-related hospitalization
  '''
  # Open a new figure
  plt.figure()
  # Add labels to the plot
  plt.xlabel("Yearly Average of Days w/ Heat Index > 90 oF")
  plt.ylabel("Yearly Sum of Hospitalizations")
  plt.title("Data for " + state)
  # Plot data
  plt.plot(heat_nums, hosp_nums, 'ro', label="Data")
  # Run the linear regression
  correlation, y_predicted = get_stats(heat_nums, hosp_nums)
  plt.plot(heat_nums, y_predicted, label="Linear Regression")
  xposition_for_text = min(heat_nums)
  yposition_for_text = .75 * (max(hosp_nums) - min(hosp_nums)) + min(hosp_nums)
  plt.text(xposition_for_text, yposition_for_text,
           "Correlation = " + str(correlation))
  plt.legend()
  plt.savefig("HeatHospData.png")


####################################################
# TO DO - Complete functions for automating analysis
####################################################
def select_a_state():
  ''' Repeatedly asks the user for a state
  until the user supplies a state given in the list STATES
    
  Return:
  - state (str): a state in STATES
  '''
  while True:
    state = input("Select a state: ")
    if state in STATES:
      return state
    else:
      print(
        "You entered a state without hospitalization data or an incorrect spelling"
      )


def select_option():
  ''' Repeatedly asks the user for a selection
  until the user supplies a valid selection
    
  Return:
  - selection (int): a valid number
  '''

  menu()

  selection = input("Enter a number from the menu: ")
  if selection.isnumeric() == True:
    selection = int(selection)
    if 0 < selection < 6:
      print(selection)
      return selection

  print("Invalid selection. You must enter a number from 1 to 5.")
  select_option()


def menu():
  ''' Prints options for the user '''
  border = 60 * '*'
  print(border)
  print("Options")
  for index in range(len(OPTIONS)):
    print(index + 1, OPTIONS[index])
  print(border)


def main():
  state = select_a_state()
  # Filenames of the two csv files
  filename_hosp = "hospitalizations.csv"
  filename_heat = "heat_index.csv"
  if DEBUG:
    # Abridged versions of the above files for testing
    filename_hosp = "test_hospitalizations.csv"
    filename_heat = "test_heat_index.csv"
  # Create an instance of the Hospitalization class
  hosp = Hospitalizations(filename_hosp, state)
  hosp_dic = hosp.make_yearly_sum()
  hosp_years, hosp_nums_separate = get_years_nums_from_d(hosp_dic)
  #print(hosp.state)
  # Remove the following hashtags as you complete code.
  #print(hosp)
  heat = HeatIndex(filename_heat, state)
  heat_dic = heat.make_yearly_av()
  heat_years, heat_nums_separate = get_years_nums_from_d(heat_dic)

  heat_nums, hosp_nums = get_hosp_heat_from_d(heat_dic, hosp_dic)
  #print(heat)
  while True:
    selection = select_option()
    if selection == 1:
      print("Printing Hospitalization Data")
      print(hosp)
    elif selection == 2:
      print("Printing Heat Index Data")
      print(heat)
    elif selection == 3:
      print("Plotting data as a function of year")
      # Add code here
      plot_data_by_year(state, heat_years, heat_nums_separate, hosp_years,
                        hosp_nums_separate)
    elif selection == 4:
      print("Plotting hospitalization data as a function of heat index data")
      # Add code here
      plot_heat_hosp(state, heat_nums, hosp_nums)
    elif selection == 5:
      print("Analysis Complete")
      return


# Following lines run main each time you hit run
# But do not interfere with the autograder.
if __name__ == "__main__":
  main()
