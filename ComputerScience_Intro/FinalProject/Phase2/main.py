########################################
# Final Project Part 2
# Fall, 2022
#
# AKM Sadman Mahmud
#
########################################
# Include import statements here.

import csv
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

########################################
# Add your code here for your analysis
# of a heat-related health issue.

# Indices for accessing data in the csv files
STATE_INDEX = 1  # state name in all csv files
YEAR_INDEX_EXCEPT_HEAT = 2  # year index in all csv files except heat.csv
DATA_INDEX_EXCEPT_HEAT = 3  # data value index in all csv files except heat.csv
YEAR_INDEX_HEAT = 4  # year index in heat.csv
HEAT_INDEX = 5  # index for the number of extreme heat days in heat_index.csv
AGE_INDEX = 5  # age-group index in hospitalizations_crude.csv and emergency_crude.csv
GENDER_INDEX = 6  #gender index in emergency_crude.csv
THRESHOLD_INDEX = 7  #heat threshold index in heat.csv
METRIC_INDEX = 8  #heat metric index in heat.csv

#Lists that are important in multiple stages of analysis
Thresholds = [
  'Relative Threshold: 95th Percentile', 'Relative Threshold: 98th Percentile'
]
Metrics = [
  'Heat Metric: Daily Maximum Temperature',
  'Heat Metric: Daily Maximum Heat Index'
]
age_grp_list = ['0 TO 4', '5 TO 14', '15 TO 34', '35 TO 64', '>= 65']

# A list of the options given to the user
OPTIONS = [
  "Compare rate of emergency visits for different genders",
  "Compare rate of emergency visits with other age groups",
  "Plot emergency visit and hospitalization data as a function of heat index data",
  "Relation between age-adjusted hospitalization and mortality", "Quit"
]


def read_data_file(filename='', age_grp='', threshold='', metric=''):
  '''  Reads in the data from a csv file and creates a dictionary
  
  Parameters:
    - filename (str): name of the csv data file
    - age_grp (str): name of the age group in the csv file
    - threshold (str): name of the heat threshold (from Thresholds list) in the csv file 
    - metric (str): name of the heat metric (from Metrics list) in the csv file 

  Returns:
    - dictionary of dictionaries (of dictionaries), depending on the CSV file
  '''

  data_dict = {}
  with open(filename, newline='') as csvfile:  # creates a file object
    csvreader = csv.reader(csvfile, delimiter=',')  # set for reading csv file
    next(csvreader)  # skips over header

    # for reading mortality.csv and hospitalizations_adjusted.csv :
    if age_grp == '' and threshold == '':
      for row in csvreader:  # reads one row at a time
        if row[DATA_INDEX_EXCEPT_HEAT].isalpha() == False:  # if the cell value is float in this case

          if row[YEAR_INDEX_EXCEPT_HEAT] not in data_dict.keys():
            data_dict[row[YEAR_INDEX_EXCEPT_HEAT]] = {}  # first creating an empty dictionary for the key

          data_dict[row[YEAR_INDEX_EXCEPT_HEAT]][row[STATE_INDEX]] = float(row[DATA_INDEX_EXCEPT_HEAT])  
          # adding age-adjusted emergency visit or hospitalization rate as the value to the most inside dictionary

    #for reading heat.csv :
    if threshold != '':
      for row in csvreader:  # reads one row at a time

        if row[YEAR_INDEX_HEAT] not in data_dict:
          data_dict[row[YEAR_INDEX_HEAT]] = {
          }  # initiating an empty dictionary for the key

        if row[THRESHOLD_INDEX] == threshold and row[
            METRIC_INDEX] == metric:  # if the row's heat threshold and heat metric are the same ones we want
          if row[STATE_INDEX] not in data_dict[row[YEAR_INDEX_HEAT]]:
            data_dict[row[YEAR_INDEX_HEAT]][row[STATE_INDEX]] = float(
              row[HEAT_INDEX])  # initiating the value for the extreme heat day

          else:
            #I have defined the sum of the number of extreme heat days of all counties in a state as the quantity for the number of extreme heat days of the state
            data_dict[row[YEAR_INDEX_HEAT]][row[STATE_INDEX]] += float(
              row[HEAT_INDEX])

    #for reading hospitalizations_crude.csv and emergency_crude.csv:
    if age_grp != '':
      for row in csvreader:

        if row[YEAR_INDEX_EXCEPT_HEAT] not in data_dict.keys():
          data_dict[row[YEAR_INDEX_EXCEPT_HEAT]] = {}

        if row[AGE_INDEX] not in data_dict[row[YEAR_INDEX_EXCEPT_HEAT]].keys():
          data_dict[row[YEAR_INDEX_EXCEPT_HEAT]][row[AGE_INDEX]] = {}

        if row[STATE_INDEX] not in data_dict[row[YEAR_INDEX_EXCEPT_HEAT]][
            row[AGE_INDEX]].keys():
          data_dict[row[YEAR_INDEX_EXCEPT_HEAT]][row[AGE_INDEX]][
            row[STATE_INDEX]] = {}

        data_dict[row[YEAR_INDEX_EXCEPT_HEAT]][
          row[AGE_INDEX]][row[STATE_INDEX]][row[GENDER_INDEX]] = float(
            row[DATA_INDEX_EXCEPT_HEAT]
          )  # adding hospitalization or emergency visit rate for a specific age-gender group as a value to the most inside dictionary

  return data_dict


def get_gender_specific_avg(age_grp):
  '''Reads emergency_crude.csv for specific age group lines and gets two nested dictionaries, which have state-differentiated male and female yearly emergency visit data, respectively and creates average, yearly male and female emergency visit rates, respectively  
  
  Parameters:
  - (str)a ge group (any element from age_grp_list)
  Returns:
  - (dictionary) yearly (years are keys) average of particular age-group emergency visit rate, ge
  '''

  data_dict = read_data_file('emergency_crude.csv', age_grp)
  avg_dict_male = {}
  avg_dict_female = {}

  for year in data_dict.keys():
    # declaring variables for sum of data of men and women, respectively and starting count of data
    sum_male = 0
    sum_female = 0
    count = 0

    for state in data_dict[year][
        age_grp]:  # accessing state (inner) keys from the nested dictionary

      sum_male += data_dict[year][age_grp][state]['Male']
      sum_female += data_dict[year][age_grp][state]['Female']
      count += 1

    avg_dict_male[year] = sum_male / count
    avg_dict_female[year] = sum_female / count

  return avg_dict_male, avg_dict_female


def gender_indpen_avg():
  ''' Receives gender-separated, yearly emergency visit rate averages for all separate age groups and creates gender-independent yearly average for all age groups  

  Parameter: NONE

  Returns: a dictionary with years as keys and gender independent average emergency visit rate for separate age groups, which are the inner keys
  
  '''

  avg_dict = {}

  for age in age_grp_list:
    dict1, dict2 = get_gender_specific_avg(age)

    for key in dict1:
      if key not in avg_dict.keys():
        avg_dict[key] = {}  # nesting an empty dictionary inside

      avg_dict[key][age] = (dict1[key] + dict2[key]) / 2

  return avg_dict


def weighted_avg(filename, heat_dict, age_grp):
  '''reads emergency_crude.csv or hospitalizations_crude.csv for particular age group, uses the dictionary read from heat.csv and creates weighted average of data with respect to data values of the dictionary from heat.csv 
  
  Parameters:
  - (str) file name: either emergency_crude.csv or hospitalizations_crude.csv
  - a dictionary with years as outer keys and states as the inner keys and heat data values (the number of extreme heat days) as the inner values
  - (str) particular age group
  
  Returns:
  - a list of yearly weighted averages and a list of yearly count of extreme heat days
  '''

  dict1 = read_data_file(
    filename, age_grp
  )  # reading data from either emergency_crude.csv or hospitalizations_crude.csv
  # initiating dictionaries of weighted averages and the total number of extreme heat days as values, respectively, and common years as keys:
  weighted_dict = {}
  heat_event_dict = {}

  for year in heat_dict.keys():
    if year in dict1.keys():  # accessing only common keys
      heat_event_count = 0  # starting count of the number of extreme heat days for a particular year

      for state in heat_dict[year]:  # accessing inner keys - states
        if state in dict1[year][
            age_grp]:  # accessing only states with both emergency or hospitalization data and heat data

          if year in weighted_dict.keys(
          ):  # adding to the value of existent keys
            weighted_dict[year] += heat_dict[year][state] * (
              dict1[year][age_grp][state]['Male'] +
              dict1[year][age_grp][state]['Female']
            ) / 2  # The sum in the formula for weighted average
          else:  # starting a key, value pair
            weighted_dict[year] = heat_dict[year][state] * (
              dict1[year][age_grp][state]['Male'] +
              dict1[year][age_grp][state]['Female']) / 2

          heat_event_count += heat_dict[year][state]

      weighted_dict[year] = weighted_dict[
        year] / heat_event_count  # averaging according to the weighted average formula
      heat_event_dict[
        year] = heat_event_count  # pairing the number of extreme heat days to the year key

  # initiating lists for weighted averages and extreme heat days
  weighted_list = []
  heat_event_list = []

  # Appending to the above lists with according yearly weighted averages and the number of extreme heat days
  for year in weighted_dict.keys():
    weighted_list.append(weighted_dict[year])
    heat_event_list.append(heat_event_dict[year])

  return weighted_list, heat_event_list


def get_best_correlation(filename, age_grp):
  '''reads emergency_crude.csv or hospitalizations_crude.csv for particular age group, uses the dictionaries read from heat.csv with all possible combinations of heat thresholds and heat matrices and determines which combination leads to best correlation between weighted yearly average hospitalization or emergency visit rate for particular age group and yearly number of extreme days across US
    
  Parameters: 
  - (str) file name: either emergency_crude.csv or hospitalizations_crude.csv
  - (str) particular age group
  
  Returns: A nested list of weighted yearly average data (list), best correlated yearly heat data (list), correlation value (float), and string of best correlation ensuring heat threshold and heat metrix
    
  '''
  data_dict_list = []  # initiating a list for dictionaries from reading heat.csv according to different combinations of heat index
  data_recog_list = []  # initiating a list for saving these combination strings
  correlation_list = []  # initiating a list for catching correlation values for different combinations
  for condition in Thresholds:
    for m in Metrics:
      data_dict_list.append(read_data_file('heat.csv', '', condition, m))
      data_recog_list.append(condition[20:22] + '%, ' + m[27:])

  weighted_nested = []  # initiating a list catching returns (nested lists) from weighted_avg fucntion of arguments of different possible heat data dictionaries
  heat_nested = []  # initiating a list to catch these heat data dictionaries
  for n in range(len(data_dict_list)):
    weighted_list, heat_event_list = weighted_avg(filename, data_dict_list[n],
                                                  age_grp)
    weighted_nested += [weighted_list]
    heat_nested += [heat_event_list]

    # Determine the correlation coefficient
    correlation_list.append(np.corrcoef(heat_event_list, weighted_list)[0][1])

  # determining the highest absolute value of correlations

  best_correlate = correlation_list[0]
  l = 0
  for i in range(len(correlation_list)):
    if abs(correlation_list[i]) > abs(best_correlate):
      best_correlate = correlation_list[i]
      l = i

  # getting weighted average data list and heat data list for the best correlation value
  y_best = weighted_nested[l]
  x_best = heat_nested[l]

  return [y_best, x_best, correlation_list[l], data_recog_list[l]]


def hosp_mort_relation():
  '''reads hospitalizations_adjusted.csv and mortality.csv, creates lists of hospitalization data and mortality data, and figures out the best polynomial relationship between these two

   Returns: A nested list with age-adjusted hospitalization and mortality data lists,  
  '''
  # inspired from: https://stackoverflow.com/questions/49179463/python-quadratic-correlation

  hosp_dict = read_data_file('hospitalizations_adjusted.csv')
  mort_dict = read_data_file('mortality.csv')

  # initializing 2 lists to append yearly average hospitalization rates and annual deaths, respectively
  hosp_list = []
  mort_list = []

  for year in hosp_dict.keys():
    if year in mort_dict.keys():
      for state in mort_dict[year]:
        if state in hosp_dict[year]:
          hosp_list.append(hosp_dict[year][state])
          mort_list.append(mort_dict[year][state])

  fit_list = [
  ]  # initializing a list to contain polyfit coefficient lists for 5 different values of degree n
  for n in range(1, 6):
    fit_list.append(np.polyfit(hosp_list, mort_list, n))

  predicted_mortlist = [
  ]  # initializing an empty list to contain 5 different lists of predicted mortality values according to 5 different degrees
  for n in range(5):
    predicted_mortlist.append(np.polyval(fit_list[n], hosp_list))

  std_list = [
  ]  # iniitializing a list to contain 5 different values of standard deviation
  for n in range(5):
    std_list.append(np.mean(np.square(mort_list - predicted_mortlist[n])))

  # determining lowest standard deviation:
  min_std = std_list[0]
  m = 1

  for i in range(len(std_list)):
    if std_list[i] < min_std:
      min_std = std_list[i]
      m = i + 1

  return [hosp_list, mort_list, predicted_mortlist[m - 1], fit_list[m - 1], m]


def get_bar_plot(age_grp):
  '''Sorts years from the dictionaries returned by the function of get_gender_specifc_avg for a particular age group and enlists the gender-separated values according to the sort.
   Makes a bar plot with years in x-axis. For each year, 2 bars corresponding to male and female genders.
   Y-axis represents the average yearly emergency visit rate
  '''
  # source: https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/

  male_dict, female_dict = get_gender_specific_avg(age_grp)
  x_list = sorted(male_dict.keys())  # sorting years

  # Making corresponding male and female data lists out of dictionaries
  Ymale = []
  Zfemale = []
  for x in x_list:
    Ymale.append(male_dict[x])
    Zfemale.append(female_dict[x])

  X_axis = np.arange(len(x_list))

  plt.bar(X_axis - 0.2, Ymale, 0.4, label='Male')
  plt.bar(X_axis + 0.2, Zfemale, 0.4, label='Female')
  plt.xticks(X_axis, x_list, fontsize=6)
  plt.xlabel("Year", fontsize=15)
  plt.ylabel("Emergency Visits/10,000", fontsize=15)
  plt.title("Gender Separated Data for " + age_grp, fontsize=20)
  plt.legend()
  plt.savefig("Gender_EMG.png")


def get_histogram(n):
  '''Creates a frame for the animation of horizontal bar charts from gender independent averages of all age groups for a particular year
  Parameter: n (integer), frame number
  Returns NONE
  '''
  #Inspired by https://holypython.com/python-visualization-tutorial/creating-bar-chart-animations/

  plt.cla()

  avg_dict = gender_indpen_avg(
  )  # accessing all separate age-groups' gender independent all yearly averages
  x_list = sorted(
    avg_dict.keys())  # sorting years (keys) of the dictionary into a list
  item_no = len(age_grp_list)  # the number of y-values in this case

  y1 = avg_dict[x_list[n]][age_grp_list[
    0]]  # accessing a yearly average of 0-4 years for the year = x_list[n]
  y2 = avg_dict[x_list[n]][age_grp_list[
    1]]  # accessing a yearly average of 5-14 years for the year = x_list[n]
  y3 = avg_dict[x_list[n]][age_grp_list[
    2]]  # accessing a yearly average of 15-34 years for the year = x_list[n]
  y4 = avg_dict[x_list[n]][age_grp_list[
    3]]  # accessing a yearly average of 35-64 years for the year = x_list[n]
  y5 = avg_dict[x_list[n]][age_grp_list[
    4]]  # accessing a yearly average of >=65 years for the year = x_list[n]

  plt.barh(range(item_no),
           sorted([y1, y2, y3, y4, y5]),
           color=['violet', 'blue', 'green', 'orange',
                  'red'])  # horizontal bar chart with sorted y-values

  tickdic = {
    "0-4": y1,
    "5-14": y2,
    "15-34": y3,
    "35-64": y4,
    ">=65": y5
  }  # creating a dictionary with age-groups as keys and the yearly averages of particular age groups as values
  sorted_tickdic = sorted(
    tickdic.items(), key=lambda x: x[1]
  )  # sorting the items of the above dictionary according to the dictionary values
  tcks = [i[0] for i in sorted_tickdic
          ]  # creating a list of vertical axis tick values

  plt.xlim(0,
           35)  # 35 is the highest value of emergency visit rate for any group
  plt.xlabel("Emergency Visit Rate", color='blue')
  plt.ylabel('Age Group', color='blue')
  plt.title("Animation of Emergency Visit Rate, Year: {} ".format(x_list[n]),
            color=("blue"))
  plt.yticks(np.arange(item_no), tcks)


def save_animation():
  '''Saves the animation of the frames from get_histogram function 
  '''
  fig = plt.figure()
  ani = FuncAnimation(
    fig, get_histogram, frames=21, interval=1000
  )  # total frame = total number of years = 21, interval between two frames = 1000 microseconds = 1s

  f = "Animation.png"
  writergif = animation.PillowWriter(
    fps=1)  # determining frame per second of the animation
  ani.save(f, writer=writergif)


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

  # Perform a linear regression
  fit = np.polyfit(x, y, 1)
  slope = fit[0]
  intercept = fit[1]
  # Determine predicted values for the dependent variable
  y_predicted = [slope * z + intercept for z in x]
  return y_predicted


def weighted_data_vs_weather(age_grp, double_nested1, double_nested2):
  ''' 
  Makes two subplots:
  - the yearly weighted average of heat-related hospitalization rates as a function of the yearly number of extreme heat days
  - the yearly weighted average of heat-related emergency visit rates as a function of the yearly number of extreme heat days 

  Parameters:
  - age group (string)
  - double_nested1: nested list, which is the return of get_best_correlation('hospitalizations_crude.csv', age_grp)
  - double_nested2: nested list, which is the return of get_best_correlation('emergency_crude.csv', age_grp)
  '''

  # Determine predicted values for the dependent variable
  y_predicted1 = get_stats(double_nested1[1], double_nested1[0])

  # Add first set of axes
  fig, ax1 = plt.subplots()

  # Add labels
  ax1.set_xlabel('The Number of Extreme Heat Days')
  ax1.set_ylabel('Weighted Average Hospitalization Rate', color='red')
  ax1.set_title('Data for Age Group ' + age_grp)
  plot_1 = ax1.plot(double_nested1[1],
                    y_predicted1,
                    color='red',
                    label='correlation = ' + str(round(double_nested1[2], 2)) +
                    ' :' + double_nested1[3])  #add lines between data
  ax1.plot(double_nested1[1], double_nested1[0], 'o',
           color='red')  #add data markers
  ax1.tick_params(axis='y', labelcolor='red')

  # Determine predicted values for the dependent variable
  y_predicted2 = get_stats(double_nested2[1], double_nested2[0])

  # Add second set of axes
  ax2 = ax1.twinx()  #overlay the two trend
  # Add heat index data
  ax2.set_ylabel('Weighted Average Emergency Visit Rate', color='blue')
  ax2.plot(double_nested2[1], double_nested2[0], 'o',
           color='blue')  #add data markers
  plot_2 = ax2.plot(double_nested2[1],
                    y_predicted2,
                    color='blue',
                    label='correlation = ' + str(round(double_nested2[2], 2)) +
                    ' :' + double_nested2[3])  #add lines between data
  ax2.tick_params(axis='x', labelcolor='blue')
  ax2.tick_params(axis='y', labelcolor='blue')

  lns = plot_1 + plot_2
  labels = [l.get_label() for l in lns]
  plt.legend(lns, labels, loc='upper left')

  plt.savefig("Weighted_Avg_vs_Weather.png")  # Save to file


def plot_hosp_mort():
  '''Plots best polynomial curve of mortality rate with respect to hospitalization rate
  '''

  nested_list = hosp_mort_relation()
  x_list = nested_list[0]  # x_list represents hospitalization rate list
  y_list = nested_list[1]  # y_list represents annual death number list
  y_predicted1 = nested_list[
    2]  # y_predicted1 is the list of best fit curve values
  coeff = nested_list[3]  # coefficient list for the best fit polynomial
  degree = nested_list[4]  # best fit polynomial degree

  z = list(
    zip(x_list, y_predicted1)
  )  # list of ordered pairs: 1st element - hospitalization rate, 2nd - best fit mortality rate
  z = sorted(
    z, key=lambda x: x[0]
  )  # sorted out ordered pairs according to the first element or hospitalization rate
  sorted_x = [
    s[0] for s in z
  ]  # creating x-values list from the first elements of the sorted pair list
  sorted_y = [
    s[1] for s in z
  ]  # creating y-values list from the 2nd elements of the sorted pair list

  # creating best fit polynomial equation string:
  equation = f"${round(coeff[0],2)}x^{degree}$  "

  for n in range(1, degree + 1):
    if coeff[n] > 0:
      equation += f"$+{round(coeff[n],2)}x^{degree-n}$  "
    elif coeff[n] < 0:
      equation += f"${round(coeff[n],2)}x^{degree-n}$  "

  # Find min and max values for the x axis
  xmin = min(x_list)
  xmax = max(x_list)
  plt.xlim([xmin, xmax])
  # Add labels
  plt.xlabel('Age-Adjusted Hospitalization Per 100,000')
  plt.ylabel('Annual Death')
  plt.title('Correlation Between Mortality & Hospitalization')
  plt.plot(sorted_x, sorted_y, color='blue')  #add lines between data
  plt.text(0.1 * max(sorted_x), max(sorted_y), 'y= ' + equation, fontsize=10)
  plt.plot(x_list, y_list, 'o', color='red')  #add data markers
  plt.savefig('Hospitalization_Mortality.png')  # Save to file


def select_age_grp():
  ''' Repeatedly asks the user for a age group
  until the user supplies an indicative number for a particular age group
    
  Return:
  - age group (str): an element of age_grp_list
  '''

  for index in range(len(age_grp_list)):
    print(index + 1, age_grp_list[index])
  print('\n')

  age_input = int(input("Select an age group by entering the left number:"))

  if age_input in range(1, len(age_grp_list) + 1):
    print('\n' + 'You have selected age-group of ' +
          age_grp_list[age_input - 1] + ' years' + '\n')
    return age_grp_list[age_input - 1]

  print("Invalid selection. You must enter a number from 1 to 5.")
  select_age_grp()


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


########################################
# Add to the main function so
# your analysis runs when main is called.


def main():
  print("Run Analysis" + '\n')

  age = select_age_grp()
  selection = select_option()

  if selection == 1:
    print("Bar Plotting Gender Separated Emergency Visit vs. Number of Years")
    get_bar_plot(age)

  elif selection == 2:
    print("Animation of Bar Plots of All Age-Groups Emergency Visit Data")
    save_animation()

  elif selection == 3:
    print(
      "Plotting Best Fit Emergency Visit and Hospitalization Data vs. Heat Index Data"
    )

    weighted_datalist = []
    for file in ['hospitalizations_crude.csv', 'emergency_crude.csv']:
      weighted_datalist.append(get_best_correlation(file, age))

    weighted_data_vs_weather(age, weighted_datalist[0], weighted_datalist[1])

  elif selection == 4:
    print(
      "Best Fit Curve of Annual Mortality vs. Age-Adjusted hospitalization data"
    )
    plot_hosp_mort()

  elif selection == 5:
    print("Analysis Complete")
    return

  print('\n')
  main()  # to have access to other options after tring an option


if __name__ == "__main__":
  main()
