'''Name: AKM Sadman (Abir) Mahmud
Date: Mar 23, 2023'''

"""This file contains Drone class and its methods, which solves and saves 
all possible paths for the drone to get out of the battlefield. 
"""

import copy

class Drone:
  '''Attributes:
  num_path; (type: integer) the number of solved paths for the drone
  all_paths; (type: list) the list of all saved battlefields, which 
            contain a specific solved path
  all_move_num: (type: list) the list of the numbers of moves to reach 
                the exit for a specific solved path
  '''
  
  def __init__(self):
    '''Initiates a Drone object.
    Parameter:
    self; an instance of Drone class. 
    '''
    self.num_path = 0

    self.all_paths = []
    self.all_move_num = []
  
  def find_battlefield_paths(self, battlefield, 
                             starting_row, starting_col, 
                             exit_row, exit_col):
    '''If there is/are a possible path(s) out, this method prints the battlefield 
    with the path(s). Else, it prints the message of no path.
    Parameter:
    self; an instance of Drone class.
    battlefield; type: battlefield.
    starting_row; (type: integer) A non-zero integer less than the maximum 
                  row size of battlefield.
    starting_col; (type: integer) A non-zero integer less than the maximum
                  column size of battlefield.
    exit_row; (type: integer) A non-zero integer less than the maximum row 
              size of battlefield.
    exit_col; (type: integer) A non-zero integer less than the maximum column 
              size of battlefield.
    '''
    #No point figuring out the path if exit-row and exit-col are not valid                           
    if battlefield.is_in_battlefield(exit_row, exit_col): 
      if battlefield.is_clear(exit_row, exit_col):                          
        self.solve_paths(battlefield, 
                        starting_row, starting_col, 
                        exit_row, exit_col)        
                               
    if self.num_path > 0:
    
          for battlefields in self.all_paths:
            print("It's a path!!")
            print(battlefields)

    else:
      print('No Path Found')
      
  def solve_paths(self, battlefield, 
                  starting_row, starting_col, 
                  exit_row, exit_col):  
    '''This method figures out possible paths to the exit from battlefield.
    Parameter:
    self; an instance of Drone class.
    battlefield; type: battlefield.
    starting_row; (type: integer) A non-zero integer less than the maximum 
                  row size of battlefield.
    starting_col; (type: integer) A non-zero integer less than the maximum 
                  column size of battlefield.
    exit_row; (type: integer) A non-zero integer less than the maximum row 
              size of battlefield.
    exit_col; (type: integer) A non-zero integer less than the maximum column 
              size of battlefield.
    '''
    
    local_battlefield = copy.deepcopy( battlefield )
    # base case 
    if starting_col == exit_col and starting_row == exit_row: 
      
      local_battlefield.set_value(starting_row, starting_col, 
                                  local_battlefield.THEWAYOUT)
      
      self.num_path += 1
      self.add_path(local_battlefield) 
      
    # recursive case 
    elif local_battlefield.is_in_battlefield(starting_row, starting_col):
      if local_battlefield.is_clear(starting_row, starting_col): 
      
        local_battlefield.set_value(starting_row, starting_col, '!')
        
        for neighbor in neighbors(starting_row, starting_col):
          self.solve_paths(local_battlefield, 
                            neighbor[0], neighbor[1], 
                            exit_row, exit_col)

  def add_path(self, battlefield):
    '''Adds a Battlefield with a unique solved path to the all_paths 
    attribute and updates the list all_move.num with the number of 
    movements associated with the path. 
    Parameter:
    self; an instance of Drone class.
    battlefield; an instance of Battlefield class.
    '''
    self.all_paths.append(battlefield)
    
    move_num = 0 
    for i in range(battlefield.get_row_size()):
      for j in range(battlefield.get_col_size()):

        if battlefield.get_battlefield()[i][j] == battlefield.THEWAYOUT:
          move_num += 1
          
    self.all_move_num += [move_num]
    
  def show_shortest_path(self):
    '''Figures out the shortest path(s) out of all paths and prints it/them.
    Parameter:
    self; an instance of Drone class.
    '''
    if self.num_path > 0:
      
      minpath_length = min(self.all_move_num)
      minpath_index_list = []
  
      for i in range(self.num_path):
        if self.all_move_num[i] == minpath_length:
          minpath_index_list.append(i)
          
      print('The Shortest Path(s)')
      print('\n')
      for index in minpath_index_list:
        print(self.all_paths[index])


#Helper function
def neighbors(row, col):
  '''This function figures out the neighbors (top, bottom, right, left) of a 
  specific cell.
  Parameter:
  row; an integer
  col; an integer
  returns a list.
  '''
  row_top = row - 1
  row_bottom = row + 1
  col_right = col + 1
  col_left = col - 1
  neighbor_list = [(row, col_right), (row, col_left), (row_top, col), (row_bottom, col)]

  return neighbor_list

  
      

      
      
      

      
      
        
    
    