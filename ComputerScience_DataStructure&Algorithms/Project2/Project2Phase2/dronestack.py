'''Name: AKM Sadman (Abir) Mahmud
Date: Mar 26, 2023'''

"""This file contains Drone class and its methods, which solves and saves 
a possible path for the drone to get out of the battlefield. 
"""

from stack import Stack
import numpy as np
import copy as cp


class Drone:
  '''Attributes:
  found_path; boolean type. Initialized as False
  path; list (of tuples) type. Saves all (row, col) of the solved track plus 
  coordinates of unsuccessful tracks.
  '''
  def __init__(self):
    '''Initiates a Drone object.
    Parameter:
    self; an instance of Drone class. 
    '''
    self.found_path = False
    self.path = []

  def find_battlefield_paths( self, battlefield, 
                             starting_row, starting_col, 
                             exit_row, exit_col ):
    '''If there is a possible path out, this method prints the battlefield 
    with the path. Else, it prints the message of no path.
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
    #Checking the validity of starting_row, starting_col, exit_row, exit_col 
  
    s = 0
    if ( battlefield.is_in_battlefield(starting_row, starting_col) and 
      battlefield.is_in_battlefield(exit_row, exit_col) ):
      if not ( battlefield.is_clear(starting_row, starting_col) and 
              battlefield.is_clear(exit_row, exit_col) ):
        
        s = 1

    else:
      s = 1

    if s == 1:
      print('No Path Found')
      return

    self.solve_paths( battlefield, 
               starting_row, starting_col,
               exit_row, exit_col ) 
                               
    if self.found_path == True:  
      single_path = self.single_path()
      
      for point in single_path:
        battlefield.set_value(point[0], point[1], '!')
      print(battlefield)
      
    else:
      print('No Path Found')


  def solve_paths( self, battlefield, 
                  starting_row, starting_col, 
                  exit_row, exit_col ):
    '''This method figures out the possible path to the exit from battlefield.
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
    S = Stack()
    S.push((starting_row, starting_col))
    local_battlefield = cp.deepcopy(battlefield)
                    
    while not S.is_empty(): 
      
      current = S.pop()
      cur_row = current[0]
      cur_col = current[1]
      
      local_battlefield.set_value(cur_row, cur_col, local_battlefield.THEWAYOUT)
      self.path.append((cur_row, cur_col))
      
      if (cur_row, cur_col) == (exit_row, exit_col):

        self.found_path = True
        print("Success!") 
        print('\n')
        print('We found a path!')
      
        break 
    
      else:
            
        for neighbor in neighbors( cur_row, cur_col ):
          if local_battlefield.is_in_battlefield( neighbor[0], neighbor[1] ):
            if local_battlefield.is_clear( neighbor[0], neighbor[1] ):
              S.push( (neighbor[0], neighbor[1]) )  


  def single_path(self):
    '''Removes the part of the unsuccessful track from the solved path. 
    Returns solved path with only the successful trail. 
    Parameter:
    self; an instance of Drone class.
    '''
    index_list = []
    
    if len(self.path) > 1:
      for i in range(1, len(self.path)):  
        
        neighbor_list_i = neighbors(self.path[i][0], self.path[i][1])

        if self.path[i-1] not in neighbor_list_i:
      
          j = i - 1
          while j >= 0 and self.path[j] not in neighbor_list_i:
            j -= 1
  
          index_list += [k for k in range(j+1, i)]
      
      #source: https://numpy.org/doc/stable/reference/generated/numpy.delete.html    
      single_path = np.delete(np.array(self.path), np.array(index_list, dtype=int), 
                              axis=0)
  
    else:
      single_path = self.path

    return single_path
    
  
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