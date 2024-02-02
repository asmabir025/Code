'''
Name: AKM Sadman (Abir) Mahmud
Date: Mar 21, 2023
'''

"""
  Class Battlefield defines a general structure of a Battlefield
  which consists of a two-dimensional array of characters.
  Each symbol on the Battlefield represents either a possible path
  or a blocker.
  
"""

import random     # random number generator

class Battlefield:
  '''
    Attributes:
      _MAXROW; int; the number of rows in Battlefield.
      _MAXCOL; int; the number of columns in Battlefield.
      PASSAGE, BLOCKER, THEWAYOUT; characters representing if a cell of the Battlefield is open, blocked, or traveled.
      PATH_BLOCKER_RATIO; float; used in the generation of random Battlefields.
      _the_battlefield; list of strings; 2D list representing the map. _the_battlefield[row][col] 
	  shows if that row and column are PASSAGE, BLOCKER, or THEWAYOUT.
  '''
  def __init__( self, max_row = 12, max_col = 12 ):
    """Create a battlefield
    Parameters:
      max_row; int; the number of rows in the randomly generated battlefield.
      max_col; int; the number of columns in the randomly generated battlefield.
    """
    
    self.PASSAGE = ' '
    self.BLOCKER      = '*'
    self.THEWAYOUT    = '!'

    self.PATH_BLOCKER_RATIO = 0.5

    self._MAXROW = max_row
    self._MAXCOL = max_col

    self._the_battlefield = self._gen_battlefield()

  def _gen_battlefield( self ):
    """Generate a random battlefield based on probability"""
    
    local_battlefield = [['*' for i in range( self._MAXCOL  )] \
                      for j in range( self._MAXROW )]

    for row in range( self._MAXROW ):
        for col in range( self._MAXCOL ):
            threshold = random.random()
            if threshold > self.PATH_BLOCKER_RATIO:
                local_battlefield[ row ][ col ] = self.PASSAGE
            else:
                local_battlefield[ row ][ col ] = self.BLOCKER

    return local_battlefield
    
  def __str__( self ):
    """Generate a string representation of the battlefield.
    Parameters: none.
    Return: string."""

    # Change this so that it looks more like a 2D battlefield
    #return str(self._the_battlefield)

    s = ' '
    
    for n in range(self._MAXCOL):
      s += str(n%10)
    s += '\n'
    
    for i in range(self._MAXROW):
      s+= str(i%10)
      for j in range(self._MAXCOL):
        s += self._the_battlefield[i][j]
      s += '\n'

    return s
   
  def get_col_size( self ):
    """Return: int; column count"""
    return self._MAXCOL

  def get_row_size( self ):
    """Return: int; row count"""
    return self._MAXROW

  def read_battlefield( self, file_name ):
    """Reading battlefield from a file.
        The file should be in the form of a matrix, e.g.,
        ** *
        *  *
        ** *
        ** *
        would be a 4x4 input battlefield.
        
      Parameters:
        file_name: string; the name of the file storing battlefield.
      Return:
        none
        """
    
    in_file=open(file_name) 
    lines = in_file.read()
    split_lines = lines.split("\n") 
    
    self._MAXROW = len(split_lines)
    self._MAXCOL = max([len(str) for str in split_lines])
    
    new_battlefield = [[split_lines[i][j] for j in range( self._MAXCOL  )] \
                      for i in range( self._MAXROW )]
    
    self._the_battlefield = new_battlefield
        
  def get_battlefield( self ):
    """Return: a battlefield object; this battlefield"""
    return self._the_battlefield

  def is_clear( self, row, col ):
    """
    Checks if the battlefield at row col is clear (i.e., has value PASSAGE).
    Parameters:
      row; int; row to check.
      col; int; col to check
    Return:
      True if this cell is clear, false otherwise."""
    return self.get_value(row, col) == self.PASSAGE

  def is_in_battlefield( self, row, col ):
    """
    Determines in a row, col is within the bounds of the battlefield.
    Attributes:
      row; int.
      col; int.
    Return:
        True if a cell is inside the battlefield. False otherwise.
    """
    if 0 <= row <= self._MAXROW - 1 and 0 <= col <= self._MAXCOL - 1:
      return True 
      
    return False

  def set_value( self, row, col, value ):
    """Set the value to a cell in the battlefield.
    Parameters:
      row; int.
      col; int.
    Return:
      none
    """
    self._the_battlefield[row][col] = value

  def get_value( self, row, col ):
    """Return the value of the given cell.
    Parameters:
      row; int.
      col; int.
    Return:
      string: value at row, col in the battlefield.
    """
    return self._the_battlefield[row][col]

  def __eq__(self, other):
    '''returns True if two Battlefields are the same
    Parameter:
    self; an instance of Battlefield class
    other; an instance of Battlefield class
    '''
    if self._the_battlefield == other._the_battlefield:
      return True 
    return False