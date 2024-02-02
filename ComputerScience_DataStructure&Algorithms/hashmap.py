'''Student name(s): AKM Sadman (Abir) Mahmud
Instructor name: Professor Edward Talmage'''

"""Implementation of the Map ADT using closed hashing and a probe with 
double hashing.

This is an incomplete file, students will have to fill in the missing
parts to make the hash map work correctly.
"""

class HashMap :
  """ A closed hashing (one key per slot) with a double hashing
      probe (one hash function that maps into two different capacities)
  
  Attributes:
    CAP; int; max capacity of the hash table.
    _array; arraylist of MapArray items; element is None if nothing stored; 
            to store data; has max capacity CAP. 
    _size; int; how many items are in the array.
    _collisions; int; number of items with same hash value.
  """

  def __init__( self, capacity):                                     
    """Creates an empty map instance.
    Parameters:
      capacity; int; number of slots available in array."""

    # Define the size of the table
    self.CAP = capacity

    # Create the data storage (empty hash table)
    self._array = [None] * self.CAP

    # how many items in the array
    self._size = 0
    
    # how many collisions so far
    self._collisions = 0
      
  def __len__( self ):
    """Returns the number of entries in the map."""
    return self._size

  def _hash1( self, key ):                               
    """The main hash function for mapping keys to table entries.
    Parameter: 
      key; hashable key.
    Return:
      int; value from hash function."""
    return abs( hash( key ) ) % len( self._array )
  
  def _hash2( self, key ):
    """ The second hash function used for backup.
    Parameter: 
      key; hashable key.
    Return:
      int; value from hash function."""
    return 1 + abs( hash( key ) ) % ( len( self._array ) - 2 )    
  
  def add( self, key, value ):
    """ If the key does not exist, adds a new entry to the map.
        If the key exists, update its value.
      Paramters:
        key; any valid value to use in hashfunciton.
        value; to store in hashmap.
      Return:
        None
    """
    ## STUDENTS WILL COMPLETE THE REST OF THIS METHOD
      
    new_map_entry = _MapEntry( key, value )
    slot1, content1 = self._lookup(self._hash1, key)
    s = 0
    
    if slot1 != None: # occupied by same key from hash1
      
      if content1 == None:
        s = 1 # keeping tab for increasing size
      self._array[slot1] = new_map_entry

    if slot1 == None:
      self._collisions += 1

      slot2, content2 = self._lookup(self._hash2, key)
      if slot2 != None:  # occupied by same key from hash2
        
        if content2 == None:
          s = 2 # keeping tab for increasing size
        self._array[slot2] = new_map_entry
    
    if s == 1 or s==2:
      self._size += 1
      
  def peek( self, key ):
    """ Returns the value associated with the key or returns None.
    Parameters:
      key; key to look up in hashmap.
    Return:
      value associated with key or None"""
    slot,contents = self._lookup(self._hash1, key)
    if contents == None:
      slot,contents = self._lookup(self._hash2, key)
      
    if contents == None:
      return None
    else:
      return contents.value

  def _lookup(self, hashFunction, key):
    """ Parameters:
        hashFunciton, function to use;
        key; key to look up using hashFunction;
      Returns:
        a tuple with two values.
          If the slot it should occupy contains the matching key, it 
            returns (slot, contents).
          If the slot it should occupy is empty it returns (slot,None).
          If the slot it should occupy has other contents, it returns 
          (None,None). """ 
    # Compute the slot.
    slot = hashFunction( key )
    contents = self._array[ slot ]

    ## STUDENTS WILL COMPLETE THE REST OF THIS METHOD
    if contents == None:
      return slot, None
    
    if contents.key == key:
      return slot, contents
      
    return None, None
    
  def __iter__(self):
    """ Return: an iterator for the hashmap. """
    ## STUDENTS WILL COMPLETE THE REST OF THIS METHOD
    return _HashIterator(self._array)
  
  def printStats( self ):
    """Print the number of items in the table and the total
    number of collisions due to insertion."""
    print( 'Entry count : ', self._size )
    print( 'Collision count : ', self._collisions )

  def remove( self, key ):
    """ Removes the entry associated with the key.
        If the key is not in the map, does nothing. 
    Parameters: 
      key; hashable key value
    Return:
      None
    """
    ## STUDENTS WILL COMPLETE THE REST OF THIS METHOD
    
    slot1, content1 = self._lookup(self._hash1, key)
    slot2, content2 = self._lookup(self._hash2, key)
    s = 0

    
    if slot1 != None: # occupied by same key from hash1
      if content1 != None:
        s = 1 # For decreasing size
        self._array[slot1] = None
      
    if slot2 != None:  # occupied by same key from hash2, used if, instead of elif, because
      #any key, if present in the hash map, is associated with only 1 bucket
      if content2 != None:
        s = 2 # For decreasing size
        self._array[slot2] = None
    
    if s == 1 or s == 2:
      self._size -= 1

# Storage class for holding a key/value pair.   
class _MapEntry :                       

  """
  Storage class for holding a key/value pair. 
    Attributes:
      key; hashable key.
      value; 
  """
  def __init__( self, key, value ):
    """Create the entry with key and value """
    self.key = key
    self.value = value 
  
  def __eq__( self, other ):
    """Overload __eq__ so key, value pairs can be compared using '=='.
    Parameters:
      self; instance of the _MapEntry class.
      other; instance of the _MapEntry class."""
    if other == None:
      return False
    return ( self.key == other.key and self.value == other.value )

## STUDENTS WILL PUT AN ITERATOR CLASS HERE.

class _HashIterator:
  '''An Iterator class for iterating hash map.
  Attributes:
  _hashItems; list of non-empty elements in the Hash array
  _curItem; int; number of items
  '''
  def __init__(self, theArray):
    self._hashItems = [item for item in theArray if item is not None]
    self._curItem = 0 

  def __iter__(self):
    return self

  def __next__(self):
    if self._curItem < len(self._hashItems):
      item = self._hashItems[self._curItem].key #iterating only key
      self._curItem +=1
      return item
    else:
      raise StopIteration

