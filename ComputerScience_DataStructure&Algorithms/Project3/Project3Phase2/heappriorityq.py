'''Name: AKM Sadman (Abir) Mahmud
Date: Apr 24, 2023
Project no. 3
Phase no. 2'''

"""A minheap implementation of priority queue.
Attributes:
  - elements; list of PriorityNode initialized as empty.
  - count; int; number of nodes in queue."""

# import time

class PriorityQueue:
  def __init__( self ):
    self._elements = []
    self._count = 0

  def __str__(self):
    "prints out the data stored in the queue"
    s = ''
    for i in range(self._count):
      s += str(self._elements[i].data) + ' '
    
    return s[:-1]
    
  def __len__( self ):
    " returns the length of the queue list"
    return self._count

  def is_empty( self ):
    """Check to see if the queue is empty.
    Return: Boolean, True if empty, False otherwise."""
    return self._count == 0

  def enqueue( self, item, priority ): 
    """Insert the new PriorityNode with item and priority into the queue.
    Parameters:
      - self; PriorityQueue
      - item; data to be stored in the node
      - priority; int; priority of the node
    Return: None"""
    priority_node = PriorityNode( item, priority )
    self._elements.append (priority_node)
    self._sift_up( self._count )
    self._count += 1

  def _sift_up( self, ndx ):
    """shifts up the item at the index (ndx) so that the list has the feature 
    of minheap
    Parameters:
      - self; PriorityQueue
      - ndx; int; the index of the item
    Return: None"""
    
    if ndx > 0: # otherwise, the parent does not exist
      parent = (ndx - 1)  // 2
      if self._elements[ parent ].priority > self._elements[ ndx ].priority:
        tmp = self._elements[ndx]
        self._elements[ ndx ] = self._elements[ parent ]
        self._elements[ parent ] = tmp

        self._sift_up(parent)

      # To deal with events of same priority:
      # elif self._elements[ parent ].priority == self._elements[ ndx ].priority:
      #   if self._elements[ parent ].timestamp > self._elements[ ndx ].timestamp:
      #     tmp = self._elements[ndx]
      #     self._elements[ ndx ] = self._elements[ parent ]
      #     self._elements[ parent ] = tmp
  
      #     self._sift_up(parent)

  def dequeue( self ):  
    """Remove an item at the first index.
    Parameters: self; PriorityQueue.
    Return: data stored in front node."""
    
    if self._count>0:
  
      priority_node = self._elements[0]
      self._count -= 1
      self._elements[0] = self._elements[self._count]
      del self._elements[self._count] 
      
      self._sift_down(0)
      return priority_node.data

  def _sift_down( self, ndx ):
    """shifts down the item at the index (ndx) so that the list has the feature 
    of minheap
    Parameters:
      - self; PriorityQueue
      - ndx; int; the index of the item
    Return: None"""
    
    if ndx <= (self._count-2)//2: # checking if the item at index is parent (or leaf) 
      child_left = 2*ndx + 1
      child_right = 2*ndx + 2

      if child_left <= self._count-1:
        
        temp = self._elements[ndx] 
        min_ndx = ndx
  
        
        child_ndx = child_left
  
        if child_right <= self._count-1:
          
          if self._elements[child_left].priority > self._elements[child_right].priority:
            child_ndx = child_right

          # To deal with elements of same priority:
          # elif self._elements[child_left].priority == self._elements[child_right].priority:
          #   if self._elements[child_left].timestamp > self._elements[child_right].timestamp:
          #     child_ndx = child_right
  
      
        if self._elements[ndx].priority > self._elements[child_ndx].priority:
            min_ndx = child_ndx

        # To deal with elements of same priority:
        # elif self._elements[ndx].priority == self._elements[child_ndx].priority:
        #   if self._elements[ndx].timestamp > self._elements[child_ndx].timestamp:
        #     min_ndx = child_ndx
              
        if min_ndx != ndx:
          self._elements[ndx] = self._elements[min_ndx]
          self._elements[min_ndx] = temp
  
          self._sift_down(min_ndx)
          

  def peek( self ):
    """Examine the value of the first node.
    Parameter: self; PriorityQueue
    Return: data from front node."""

    if self._count > 0:
      return self._elements[0].data

  def __iter__(self):
    """ Return: an iterator for the PriorityQueue. """
   
    return _HeapIterator(self._elements)


class _HeapIterator:
  '''An Iterator class for iterating heap structured PriorityQueue.
  Attributes:
  _heapItems; list of non-empty elements in the heap list
  _curItem; int; number of items
  '''
  def __init__(self, theList):
    self._heapItems = [ item for item in theList ]
    self._curItem = 0 

  def __iter__(self):
    return self

  def __next__(self):
    if self._curItem < len(self._heapItems):
      item = self._heapItems[self._curItem].priority #iterating only priority
      self._curItem +=1
      return item
    else:
      raise StopIteration


class PriorityNode:
  """The node for a priority queue.
  Attributes:
    - data; information to store
    - priority; int; to order nodes. lower values are higher priority."""
  
  def __init__(self, item, priority):
    self.data = item 
    self.priority = priority
    # self.timestamp = time.monotonic() 


