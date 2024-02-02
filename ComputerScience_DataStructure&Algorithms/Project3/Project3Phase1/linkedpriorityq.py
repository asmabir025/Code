'''Name: AKM Sadman (Abir) Mahmud
Date: Apr 08, 2023'''

"""A linked list implementation of priority queue.
Attributes:
  - head; PriorityNode or None if queue empty.
  - tail; PriorityNode or None if queue empty.
  - count; int; number of nodes in queue."""
class PriorityQueue:
  def __init__( self ):
    """Constructor to create an empty queue."""
    self.head = None
    self.tail = None
    self.count = 0

  def __str__(self):
    "prints out the data stored in the queue"

    s = ''
    cur = self.head 
    while cur != None:
      s += str(cur.data) + ' '
      cur = cur.next
    
    return s[:-1]
      
  def is_empty( self ):
    """Check to see if the queue is empty.
    Return: Boolean, True if empty, False otherwise."""
    return self.count == 0

  def __len__( self ):
    """Return the length (int) of the queue."""
    return self.count

  def enqueue( self, item, priority ):
    """Insert the new node with item and priority into the queue.
    Parameters:
      - self; PriorityQueue
      - item; data to be stored in the node
      - priority; int; priority of the node
    Return: None"""
    
    new_node = PriorityNode(item, priority)
    
    if self.head == None: # adding to an empty queue 
      self.head = new_node
      self.tail = new_node

    else:
      cur_node = self.head
      prev = None
      while cur_node != None: # searching though the entire queue 

        if cur_node.priority > new_node.priority: # adding before cur_node
          new_node.next = cur_node

          if cur_node == self.head:
            self.head = new_node 
          else:
            prev.next = new_node

          break

        elif cur_node.priority == new_node.priority: #adding after cur_node
          
          next_cur = cur_node.next
          cur_node.next = new_node

          if next_cur == None:
            self.tail = new_node
          else:
            new_node.next = next_cur

          break

        else: # updating prev and cur_node and continuing the loop
          prev = cur_node
          cur_node = cur_node.next
        

      if cur_node == None: # if the node being added has the highest priority 
        prev.next = new_node
        self.tail = new_node

    self.count += 1 # adding to count due enqueueing
    
  def dequeue( self ):
    """Remove an item at the front.
    Parameters: self; PriorityQueue.
    Return: data stored in front node."""

    if self.head != None:
      
      first_node = self.head
      self.head = first_node.next
      self.count -= 1
      return first_node.data
 
  def peek( self ):
    """Examine the value of the first node.
    Parameter: self; PriorityQueue
    Return: data from front node."""

    if self.count > 0:
      return self.head.data
        
class PriorityNode:
  """The node for a priority queue.
  Attributes:
    - data; information to store
    - priority; int; to order nodes. lower values are higher priority.
    - next; PriorityNode of next node or None if at end of queue."""
  
  def __init__( self, data, priority ):
    
    self.data = data
    self.priority = priority
    self.next = None

  # def __eq__( self, other ):
  #   '''Compares a priority node with the other 
  #   returns True if both nodes have same data and priority, otherwise False
  #   Parameter:
  #   self; an instance of PriorityNode
  #   other; an instance of PriorityNode'''

  #   return self.data == other.data and self.priority == other.priority