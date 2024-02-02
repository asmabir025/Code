'''Name: AKM Sadman (Abir) Mahmud
Date: Mar 25, 2023'''
'''I copied this from the lecture material'''

class Node:
  '''
  A simple Node class
  '''
  def __init__(self, data = None, next = None):
    self.data = data
    self.next = next

class Stack:
  ''' 
  A class of linked lists based stack

  Data attributes
  - self.top = a reference to the first Node
          in the list
  '''

  def __init__(self, init_list = None):
    '''
    Given a python list, creates a linked list. 
    Initializes self.head as the start of the list

    Input:
    - initList: a python list
    '''
    self.top = None

    if init_list is not None:
      self._create_list(init_list)

  def _create_list(self, init_list):
    ''' Given a python list, create a linked list'''
    for index in range(len(init_list)-1, -1, -1):
      self.top = Node(init_list[index], self.top)

  def __str__(self):
    linked_list = ""
    curr_node = self.top
    while curr_node:
      linked_list += str(curr_node.data) + " -> " 
      curr_node = curr_node.next
    linked_list_finished = linked_list[:-4]
    return linked_list_finished

  # ----------- BASIC OPERATIONS --------------#

  def push(self, data):
    """ Push a itemm to the top of the stack.
    Parameters:
      - data; information to store in stack
    Return: None"""
    
    '''
    1. make a node with data
    2. assign node.next to be the top
    3. move the stack top to be the new node
    '''
    self.top = Node(data, self.top)

  def pop(self):
    """ Pop the top item (data) from the stack.
    Parameters: self
    Return: data from the node on the top of the stack"""
    '''
    1. if the stack is empty, print an error messge, return
    2. otherwise, retrive the data from the stack top
    3. move the stack top to the next node
    4. return the data
    '''
    curr_node = self.top
    if curr_node == None:
      self.peek() 

    else:
      pop_data = self.top.data
      self.top = curr_node.next
      return pop_data
    
    
  
  def is_empty(self):
    """ Return True if stack is empty, False otherwise"""
    return self.top == None

  def peek(self):
    """ Return the data at the top of the stack.
    Parameters: self
    Return: data from the top of the stack."""
    assert not self.is_empty(), "Cannot peek at an empty stack"
    return self.top.data

