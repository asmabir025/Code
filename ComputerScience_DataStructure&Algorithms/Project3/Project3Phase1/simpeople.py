"""Name: AKM Sadman (Abir) Mahmud
Date: Apr 08, 2023
This file contains the class StorePeople and child classes, Cashier and 
Customer, used to store and manage information related to cashiers and 
customers, respectively."""

class StorePeople:
  """A generic class of store people from which customers and cashiers
  inherit properties.
  Attributes:
    id; int. unique identifier."""
  def __init__(self, id):
    self.id = id

class Cashier(StorePeople):
  """A subclass of super-class, StorePeople. 
   Attributes:
   Derived from StorePeople:
   - id; int, a unique identifier 
   or available)
   Additional:
   - type; int (class constants), status of cashier (whether serving,
   - served_customers; list, list of served customers
   - avg_service; int, average time of service per customer
  """
  # These constants define the availability of Cashier
  
  BUSY = 0
  AVAILABLE = 1

  def __init__(self, id, type = AVAILABLE):
    super().__init__(id)
    self.type = type
    self.served_customers = []
    self.avg_service = 0

  def __str__( self ):
    '''Returns details of Cashier for print(Cashier)
    Paramters:
    - self; an instance of Cashier class
    '''
    s = ''
    s += f'Cashier {str(self.id)}' + '\n'
    s += f'Average Service time: {self.avg_service} \n' 
    s += f'Total Served Customers: { len(self.served_customers) } '
    
    return s


class Customer(StorePeople):
  """A subclass of super-class, StorePeople. 
   Attributes:
   Derived from Storepeople:
   - id; int, a unique identifier
   Additional:
   - cashierID; int, which cashier Customer was served by
   - arrival_time; int, when Customer arrived 
   - service_start; int, when Customer started receiving service
   - service_end; int, when Customer departed the store
  """
  # These constants define the state of Customer
  
  def __init__(self, id):
    
    super().__init__(id)
    self.cashierID = -1
    self.arrival_time = 0 # initiated as 0 for the first customer 
    self.service_start = -1
    self.service_end = -1
    
  def wait_time(self):
    '''Returns wait time for Customer
    Parameters:
    - self; an instance of Customer class
    '''
    return self.service_start - self.arrival_time 

  def service_time(self):
    '''Returns service time for Customer
    Parameters:
    - self; an instance of Customer class
    '''
    return self.service_end - self.service_start
    
  def __str__(self):
    '''Returns details of Customer for print(Customer)
    Paramters:
    - self; an instance of Customer class
    '''
    s = ''
    s+= f'Customer {self.id}' + '\n'
    s += f'Served by Cashier {self.cashierID}' + '\n'
    s+= f'Arrival Time = {self.arrival_time}' + '\n'
    s+= f'Departure Time = {self.service_end}' + '\n'
    s+= f'Wait Time = {self.wait_time()}' + '\n'
    s+= f'Service Time = {self.service_time()} ' + '\n'
    
    return s 
  