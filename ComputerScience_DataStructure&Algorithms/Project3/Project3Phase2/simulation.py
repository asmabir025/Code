'''Name: AKM Sadman (Abir) Mahmud
Date: Apr 24, 2023
Project no. 3
Phase no. 2
This file contains classes: StoreSimulation and Event, which are necessary to
implement the event driven simulalation of interactions between customers and
cashiers. Three methods of StoreSimulation: arrival, begin_service, and depar-
ture: are crucial for the most import method of the class: run.

'''

"""Implementation of the main simulation class."""
from array204 import Array
from pylistqueue import Queue
from heappriorityq import PriorityQueue
from simpeople import Customer, Cashier

##
## Phase 1
##
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy


# Optionally, tweak styles.
mpl.rc('figure',  figsize=(5, 3))
mpl.rc('image', cmap='gray')

"""This class simulates the operation of a store where a number
of cashiers to serve customers arriving  in a random 
fashion. The customers arrive at a common waiting queue for a group
of cashiers. If a cashier is free, the newly arrived customer receives 
the service immediately. If no cashier is availabe, the customer
waits in the queue for the next available cashier. When a customer
finishes the service at the cashier, with a probability, the customer
goes to the gift-wraper to receive a further service.

Phase 1 will utilize array-based priority queues.

Phase 2 of the project will use heaps-based priority queues."""

class StoreSimulation :
  ''' Class Attributes:
    - debug; boolean; use to create debug print statements
    - max_service_time; int; maximum time to serve one customer
    - total_sim_time; int; number of steps in simulation
    - max_interarrival_time; int; maximum time between customer arrivals
    - waiting; Queue of Customers; for customers who are waiting for cashier
    - cashiers; Array of Cashiers; 
    - customers; list of customers;
    - events; PriorityQueue of Events, prioritized by time
    - clock; int; current time in the simulation
    - total_wait_time; int; sum of time each customer waited between arriving
    and beginning service
    - customer_count; total number of customers that have arrived
    - service_gen; random number generator for service times
    - arrival_gen; random number generator for arrival times
  '''
  def __init__( self, num_cashier, serv_time, \
                interarrival_time, total_sim_time ):
    """Create a simulation object.
    Parameters:
      num_cashier; int
      serv_time; int; maximum service time for a single customer
      interarrival_time; int; maximum time between two customer arrivals
      total_sim_time; int
    """
    # Debug state or not
    self.debug = True
    # self.debug = False

    # Time parameters supplied by the user.
    self.max_service_time = serv_time
    self.total_sim_time = total_sim_time
    self.max_interarrival_time = interarrival_time
     
    # Simulation components.
    self.waiting = Queue()   
    self.cashiers = Array( num_cashier )    
    for i in range( num_cashier ):
      self.cashiers[ i ] = Cashier( i)   # 'i' is the ID

    self.customers = []
    self.events = PriorityQueue()
    self.clock = 0
    
    # Computed during the simulation.
    self.total_wait_time = 0
    self.customer_count = 0

    # Random number generator, we define two separate random generators
    # so they don't share states.
    self.service_gen = random.Random()
    self.arrival_gen = random.Random()

  def run( self ):
    '''Runs the simulation until it finishes all customer services that it 
    started within total_sim_time
    Parameters:
    - self; an instance of StoreSimulation class
    '''
    # Initializing customer arrival, therefore, simulation
    first_event = Event( Event.ARRIVAL, Customer(0), 0 )
    self.events.enqueue( first_event, 0 )
    
    # event-driven, not time-step simulation
    while not self.events.is_empty():

      if self.debug: # extra code to track the error in my code
        print( 'Before Dequeue:' )
        queue_list = []
        for events in self.events:
          queue_list.append(events)
        print(queue_list) 

        
      current_event = self.events.dequeue() 
      current_customer = current_event.who
      self.clock = current_event.time 
      
      if current_event.time < self.total_sim_time:
  
        # In case of a Customer leaving
        if current_event.type == Event.DEPARTURE: 
          self.departure( current_customer )
          self.begin_service( current_customer )
          
        # In case of a Customer arriving
        else: 
          self.arrival( current_customer )

      elif current_event.type == Event.DEPARTURE: 
        # for the events that before total_sim_time
        self.departure( current_customer )
        

  def print_results( self ):
    """Print the simulation results."""
    print( '====== Simulation Statistics =======' )
    print( 'number of cashier : ', len( self.cashiers ) )

    print( 'totalSimTime: ', self.total_sim_time )
    print( 'max interarrival time: ', self.max_interarrival_time )
    print( 'cashier max service time: ', self.max_service_time )

    num_served = self.customer_count - len( self.waiting )
    avg_wait = round(float( self.total_wait_time ) / num_served, 2)
    print( "" )
    print( "Number of customers served = ", num_served )
    print( "Number of customers remaining in line = ",
           len(self.waiting) )
    print( "The average wait time was", avg_wait, "minutes.") 
    print( '=====================================' )


  def arrival( self, cur_customer ):
    '''Processes arrival of a new customer and possibly her service 
    Parameters:
    - self; an instance of StoreSimulation class
    - cur_customer; an instance of Customer class
    '''
    self.customer_count += 1 # updating customer count
    if self.debug:
      print(f'Time {self.clock} : Customer {cur_customer.id}  arrives')

    # new customer arrival is calculated next
    new_customer = Customer( cur_customer.id + 1 )
    new_customer.arrival_time = self.clock + self.arrival_gen.randint(1, self.max_interarrival_time)
    next_arrival = Event( Event.ARRIVAL, new_customer, new_customer.arrival_time )
    

    # checking if cur_customer can start her service
    id_cashier = -1
    for i in range( len(self.cashiers) ):
      cashier_no = self.cashiers[i]
      if cashier_no.type == Cashier.AVAILABLE:
        id_cashier = i
        break
        
    if id_cashier != -1:
      cashier = self.cashiers[ id_cashier ] 
      cashier.type = Cashier.BUSY
      self.current_customer_service( cur_customer, cashier )
      if self.debug:
        print(f'Time {self.clock} : Customer {cur_customer.id} service starts by Cashier {cashier.id}')

    else:
      self.waiting.enqueue(cur_customer)

    self.events.enqueue( next_arrival, new_customer.arrival_time )
    
  def begin_service( self, cur_customer ):
    '''Processes the beginning of service for someone in the waiting queue
    Parameters:
    - self; an instance of StoreSimulation class
    - cur_customer; an instance of Customer class
    '''
    serving_cashier = self.cashiers[ cur_customer.cashierID ]
    if self.waiting.is_empty():
      serving_cashier.type = Cashier.AVAILABLE # changing cashier status
    else:
      next_customer = self.waiting.dequeue()
      self.current_customer_service( next_customer, serving_cashier )
      self.total_wait_time += next_customer.service_start - next_customer.arrival_time
      
      if self.debug:
        print('Wait-time increased to',self.total_wait_time)
        print(f'Time {self.clock} : Customer {next_customer.id}  service starts by Cashier {serving_cashier.id}')

  
  def departure( self, cur_customer ):
    '''Processes the departure of a customer
    Parameters:
    - self; an instance of StoreSimulation class
    - cur_customer; an instance of Customer class
    '''
    serving_cashier = self.cashiers[ cur_customer.cashierID ]
    cur_customer.service_end = self.clock
    serving_cashier.served_customers += [ cur_customer.id ] 
    # updating customer list (sorted by departure time)  
    self.customers.append( cur_customer )

    if self.debug:
      print(f'Time {self.clock} : Cashier {serving_cashier.id} stopped serving Customer {cur_customer.id}')
  
    
  def current_customer_service( self, cur_customer, serving_cashier ):
    '''Processes the beginning of a customer's service (this method is called in
    arrival and begin_service methods)
    Parameters:
    - self; an instance of StoreSimulation class
    - cur_customer; an instance of Customer class
    '''
    cur_customer.cashierID = serving_cashier.id
    
    cur_customer.service_start = self.clock
    cur_service_time = self.service_gen.randint(1, self.max_service_time)
    next_departure_time = self.clock + cur_service_time 
    next_departure = Event( Event.DEPARTURE, cur_customer, next_departure_time )
    self.events.enqueue( next_departure, next_departure_time )

  
  def update_cashiers( self ):
    '''Updates the cashiers' average service time after simulation
    Parameters:
    - self; an instance of StoreSimulation class
    '''
    sorted_customers = deepcopy( self.customers )
    sorted_customers.sort( key = lambda ele : ele.id  ) # sorting by customers' id

    for cashier in self.cashiers:
      avg_service = 0
      for customerID in cashier.served_customers:
        avg_service += sorted_customers[customerID].service_time()
      if len( cashier.served_customers ) > 0:
        avg_service /= len( cashier.served_customers  ) 
        cashier.avg_service = round(avg_service, 2)
  
  
  def wait_list( self ):
    '''returns a list of wait times of all customers
    Parameters:
    - self; an instance of StoreSimulation class
    '''
    wait_t_list = []
    for customer in self.customers:
      wait_t_list.append( customer.wait_time() )

    return wait_t_list

  
  def service_list( self ):
    '''returns a list of service times of all customers
    Parameters:
    - self; an instance of StoreSimulation class
    '''
    service_t_list = []
    for customer in self.customers:
      service_t_list.append( customer.service_time() )

    return service_t_list

      
  def hist_quantity( self, n, quantity ):
    '''makes and saves a histogram of the given quantity (wait time or 
    service time)
    Parameters:
    - self; an instance of StoreSimulation class
    - n; int, number of bins
    - quantity; str
    '''
    if quantity == 'wait time':
      quantity_list = self.wait_list()

    else:
      quantity_list = self.service_list()
      
    fig, axs = plt.subplots(1, 1,
                        figsize =(5, 3),
                        tight_layout = True)
    axs.set_xlabel( quantity )
    axs.set_ylabel( 'Number of Customers' )
    axs.set_title( 'Histogram of ' + quantity )
 
    axs.hist( quantity_list, n )
 
    # Show plot
    plt.savefig(quantity + '_histogram')
      
    
class Event:
  # These constants define the types of events
  ARRIVAL     = 0    # customer arrival
  DEPARTURE   = 1    # cashier ending
  
  '''Class attributes:
    time; int; the time step the event occurs
    type; int constant (see below); identifies the event
    who; Customer; customer the event serves
  '''
  def __init__(self, type, person, tme):
    '''Parameters:
      tme; int; the time step the event occurs
      type; int constant (see below); identifies the event
      person; StorePerson; 
    '''
    self.time = tme
    self.type = type
    self.who = person

