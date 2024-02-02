from simulation import *   # Phase 1
from linkedpriorityq import *


def simulation():
  '''I ran the same tests over and over again to get a 'feel' for the 
  'correct' value of customers and waiting time.
  
  '''
 

  # inputs of parameters
  # print('Give values for each parameter of simulation')
  # cashiers = int( input('Number of Cashiers: ') )
  # total_time = int( input('Total Simulation Time: ') )
  # interarrival_time = int( input('Maximum time between arrivals of two customers: ') )
  # cash_time = int(  input(  'Maximum service time for a customer: ' ) )
  
  cashiers = 5
  total_time = 3000
  interarrival_time = 2
  cash_time = 14

  # cashiers = 2
  # total_time = 1000
  # interarrival_time = 2
  # cash_time = 5

  # cashiers = 2
  # total_time = 50
  # interarrival_time = 1
  # cash_time = 2
  
  cashiers = 1
  total_time = 5
  interarrival_time = 1
  cash_time = 2
  
  
  store_sim = StoreSimulation( cashiers, cash_time, \
                              interarrival_time, total_time)
  print('\n')
  print( 'Simulation starts ...' )
  store_sim.run()
  store_sim.print_results()
  print('\n')

  # Making histograms of wait-time and service time
  store_sim.hist_quantity( 100, 'wait time' )
  store_sim.hist_quantity( cash_time, 'service time' )
  
  print( 'Simulation ends ...' )
  print('Check the saved histograms of wait time and service time!')
  print('\n')

  # printing information about cashiers
  store_sim.update_cashiers()
  if len(store_sim.cashiers) < 10:
    print('Details of Cashiers:')
    for cashier in store_sim.cashiers:
      print(cashier)
      print('\n')
  else:
    print('Details of 5 Cashiers:')
    for cashier in store_sim.cashiers[:5]:
      print(cashier)
      print('\n')

  # printing information of customers
  if len(store_sim.customers) > 5:
    print('Details of first 5 customers (sequentially as departure time)')
    for customer in store_sim.customers[:10]:
      print(customer)
      print('\n')
  else:
    print('Details of customers (sequentially as departure time)')
    for customer in store_sim.customers:
      print(customer)
      print('\n')

def test_priority_queue():
  
  pq = PriorityQueue()
  print(pq)
  print(pq.is_empty())
  print(pq.peek())
  
  pq.enqueue(5, 3)
  print(pq)
  print(pq.is_empty())
  print(pq.peek())
  
  pq.enqueue(3, 2)
  print(pq)
  print(pq.is_empty())
  print(pq.peek())

  pq.enqueue(10, 1)
  print(pq)
  print(pq.peek())

  pq.enqueue(2, 2)
  print(pq)
  print(pq.peek())

  pq.dequeue()
  print(pq)
  print(pq.peek())

  pq.enqueue(2, 5)
  print(pq)
  print(pq.peek())

  pq.enqueue(2, 5)
  print(pq)
  print(pq.peek())

  pq.enqueue(1, 4)
  print(pq)
  print(pq.peek())

# add any additional incremental testing

####### COMMENT OUT WHICH METHOD YOU'D LIKE TO RUN ####
simulation()
#test_priority_queue()