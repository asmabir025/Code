Name: AKM Sadman (Abir) Mahmud
Date: Apr 08, 2023
Porject Number: 03
Phase: 01

Results:

#1

Simulation starts ...
====== Simulation Statistics =======
number of cashier :  2
totalSimTime:  1000
max interarrival time:  2
cashier max service time:  5

Number of customers served =  652
Number of customers remaining in line =  22
The average wait time was 6.68 minutes.
=====================================

Simulation ends ...


#2

Simulation starts ...
====== Simulation Statistics =======
number of cashier :  5
totalSimTime:  3000
max interarrival time:  2
cashier max service time:  14

Number of customers served =  1993
Number of customers remaining in line =  20
The average wait time was 17.21 minutes.
=====================================

Simulation ends ...

Program Details:

PriorityQueue coding was like a weekly lab practice. However, I realized, not immediately, that I have to consider three cases of priority, including one with equal priority of a new element with one already in the queue. Other than this, it was fine. I also made an eq function for PriorityQueue, but ended up commenting it out. That is because I also had to create isinstance function for that; otherwise, nonetype object messes up with the eq function. As I did not need this usage much for the purposes of PriorityQueue, I ended up commenting the eq function out. 

For run method in StoreSimulation, I created separate methods for arrival, departure, and begin customer service processed and incorporated them in the run. I also made another method, current_customer_service, which is being used in all the three processes/methods. This helped me debug my codes much easily. 

For Cashier and Customer methods, I added extra attributes besides id and type (for Cashier). For Customer, I am tracking arrival time, service start, and service end times. As a result, I am able to print out wait time and service time through wait_time and service_time methods. For Cashier, I am tracking average service time and the list of served customers' id. 

For creativity part, I explored wait time aspect of the simulation. I realized average wait time is not within a small range if the same simulation is run multiple times. So, I wanted to see the distribution of individual customer wait times and ended up plotting histogram. However, the distribution extremely varries too. So, I wanted to check the distribution of individual customer service times and ended up plotting histogram. But this distribution is uniform distribution and does not varry significantly. That might be related to how random.randint works. Anywway, for helping make these distributions, I also created service_list and wait_list methods. I have also created update_cashiers method to update the average time of service for cashiers. 