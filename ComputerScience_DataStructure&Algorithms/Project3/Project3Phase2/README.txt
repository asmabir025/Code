Name: AKM Sadman (Abir) Mahmud
Date: Apr 24, 2023
Porject Number: 03
Phase: 02

Results:

#1

====== Simulation Statistics =======
number of cashier :  2
totalSimTime:  1000
max interarrival time:  2
cashier max service time:  5

Number of customers served =  654
Number of customers remaining in line =  7
The average wait time was 4.07 minutes.
=====================================


Simulation ends ...


#2

Simulation starts ...
====== Simulation Statistics =======
number of cashier :  5
totalSimTime:  3000
max interarrival time:  2
cashier max service time:  14

Number of customers served =  1991
Number of customers remaining in line =  0
The average wait time was 13.23 minutes.
=====================================


Simulation ends ...

Program Details:

Creating heap priority queue was challenging for me. Especially, I was confused about how to deal with same priority events. But the bigger concern was the implementation of sift_down method. My initial implementation was producing negative wait time. Further, when I set self.debug = True, i.e., when I was looking into how the events were getting dequeued, I noticed that some events were not being dequeued according to their time priority. This resulted in only one cashier serving for a long time and another cashier(s) being 'busy' because the departure event was not processed. I decided to print out the events queue every time (that is why I created HeapIterator class and the corresponding method in PriorityQueue to use that class) before dequeuing and see which events were being dequeued. To my surprise, I found out that in a simple case of three events, the last event was always being dequeued. So, I checked the dequeue method and found out that I am always ignoring one comparison of parent-children pair. When I fixed that, my code prooduced 'list out of index' error. That is because, for example of the three events case, the right child does not even exist. So, then, I had to ensure that both children exist first before their supposed indices were being compared to the parent index. Thus, I solved the major issue of my code. 

In a sense, I still use priority node but I do not utilize the next attribute. I could use a tuple to store the priority. Also, for same priority events (for example, new arrival and departure happening at the same time), it does not matter what I am dequeuing first. If I dequeue the arrival event first, if no cashier is free, it is being sent to the waiting queue. However, the departure event, happening at the same time, is being dequeued right after, and the cashier will be available for the new customer. As a result, the customer will be waiting for 0 min in the waiting queue; so, the total waiting time does not get affected. So, at the beginning, I created the timestamps attribute to store the real time an operation of dequeuing and use that time as a second prioirty. However, it seemed unnecessary according to my given reasoning above, so I commented this part of the code out. 