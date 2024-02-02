'''CSCI 204 Lab 02 Class Design, Implementation, and Exceptions
Lab section: CSCI 204.L61, Thursday 10-11:52
Student name(s): AKM Sadman (Abir) Mahmud
Instructor name: Professor Edward Talmage'''


import csv

class ContactTracer():
  '''
  Use adjacency-list based Graph to trace contagion contacts.

  Attributes:
    contacts: dictonary; keys: Vertex object, value: list of Vertex objects that are the outgoing edges from key.
    edge_weights: dictionary; keys: tuple of Vertex objects (from_vertex, to_vertex); value: float, weight of the edge between those verticies
  '''
  def __init__(self, filename):
    
    self.contacts = {}  # this is a dictionary representing the graph
    self.edge_weights = {} # an extra dictionary to store the edges as key and corresponding weights as value
    self.read_file(filename)
    
    
  def read_file(self,filename):
    with open(filename, mode='r') as file:
      csvFile = csv.reader(file)
      # each line in the csv is formatted
      # name, contact1, contact2, ...
      for line in csvFile:
        # remainder of values is names of all contacts
        for contact_label in line[1:]:
          #contact = self.get_vertex(contact_label)
          self.add_edge(line[0],contact_label)

  def get_vertex(self, vertex_label):
    '''
      Return the Vertex object associated with a label. If the label has no vertex, create one.

    Attributes:
      self; Graph.
      vertex_label; string; uniquie ID/label of vertex.
    Return:
      Vertex object with the label.
    '''
    for vertex in self.contacts.keys():
      if vertex.label == vertex_label:
        return vertex
    new_vtx = Vertex(vertex_label)
    self.contacts[new_vtx] = []
    return new_vtx

  def get_edge_weight(self, from_vtx, to_vtx):
    '''
    Return the edge weight between two verticies
    Parameters:
      from_vtx; Vertex.
      to_vtx; Vertex.
    '''
    return self.edge_weights[(from_vtx,to_vtx)]
    
  def add_edge(self, from_vtx_label, to_vtx_label, weight=1):
    '''
    Add edge of given weight (default 1) between to verticies.
    Parameters:
      from_vtx_label; string.
      to_vtx_label; string.
    Return: None
    '''
    # get or create Vertex objects for each label
    fm_vtx = self.get_vertex(from_vtx_label) 
    to_vtx = self.get_vertex(to_vtx_label)

    # add to_vertex to from vertex's adjacency list
    self.contacts[fm_vtx].append(to_vtx)
    # set weight of that edge
    self.edge_weights[(fm_vtx, to_vtx)] = weight
    
  def __str__(self):
    '''
    Return string representing all verticies and their adjacent verticies.
    Paramseters: self.
    Return: string.
    '''
    s = 'Contact Records:\n'
    sorted_names = sorted(self.contacts, key = lambda x: x.label ) 
    # sorting the dictionary by vertex labels
    
    for sick_contact in sorted_names:
      num_contact = len( self.contacts[ sick_contact ] )
      
      if num_contact > 0: # for 0 case, nothing will be printed
        s += sick_contact.label + ' had contact with '
        contact_list = [ self.contacts[ sick_contact ][i].label for i in \
                        range( num_contact ) ]
        contact_list.sort()
        
        for contacts in contact_list[:-1]:
          s += contacts + ', '
  
        if len(contact_list) > 1:
          s = s[:-2]
          s += ' and ' + contact_list[-1] + '\n'
        elif len( contact_list ) == 1:
          s += contact_list[0] + '\n'
      
    return s

  def get_potential_zombies(self):
    '''
    Return a list of Vertex objects of all potential zombies, i.e., verticies with no outgoing edges.
    Paramteres: self.
    Return: listof Vertex objects.
    '''
    
    zombies = [ vertex_contact for vertex_contact in \
              self.contacts.keys() if len(self.contacts[vertex_contact])==0]

    return zombies
  
  def get_patient_zeros(self):
    '''
    Returns list of Vertex objects of the patient zeros, i.e., verticies with no incoming edges.
    
    Paramters: self.
    Return: list of Vertex objects.
    '''

    vlist = list(self.contacts.keys())
    
    for vertex, adj_list in self.contacts.items():
      for adj_vertex in adj_list:
        if adj_vertex in vlist:  # adj_vertex could have been removed earlier
          vlist.remove(adj_vertex)  
          
    return vlist

    
  def shortest_path(self, start_vertex):
    '''
      Runs your favorite shortest path algorithm to find the shortest path from start_vertex_label to all
      other connected verticies.
      Prints all shortest distances.
    Parameters:
      start_vertex_label: label of vertex in graph. if label does not exist, does nothing. 
    Return: None
    '''
    if start_vertex not in self.contacts:
      return
    # Set up
    # set all verticies initial distance to infinity
    # and initial pred_vertex to None
    for vertex in self.contacts:
      vertex.distance = float('inf')
      vertex.prev_vertex = None
    # create deep copy of all verticies to start them in unvisited list
    unvisited = list(self.contacts.keys())[:]

    # start_vertex has a distance of 0 from itself
    start_vertex.distance = 0

    # One vertex is removed with each iteration; repeat until the list is
    # empty.
    while len(unvisited) > 0:
        
        # Visit vertex with minimum distance from start_vertex
        smallest_index = 0
        for i in range(1, len(unvisited)):
            if unvisited[i].distance < unvisited[smallest_index].distance:
                smallest_index = i
        current_vertex = unvisited.pop(smallest_index)

        # Check potential path lengths from the current vertex to all neighbors.
        for adj_vertex in self.contacts[current_vertex]:
            edge_weight = self.edge_weights[(current_vertex, adj_vertex)]
            alternative_path_distance = current_vertex.distance + edge_weight
                  
            # If shorter path from start_vertex to adj_vertex is found,
            # update adj_vertex's distance and predecessor
            if alternative_path_distance < adj_vertex.distance:
                adj_vertex.distance = alternative_path_distance
                adj_vertex.prev_vertex = current_vertex
  
  def print_shortest_path(self, start_vertex, end_vertex):
    '''
    Print shortest path and length of shortest path between twoVerticies.
    Parameters:
      start_vertex:Vertex.
      end_vertex:Vertes.
    Return:
    None
    '''
    self.shortest_path(start_vertex)
    path=""
    current_vertex = end_vertex
    # end vertex stores distance
    dist = current_vertex.distance

    while current_vertex != start_vertex:
      path = " -> " + str(current_vertex.label) + path
      current_vertex = current_vertex.prev_vertex
    path = start_vertex.label + path + ". Dist: "+ str(dist) +'\n'
    print(path)
    


class Vertex:
  def __init__(self, label):
    self.label = label
    self.distance = float('inf')
    self.prev_vertex = None

  def __str__(self):
    return self.label
