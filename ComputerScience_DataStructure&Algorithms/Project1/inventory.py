# Put all your classes for Phase 1 in here

import csv

class Item:
  '''
  Base Class. 
  A class saving information about an item.

  Attributes:
  _id; integer
  _name;  string
  _unit_price; string
  _date; string 
  _manufacturer; string
  _quantity; string
  '''
  def __init__(self, id, name, price, date, manufac, count):
    self.id = id
    self.name = name
    self.unit_price = price
    self.date = date
    self.manufacturer = manufac
    self.quantity = count 

  def __str__(self):
    '''
    Parameter:
    _self; an instance of item class
    returns string
    
    '''
    s = '----------'+'\n'
    s += 'ID: ' + str(self.id) + '\n'
    s += 'Name: ' + self.name + '\n'
    s += 'Manufacturer: ' + self.manufacturer + '\n'
    s += 'Date: ' + self.date + '\n'
    s += 'Unit Price: $' + self.unit_price + '\n'
    s += 'Quantity: ' + self.quantity + '\n'

    return s

    
class Book(Item):
  '''
  Derived Class: Book, Base class: Item
  A class capturing information about a book
  
  Attributes:
  Inherited:
  _id; integer
  _name;  string
  _unit_price; string
  _date; string 
  _manufacturer; string
  _quantity; string
  
  Additional:
  _publisher; string
  _isbn; string
  '''
  def __init__(self, id, title, date, publish, auth, price, isbn, count):
    super().__init__(id, title, price, date, auth, count)
    self.publisher = publish
    self.isbn = isbn

  def __str__(self):
    '''
    Parameters:
    _self; an instance of Book class
    Returns string 
    '''
    s = super().__str__()

    s = s.replace('Manufacturer', 'Author')  
    s = s.replace('Name','Title')
    
    s += 'Publisher: ' + self.publisher + '\n'
    s += 'ISBN: ' + self.isbn + '\n'

    return s


class Cd(Book):
  '''
  Derived Class: Cd, Base class: Book
  A class capturing information about a Cd
  
  Attributes:
  Inherited:
  _id; integer
  _name;  string
  _unit_price; string
  _date; string 
  _manufacturer; string
  _quantity; string
  _publisher; string
  _isbn; string
  Additional:
  None
  '''

  def __init__(self, id, title, artist, label, asin, date, price, count):
    super().__init__(id, title, date, label, artist, price, asin, count)
    
  def __str__(self):
    '''
    Parameters:
    _self; an instance of Cd class
    Returns string 
    '''
    s = super().__str__()
    s = s.replace('Author', 'Artist(s)')
    s = s.replace('Publisher','Label')
    s = s.replace('ISBN', 'ASIN')

    return s

    
class Collectible(Item):
    '''
    Derived Class: Collectible, Base class: Item
    A class capturing information about a Collectible 
    
    Attributes:
    Inherited:
    _id; integer
    _name;  string
    _price; string
    _date; string 
    _manufacturer; string
    _quantity; string
    Additional:
    None
    '''
    def __str__(self):
      '''
      Parameters:
      _self; an instance of Book class
      Returns string 
      '''
      s = super().__str__()
      s = s.replace('Manufacturer', 'Owner') 
      # for upper code: online help: https://www.geeksforgeeks.org/python-string-replace/
      
      return s

class Electronics(Item):
  '''
    Derived Class: Electronics, Base class: Item
    A class capturing information about an Electronics 
    
    Attributes:
    Inherited:
    _id; integer
    _name;  string
    _price; string
    _date; string 
    _manufacturer; string
    _quantity; string
    Additional:
    None
  '''
  def __str__(self):
    '''
      Parameters:
      _self; an instance of Electronics class
      Returns string 
    '''
    s = super().__str__()
    return s

  
class Fashion(Item):
  '''
  Derived Class: Fashion, Base class: Item
    A class capturing information about a Fashion (product)
    
    Attributes:
    Inherited:
    _id; integer
    _name;  string
    _price; string
    _date; string 
    _manufacturer; string
    _quantity; string
    Additional:
    None
  '''
  def __str__(self):
    '''
      Parameters:
      _self; an instance of Fashion class
      Returns string 
    '''
    s = super().__str__()
    return s

    
class Homegarden(Item):
  '''
    Derived Class: Homegarden, Base class: Item
    A class capturing information about a Homegarden (product)
    
    Attributes:
    Inherited:
    _id; integer
    _name;  string
    _price; string
    _date; string 
    _manufacturer; string
    _quantity; string
    Additional:
    None
  '''
  def __str__(self):
    '''
      Parameters:
      _self; an instance of Homegarden class
      Returns string 
    '''
    s = super().__str__()
    return s


def read_items(filename, classtype):
  '''
  Reads in csv files, captures each line as an object of given classtype
  
  Parameters:
  _filename; string (full path)
  _classtype; string
  retruns a list of objects of a distinct class
  '''
  item_list = []
  with open(filename, newline='') as csvfile:  # creates a file object
    csvreader = csv.reader(csvfile, delimiter=',')  # set for reading csv file
    next(csvreader)  # skips over header

    if classtype == 'book':
      i = 0
      for row in csvreader:  # reads one row at a time
        item_list.append(Book(i, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        i += 1
        
    elif classtype == 'cd_vinyl':
      i = 10 #As there are 9 items in book.csv
      for row in csvreader:  
        item_list.append(Cd(i, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        i += 1

    elif classtype == 'collectible':
      i = 20 #As there are 10+10 items in book.csv and cd_vinyl.csv
      for row in csvreader:  
        item_list.append(Collectible(i, row[0], row[1], row[2], row[3], row[4]))
        i += 1
    
    elif classtype == 'electronics':
      i = 27 #Similar calculation as before
      for row in csvreader:  
        item_list.append(Electronics(i, row[0], row[1], row[2], row[3], row[4]))
        i += 1

    elif classtype == 'fashion':
      i = 36
      for row in csvreader:  
        item_list.append(Fashion(i, row[0], row[1], row[2], row[3], row[4]))
        i += 1

    else:
      i = 45
      for row in csvreader:  
        item_list.append(Homegarden(i, row[0], row[1], row[2], row[3], row[4]))
        i += 1

  return item_list


def add_indiv_items():
  '''
  Adds lists made by read_items function
  No Parameters
  Returns a list
  '''
  item_list_tot = read_items('data/book.csv', 'book') 
  item_list_tot += read_items('data/cd_vinyl.csv', 'cd_vinyl') 
  item_list_tot += read_items('data/collectible.csv', 'collectible')
  item_list_tot += read_items('data/electronics.csv', 'electronics')
  item_list_tot += read_items('data/fashion.csv', 'fashion')
  item_list_tot += read_items('data/home_garden.csv', 'home_garden')

  return item_list_tot

  
class Inventory:
  '''
    A class capturing information about all items from different csv files
    Attributes:
    _items; list
    _count; integer
  '''
  CATEGORY = ['Book', 'cd_vinyl', 'collectible','electronics', 'fashion', 'homegarden']

  def __init__(self):
    self.items = add_indiv_items()
    self.count = len(self.items)

  def check_type(self, item):
    '''
    Parameters:
    _self; an instance of Inventory class
    _item; an instance of Book/Cd/Collectible/Electronics/Fashion/Homegarden class
    Return the type of item as string
    '''
    classes = [Book, Cd, Collectible, Electronics, Fashion, Homegarden]
    category_list = []
    
    for i in range(len(classes)):
      if isinstance(item, classes[i]) == True:
        category_list.append(self.CATEGORY[i])

    if len(category_list) == 2: 
      return category_list[1] #a CD will be both book and CD, but we want it printed as a CD class
    return category_list[0] #All other classes are not subclasses of each other
  
  def compute_inventory(self):
    '''
    Calculalates total price of all inventory items
    Parameters:
     self; an instance of Inventory class
    Returns a float
    '''
    total = 0.0
    for item in self.items:
      total += float(item.unit_price)*float(item.quantity)

    return total

  def print_inventory(self, begin = 0, end = -1):
    '''
    Prints information of inventory items from items list
    Parameters:
     self; an instance of Inventory class
     begin; an integer
     end; an integer
    '''
    for i in range(begin, end):
      print(self.items[i])

  def print_category(self, category):
    '''
    Prints information of inventory items from items list based on category
    Parameters:
     self; an instance of Inventory class
     category; string
    '''
    p = '---'
    s = p 
    s += ' Print informaion of items of the type ' + category + ' '
    s += p
    print(s)
    
    for item in self.items: 
      category0 = self.check_type(item)
      if category0 == category:
        print(item)
        
    s = p + ' Print completes. ' + p + '\n'
    print(s)

  def search_item(self, str_pattern):
    '''
    Prints all items whose name contains str_pattern
    Parameters:
      self; an instance of Inventory class
      str_pattern; string
    '''
    s = 'Search for all items with "' +str_pattern+ '" in name.'
    print(s)
    
    for item in self.items:
      if str_pattern in item.name:
        print(item)