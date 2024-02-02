'''
DO NOT ALTER
'''
# parse HTML from dictionaries into classes
# count how many do not have alt tags
# correct bad labels

# import required module
import os
from html.parser import HTMLParser
#https://docs.python.org/3/library/html.parser.html

class WebPageTester(HTMLParser):
  IMAGE_TAG = "img"
  ALT_TEXT_TAG = "alt"
  
  def __init__(self, file):
    self.file = file
    self.images = []
    super().__init__()
  
  
  def handle_starttag(self, tag, attrs):
    if tag == WebPageTester.IMAGE_TAG:
      alt = None
      attrs = dict(attrs)
      self.images.append(attrs)   

def parse_file(filepath):
  '''
  Parameters:
    filepath; string; full path to local html file
  Return:
    list of dictionaries. one list entry per image element on the html page. each entry in the dictionary is an attribute of the image.
  '''
  #print(filepath)
  # checking if it is a file
  if os.path.isfile(filepath):
    #print(filepath)
    parser = WebPageTester(filepath)
    with open(filepath,"r") as file:
      lines = file.readlines()
      for line in lines:
        parser.feed(line)
    return parser.images
  else:
    print("No file")



    
def parse_directory():
  # assign directory
  #directory = 'bucknell_web'
  directory = 'bucknell_athletics'
  
  # iterate over files in
  # that directory
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
      print(f)
      parser = WebPageTester(f)
      with open(f,"r") as f:
        lines = f.readlines()
        for line in lines:
          parser.feed(line)
      print(parser.images)
          # print(parser)
      # parser.check_labels()
      # parser.print_image_labels()
      # print(parser)

    
