import parser
import matplotlib.pyplot as plt
import csv

DEBUG = False

def parse_html(date, html_filepath):
  '''
  Counts the number of images, unlabeled images, and images with missing alt text
  in a given html file. Writes that information to a CSV file.
  Parameters:
     date; integer; represents year of html file
     html_filepath: string. full path to html file to read in.
  Return: dictionary mapping the following keys (strings) to their appropirate values. 
  
     Keys: value description
      "DATE": int, html file year 
      "TOTAL_IMG": int, total number of images
      "NUM_UNL": int, number of unlabeled images
      "PROP_UNL": float, proportion of images missing labels out of all images (will be 0 if there are no images in the file)
  '''
  
  # given html filepath, get list of dictonary objects for all images on page
  # images will be a list of dictonaries. 
  # each dictonary represents an image on the html page
  # the keys of the dictonary are attributes of the image
  images = parser.parse_file(html_filepath)
  if DEBUG:
    print("html file:",html_filepath)
    print("images:",images)
    print("number of elements:",len(images))
    if len(images)>0: #to avoid error for html files without images
      print("first element in images:",images[0])
      print("data type of the first element in images:",type(images[0]))
      if 'alt' in images[0].keys(): #to avoid error for html files with the first image without any label 
        print("value of alt attribute of first element in images:",images[0]['alt'])
    
  # Step 2,

  # add print statements to check your work from
  # steps 2 here 

  num_img = len(images) #num_img = number of image in each html file
  
  '''
  For counting unlabeled images, we will go through each element in images (for loop), will only count that unlabeled if it does not have alt attribute (i.e. if 'alt' is not a key )
  
  '''
  count_unlabel = 0
  for img in images: 
    if 'alt' not in img.keys():
      count_unlabel+=1 

  '''
  For calculating the proportion of unlabeled images, there will be two cases:
case 1: no image in html file.
- prop = 0
case 2: non-zero images in html file. 
  - prop = count_unlabel/num_img
  '''
  prop = 0
  if num_img > 0:
    prop = round(count_unlabel/num_img,2)

  dict = {"DATE":date, "TOTAL_IMG":num_img, "NUM_UNL":count_unlabel, "PROP_UNL": prop} 
  
  if DEBUG:
    
    print("number of images in html file:",num_img)
    print("number of unlabled images in html file:",count_unlabel)
    print("proportion unlabled:",prop)
  
  
  # update to return dictionary
  return dict


def graph_prop_unlabled_by_date(csv_filepath):
  '''
  Graphs the proportion of unlabeled images by date based 
  on data from csv file.

  CSV must have headers "date" "prop_unl" and "no_alt"
  
  Parameters:
    csv_filepath: string. full filepath of csv with image data.
    
  '''
  
  # To plot, we need 2 lists, the x-axis values and y-values

  # Step 5
  # create lists for dates and proportion unlabeled
  # read in data from CSV to fill list
  
  dates = [] #initiating a list for dates
  prop_unl = [] #initiating a list for unlabeled proportions
  
  with open(csv_filepath, 'r') as f: 
  # read the CSV file
    csv_dicts = csv.DictReader(f)
    
  # go through every row's dictionary
    for entry in csv_dicts:
       #print("entry type:", type(entry))
       print("entry:",entry,"\n")
       dates.append(int(entry['DATE']))
       prop_unl.append(float(entry['PROP_UNL']))

    
  print('list of dates:',dates)
  print('len(dates):',len(dates))
  print('type(dates[0]):', type(dates[0]))
  print('list of proportions of unlabeled images:',prop_unl)
  print('len(prop_unl):',len(prop_unl))
  print('type(prop_unl[0]):',type(prop_unl[0]))
  
  # Step 6: graph results
  # plot
  fig, ax = plt.subplots()
  # label axes
  ax.set_title("Proportion Unlabled Images, bucknell.edu")
  ax.set_xlabel("year")
  ax.set_ylabel("proportion")
  # plot data
  ax.scatter(dates,prop_unl)
  #save plot
  graph_file = "results_graph.png"
  if DEBUG:
    graph_file ="test_graph.png"
  
  plt.savefig(graph_file)

def write_headers(results_csv_file):
  '''
  Write column labels (headers) to CSV file.
  Parameters:
    results_csv_filename; string; filepath to CSV to write to
  Return:
    None 
  '''
  with open(results_csv_file, "w") as outfile:
    #write header
    # Note: in a CSV, every value is separated by a comma
    outfile.write("DATE"+",")
    outfile.write("TOTAL_IMG"+",")
    outfile.write("NUM_UNL"+",")
    # note the newline rather than a comma
    # since this will be our last column 
    outfile.write("PROP_UNL"+"\n")
  

def write_info(html_img_info,header_file):
  '''
  This function takes the data about html file images from html_img_info and adds the data from it to header_file.
  
  Parameters:
  html_img_info: dictionary; keys represent the image properties: DATE, TOTAL_IMG (total number of images), NUM_UNL (number of unlabeled images),PROP_UNL (proportion of unlabeled images)
  header_file: string; a full file-path to CSV to write to

  Returns:
  None
  
  '''
  
  with open(header_file, "a") as outfile:
    #appending to outfile
    outfile.write(str(html_img_info['DATE'])+',')
    outfile.write(str(html_img_info['TOTAL_IMG'])+',')
    outfile.write(str(html_img_info['NUM_UNL'])+',')
    # note the newline rather than a comma
    # since this will be our last column
    outfile.write(str(html_img_info['PROP_UNL'])+'\n')

  
  
def evaluate_accessibility():
  # the directory path of the local HTML files we're analyzing
  basepath = "bucknell_web/"
  results_csv_filename = "unl_img_per_year.csv"
  if DEBUG:
    results_csv_filename = "test_img_count.csv"
    '''
    Commented next 3 lines out for preventing writing in test_img_count.csv by a single html file (outside the for loop) in step 4
    
    '''
    #basepath = "test_html/"
    #date = 4
    #filepath = basepath + "test_"+str(date)+".html" 
  
    write_headers(results_csv_filename) #adding header to results_csv_filename
  
  # Step 4 add code to parse all files

    date_list = list(range(1,5)) #for covering all date in test_html  
    parsed_html_list = [] #initiating the list for catching parsed list from each html file 
    for n in date_list: 
      
      #looping over the whole folder and catching each html file info to the parsed_html_list
      
      basepath = "test_html/" #specific basepath/directory for test files
      filepath = basepath + "test_"+str(n)+".html"
      print ('Processing test_html/test_'+str(n)+'.html') 
      single_parsed_html = parse_html(n,filepath) 
      parsed_html_list.append(single_parsed_html)
      #for checking 
      print('parsed_html_list:',parsed_html_list)
    
    print('Finished processing html files')
    
    
    for m in date_list:
      #adding each html info to the csv file
      write_info(parsed_html_list[m-1],results_csv_filename)
      
  # Step 2: Parse single html file
  
  #results = parse_html(date,filepath)
  
  #if DEBUG:
    #print("results for",filepath,":",results)
    
  

  # Step 3: Write results to csv file
  #write_info(results,results_csv_filename)

  '''
  The next few lines of codes are for parsing Bucknell html data. The plan has been described in each comment.

  '''
  
  write_headers(results_csv_filename) #adding header to results_csv_filename
  
  date_list = list(range(1996,2023)) #for covering all bucknell html files from different dates 
  parsed_html_list = [] #initiating the list for catching parsed list from each html file 
  
  for n in date_list: 
    #looping over the whole folder and catching each html file info to the parsed_html_list
    
    filepath = basepath+'bucknell_'+str(n)+".html"
    print ('Processing '+filepath) #to show what html file is being processed at the moment 
    single_parsed_html = parse_html(n,filepath)
    parsed_html_list.append(single_parsed_html) 
  
  print('Finished processing html files')

  for m in date_list:
    #adding each html info to the csv file
    write_info(parsed_html_list[m-1996],results_csv_filename) #for example, the first element's index is 0 = 1996 - 1996, where m =1996 

  
    
  
  # Step 5 & 6
  graph_prop_unlabled_by_date(results_csv_filename)  



