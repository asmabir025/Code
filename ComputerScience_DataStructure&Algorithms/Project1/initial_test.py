from inventory import *

book1 = Book(1,'A Wrinkle in Time (Time Quintet)','1-May-07','Square Fish',"Madeleine L'Engle" ,'5.50','978-0312367541','20')
print(book1)

cd1 = Cd(1, '4:44','JAY-Z','Roc Nation','B073MBN7BY','7-Jul-17','12.94', '10')
print(cd1)

col1 = Collectible(2, 'Warhammer 40k Ork', '213.00' ,'6/7/1958','Emma Jenkins', '1')
print(col1)

elec1 = Electronics(5, 'Amplifier', '217.00' ,'12/12/2011','Sony', '25')
print(elec1)

fash1 = Fashion(3, 'TAG Heuer Watch', '793.00' ,'10/13/1995','TAG', '15')
print(fash1)

homegarden1 = Homegarden(10,'Polaris 280 In Ground Pressure Side Automatic Pool Cleaner Sweep F5 Scrubber','418.00','9/4/2003','Polaris','15')
print(homegarden1)

it1 = read_items('data/book.csv','book')
print(it1)

s = Inventory()

print(s.check_type(s.items[20]))

print(s.compute_inventory())