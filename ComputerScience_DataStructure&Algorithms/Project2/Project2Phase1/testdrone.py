'''Name: AKM Sadman (Abir) Mahmud
DaTE: Mar 26, 2023
'''
# Uncomment when you're ready for your dronerecursion.py functionality
from dronerecursion import Drone  # uncomment for phase 1
#from dronestack import Drone      # uncomment for phase 2
from battlefield import Battlefield

# Uncomment once you have your drone class (phase 1 or phase 2)
drone = Drone()
battlefield = Battlefield()

f_name = input( "Enter the name of the battlefield file (\"none\" if " \
		     + " using random file): ")
print( f_name )

if f_name != 'none':
    battlefield.read_battlefield( f_name )

print( '-------- The Original Battlefield --------' )
print( battlefield )

starting_row = int( input( 'Please enter the starting row : ' ) )
print( starting_row )
starting_col = int( input( 'Please enter the starting column : ' ) )
print( starting_col )

exit_row = int( input( 'Please enter the exiting row : ' ) )
print( exit_row )
exit_col = int( input( 'Please enter the exiting column : ' ) )
print( exit_col )

# Uncomment once you have your drone class
drone.find_battlefield_paths( battlefield, starting_row, starting_col, exit_row, exit_col )
drone.show_shortest_path()

# I added this code only for testing when I was checking if all solved paths are unique 
# for i in range(drone.num_path):
#   for j in range(drone.num_path):
#     if drone.all_paths[i] == drone.all_paths[j]:
#       print((i,j))
