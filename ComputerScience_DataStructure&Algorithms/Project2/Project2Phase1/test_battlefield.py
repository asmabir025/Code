'''Name: AKM Sadman (Abir) Mahmud
DaTE: Mar 26, 2023
'''


from battlefield import *

# For a shorter (4*4) Battlefield example
battlefield1 = Battlefield(4,4)
print(battlefield1)

col = battlefield1.get_col_size()
row = battlefield1.get_row_size()
print(row, col)

print(battlefield1.is_in_battlefield(row-1, col-1))
print(battlefield1.is_in_battlefield(row+1, col+1))
print(battlefield1.is_in_battlefield(-1, 0))
print(battlefield1.is_in_battlefield(0, -1))

battlefield1.set_value(row-1, col-1, '!')
print(battlefield1)

print(battlefield1.get_value(row-1, col-1))

# For a Battlefield example, which uses the same size as the sample files
battlefield2 = Battlefield()
f_name = input( "Enter the name of the battlefield file (\"none\" if " \
		     + " using random file): ")
print( f_name )

if f_name != 'none':
    battlefield2.read_battlefield( f_name )

print(battlefield2)

col2 = battlefield2.get_col_size()
row2 = battlefield2.get_row_size()
print(row2, col2)

print(battlefield2.is_in_battlefield(row2-1, col2-1))
print(battlefield2.is_in_battlefield(row2+1, col2+1))
print(battlefield2.is_in_battlefield(-1, 0))
print(battlefield2.is_in_battlefield(0, -1))

battlefield2.set_value(row2-1, col2-1, '!')
print(battlefield2)

print(battlefield2.get_value(row2-1, col2-1))