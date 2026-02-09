""" n = 0
print('n =', n)
n = "abc"
print('n =', n)

# Mutiple assignment
n,m = 0,"abc"
print('n =', n, 'm =', m)
n,m,z = 0.125, "abc",  False
print('n =', n, 'm =', m, 'z =', z)

#Incrementing a variable 
n = 0
n = n+1 #Good but verbose
print('n =', n)
n += 1 #Better
print('n =', n)
#n++ #Not valid in Python
# None is null in Python
n = None
print('n =', n) 

#If statement don't need parentheses but do need a colon and indentation
#n = 1
#n = 0
#n = -1 
if n > 0:
    print('n is positive')
elif n == 0:
    print('n is zero')
else:
    print('n is not positive')


# Parenthese needed for mutiple assignment
# and = &&


 def get_full_name(first_name,last_name):
    full_name = first_name.title() " " + last_name.title()
    return full_name   
print(get_full_name("john", "doe")



def get_full_name(first_name, last_name):
    full_name = first_name.title() + " " + last_name.title()
    return full_name


print(get_full_name("john", "doe"))"""