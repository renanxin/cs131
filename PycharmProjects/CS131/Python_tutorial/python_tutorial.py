############################################################################
#String
s='hello'
print(s.capitalize())
print(s.upper())
print(s.rjust(7))
print(s.center(7))
print(s.replace('l','ell'))
print('   world'.strip())
############################################################################
#dictionary
d={'cat':'cute','dog':'furry'}
d['fish']='wet'
print(d.get('monkey','N/A'))        #取值，如果不存在，则会采用默认值
print(d.get('fish','N/A'))
############################################################################
#set
animals = {'cat', 'dog', 'fish'}
for idx,animal in enumerate(animals):
    print('#%d: %s'%(idx+1,animal))
