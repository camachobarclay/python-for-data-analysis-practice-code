import numpy as np
from math import *
values = 1,2,3,4,5

a,b, *rest = values
print(a,b); print(rest)

a = (1,2,2,2,3,4,2); print(a.count(2))
a_list = [2,3,7,None]; print(a_list)

tup = ('foo', 'bar', 'baz')
b_list = list(tup); print(b_list)

b_list[1] = 'peekaboo'; print(b_list[1])

gen = range(10)
print(gen); print(list(gen))

b_list.append('dwarf'); print(b_list)

b_list.insert(1,'red'); print(b_list)

b_list.pop(2); print(b_list)

b_list.append('foo'); print(b_list)

b_list.remove('foo'); print(b_list)

print('dwarf' in b_list); print('dwarf' not in b_list)

print([4,None, 'foo']+ [7,8,(2,3)])

x = [4,None,'foo']; x.extend([7,8,(2,3)]); print(x)

#everything = []
#for chunk in list_of_lists:
#    everything.extend(chunk)

#is faster than

#
#everything = []
#for chunk in list_of_lists:
#   everything = everything + chunk

a = [7, 2, 5,1,3]; a.sort(); print(a)

b = ['saw', 'small', 'He', 'foxes', 'six']; b.sort(key = len); print(b)

import bisect

c = [1, 2, 2, 2, 3, 4, 7]

print(bisect.bisect(c,2)); print(bisect.bisect(c,5))

print(bisect.insort(c,6))

seq = [7, 2, 3, 7, 5, 6, 0, 1]

print(seq[1:5])

seq[3:4] = [6,3]; print(seq)

print(seq[:5])

print(seq[3:])

print(seq[-4:])

print(seq[-6:-2])

print(seq[::2])

print(seq[::-1])

some_list = ['foo', 'bar', 'baz']

mapping = {}

for i, v in enumerate(some_list):
    mapping[v] = i

print(mapping)

seq1 = ['foo', 'bar', 'baz']

seq2 = ['one', 'two', 'three']

zipped = zip(seq1, seq2)


print(list(zipped))


seq3 = [False, True]

print(list(zip(seq1, seq2, seq3)))

for i, (a,b) in enumerate(zip(seq1,seq2)):
    print('{0}: {1}, {2}'.format(i,a,b))

pitchers = [('Nolan', 'Ryan'), ('Roger','Clemens'), ('Schilling', 'Curt')]

first_names, last_names = zip(*pitchers)

print(first_names); print(last_names)

empty_dict = {}

d1 = {'a': 'some value', 'b': [1,2,3,4]}
d1[7] = 'an integer'                       
print(d1); print(d1['b']); print('b' in d1)

d1[5] = 'some value'; print(d1)

d1['dummy'] = 'another value'; print(d1)

del d1[5]; print(d1)

ret = d1.pop('dummy'); print(ret)

print(d1)

print(list(d1.keys())); print(list(d1.values()))

d1.update({'b':'foo', 'c': 12}); print(d1)

mapping = {}


#for key, value in zip(key_list, value_list):
#    mapping[key] = value

mapping = dict(zip(range(5), reversed(range(5))))

print(mapping)

by_letter = {}

words = ['apple', 'bat', 'bar', 'atom', 'book']

for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter = {}

print(by_letter)

for word in words:
    letter = word[0]
    by_letter.setdefault(letter,[]).append(word)

from collections import defaultdict

by_letter = defaultdict(list)

for word in words:
    by_letter[word[0]].append(word)

print(by_letter)

d = {}

d[tuple([1,2,3])] = 5; print(d)

a = set([2,2,2,1,3,3])
aa = {2,2,2,1,3,3}
b = {3,4,5,6,7,8}

print(a); print(aa); print(b)
print(a.union(b)); print(a|b)
print(a.intersection(b)); print(a&b)


d = a.copy(); print(d)
d &= b; print(d)

my_data = list(range(1,5))
my_set = {tuple(my_data)}; print(my_set)

a_set = {1,2,3,4,5}

print({1,2,3}.issubset(a_set))

print(a_set.issuperset({1,2,3}))

print({1,2,3}=={3,2,1})

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

print([x.upper() for x in strings if len(x) > 2])

unique_lengths = {len(x) for x in strings}

print(unique_lengths)

print(set(map(len,strings)))

loc_mapping = {val: index for index, val in enumerate(strings)}

print(loc_mapping)

all_data = [['John', 'Emily', 'Michael', 'Mary','Steven'],[
    'Marla','Juan',' Javier','Natalia', 'Pilar']]

names_of_interest = []

for names in all_data:
    enough_es = [name for name in names if name.count('e') >= 2]
    names_of_interest.extend(enough_es)

result = [name for names in all_data for name in names if name.count('e') >=2]

some_tuples = [(1,2,3),(4,5,6),(7,8,9)]

flattened = [x for tup in some_tuples for x in tup]

print(flattened)

flattened = []

for tup in some_tuples:
    for x in tup:
        flattened.append(x)

print([[x for x in tup] for tup in some_tuples])

def my_function(x,y,z = 1.5):
    if z > 1:
        return z*(x+y)
    else:
        return z/(x+y)

def func():
    a = []
    for i in range(5):
        a.append(i)

b = []

def func2():
    for i in range(5):
        b.append(i)

c = None

def bind_c_variable():
    global c 
    c = []

bind_c_variable()

print(c)

def f():
    a = 5
    b = 6
    c = 7
    return a, b, c

a,b,c = f()

return_value = f()

def F():
    a = 5
    b = 6
    c = 7
    return {'a': a, 'b': b, 'c': c}

states = ['  Alabama  ', 'Georgia!', 'Georgia', 'georgia', 'FlOrida', 'south carolina##', 'West virginia?']

import re

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]','',value)
        value = value.title()
        result.append(value)
    return result

clean_strings(states)

def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings,ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

print(clean_strings(states,clean_ops))

def short_function(x):
    return x*2

equiv_anon = lambda x: x*2

def apply_to_list(some_list, f):
   return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x:x*2)

strings = ['foo', 'card', 'bar', 'aaaa', 'abab']

strings.sort(key = lambda x: len(set(list(x))))

print(strings)

def add_numbers(x,y):
    return x+ y

add_five = lambda y: add_numbers(5,y)

from functools import partial

add_five = partial(add_numbers, 5)

some_dict = {'a':1, 'b':2, 'c':3}

for key in some_dict:
    print(key)

dict_iterator = iter(some_dict)
print(dict_iterator)

list(dict_iterator)

def squares(n = 10):
    print('Generating squares from 1 to {0}'.format(n**2))
    for i in range(1,n+1):
        yield i**2

gen = squares()

print(gen)

for x in gen:
    print(x, end =' ')

gen = (x ** 2 for x in range(100))

print(gen)

def _make_gen():
    for x in range(100):
        yield x ** 2
gen = _make_gen()

sum(x ** 2 for x in range(100))

dict((i, i **2) for i in range(5))

{0: 0, 1:1, 2:4, 3:9, 4:16}

import itertools
first_letter = lambda x: x[0]

names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']

for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator

print(float('1.2345'))

#print(float('something'))

def attempt_float(x):
    try:
        return float(x)
    except:
        return x

print(attempt_float('1.2345'))

print(attempt_float('something'))

def Attempt_Float(x):
    try:
        return float(x)
    except (TypeError,ValueError):
        return x

print(Attempt_Float('(1,2)'))

#f = open(path, 'w')

#try:
#    write_to_file(f)
#finally:
#    f.close()

#f = open()

#try:
#   write_to_file(f)
#except:
#   print('Failed')
#else:
#   print('Succeeded')
#finally:
#   f.close()

#path = 'examples/segismundo.txt'
#f = open(path)

#for line in f:
#    pass

#lines = [x.rstrip() for x in open(path)]

#print(lines)

#f.close()

#with open(path) as f:
#    lines = [x.rstrip() for x in f]

#f = open(path)

#f.read(10)

#f2 = open(path,'rb') #binary mode

#f2.read(10)

#f.tell()

#f2.tell()

#import sys
#sys.getdefaultenconding()
#f.seek(3)
#f.read(1)

#f.close()
#f2.close()

#with open('tmp.txt', 'w') as handle:
#    handle.writelines(x for x in open(path) if len(x) > 1)

#with open('tmp.txt') as f:
#    lines = f.readlines()

#print(lines)

#with open(path) as f:
#    chars = f.read(10)

#print(chars)

#with open(path, 'rb') as f:
#    data = f.read(10)

#print(data)

#data.decode('utf8')

#print(data[:4].decode('utf8'))

#sink_path = 'sink.txt'

#with open(path) as source:
#    with open(sink_path, 'xt', encoding = 'iso-8859-1') as sink:
#        sink.write(source.read())

#with open(sink_path, encoding = 'iso-8859-1') as f:
#    print(f.read(10))

#f = open(path)

#f.read(5)

#f.seek(4)

#f.read(1)

#f.close()
