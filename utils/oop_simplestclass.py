#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 20:42:25 2018

@author: dawnstear
"""


# *--------Example 1--------------*

class Person1:
    pass  # An empty block

p = Person1()
print(p)

# *----------- Example 2------------* class with say_hi method
# We have already discussed that classes/objects can have methods 
# just like functions except that we have an extra self variable
class Person2:
    def say_hi(self):
        print('hi')
        
p = Person3()
p.say_hi()        
# The previous 2 lines can also be written as Person().say_hi()

# *----------- Example 3------------* init method
# The __init__ method is run as soon as an object of a class is
# instantiated (i.e. created). The method is useful to do any initialization
#  (i.e. passing initial values to your object) you want to do with your object. 
class Person3:
    def __init__(self,name):
        self.name = name
        
    def say_hi(self):
        print('Hi, I am', self.name)
        
p = Person3('Ben')        
p.say_hi()
# The previous 2 lines can also be written as Person('Swaroop').say_hi()



