#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# Here we develop a one-dimensional model of diffusion.
# It assumes a constant diffusivity. 
# It uses a regular grid.
# It has fixed boundary conditions.

# The diffusion equation:
# 
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# The discretized version of the diffusion equation that we'll solve with our model:
# 
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# 
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011). (Or see Wikipedia.)

# FTCS means forward in time, centered in space

# We will use two libraries, Numpy (for arrays) and Matplotlib (for plotting) that aren't part of the base Python distribution.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 


# Set two fixed model parameters, the diffusivity and the size of the model domain.

# In[ ]:


D = 100
Lx = 300


# D is diffusivity and Lx is domain size

# Next, set up the model grid using the NumPy array.

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)


# In[ ]:


whos


# dx is the grid spacing, x is and array from 0 up 299.5 (well 300 ish) at an interval of 0.5, and nx is telling you how many values in the elements in the array which should be 600

# The whos command tells you what elements are available

# In[ ]:


x


# What if you want a subset of x? 

# In[ ]:


x[0]


# This is showing you that 0.0 is the first value of the numpy element of the array

# In[ ]:


x[nx-1]


# This is showing the last element of the array 

# In[ ]:


x[-1]


# In[ ]:


x[0:5]


# This is showing the first 5 elements of the array

# Set the initial concentration profile for the model. 
# The concentration `C` is a step function with a high value on the left, a low value on the right, and the step at the center of the domain.

# We ued the back tic (  `) on either side of C, it makes it into code 

# In[ ]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x <= Lx//2] = C_left
C[x > Lx//2] = C_right


# Inside the numpy library there is a function called zeros_like which makes an array of zeros that is like another array, for this example we are using x, so now we are making C a 600 element array with floats that are filled with zeros. On the left side, we want C to be 500 and on the right side we want C to go down to 0. Then we are saying that C, for values of x (0 to 299.5) that are less than or equal to the domain length divided by 2 (which has 2 hashes for division to make sure it is an integer) or half the domain size will be equal to 500, then C for vavlues of x that are greater than half the domain length are equal to 0.

# In[ ]:


C


# Plot the initial profile 

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")


# Now we are using the matplotlib function to create an initial concentration profile. The r in quotations is just saying the color of the line to be red

# Set the start time of the model and the number of time steps.
# Calculate a stable time stepfor the model using a stability criterion. 

# In[ ]:


time = 0 
nt = 5000
dt = 0.5 * (dx**2 / D)


# nt is the number of time steps, the timestep itself is dt which is 0.5 times dx^2 divided by D

# In[ ]:


dt


# In[ ]:


z = 5


# In[ ]:


z = z + 5
z


# In[ ]:


z += 5
z


# Loop over the time steps of the model, solving the diffusion equation using the FTCS explicit scheme described above. 
# The boundary conditions are fixed, so reset them at each time step.

# In[ ]:


for t in range(0, nt):
    C += D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C, 1))
    C[0] = C_left
    C[-1] = C_right


# brackets are for array indexing,
# parentheses are for grouping

# We move our model forward in time one step at a time. This is an example of a forloop (execute a statement or group of statements a set number of times), the loop counter is t which is intended to be current model time (you could also use an _ instead of t, but it is easier to see t) using the range function (built in function in python) (range from 0 to nt). This is how we step through the model with time

# The next line maps to the discretized version of the diffusion equation at the top of the page 

# C (concentration), the initial value of C plusequal the diffusivity time the timestep divided by gridspacing^2 times the center difference is the stuff in the parentheses at the end which you have a shift by 1 to the left and then there is a shift by 1 to the right minus 2 in the middle

# You are not updating concentration in the expression, it is creating a copy of C shifted to the left and right 

# Plot the result. 

# In[ ]:


plt.figure()
plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")


# The roll stuff, we are shifting to the left but there is nothing to the left, so we are just rolling over the value from the end to the beginning so that we have a zero there and we do not get a null value in the equation 

# In[ ]:


z = np.arange(5)
z


# In[ ]:


np.roll(z, -1)


# In[ ]:


np.roll(z, 1)


# In[ ]:


for i in range(len(z)):
    print(z[i] +1)


# This is the same as 

# In[ ]:


z + 1


# The first is printing out the elements of the array while the second is showing the array. The array functions are just faster to do. Doing a loop (the first one) isnt the worst thing to do, just try to avoid it. 

# In[ ]:




