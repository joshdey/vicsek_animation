import streamlit as st
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components

L = st.sidebar.number_input(label='2D Square Lattice Length', value=32.0)
rho = st.sidebar.number_input(label='ρ, Density', value=3.0)
N_default = int(rho*L**2)
N = st.sidebar.number_input(label='Number of particles', value=N_default)

r0 = st.sidebar.number_input(label='r0, Initial Position', value=1.0)
deltat = st.sidebar.number_input(label='Δt, Change in Time', value = 1.0)
factor = st.sidebar.number_input(label='Velocity Scaling Factor', value=0.5)
v0 = r0/deltat*factor
iterations = st.sidebar.number_input(label='Number of Iterations', value=10000)
eta = st.sidebar.number_input(label='η, Noise', value=0.15)

pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)
 
fig, ax= plt.subplots(figsize=(6,6))
 
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient[0]), np.sin(orient), orient, clim=[-np.pi, np.pi])

#@st.cache(suppress_st_warning=True)
def animate(i, r0=r0, L=L):
    #print(i)
    #st.progress(i/2)
    global orient
    tree = cKDTree(pos,boxsize=[L,L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')
 
    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
     
     
    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)
 
 
    cos, sin= np.cos(orient), np.sin(orient)
    pos[:,0] += cos*v0
    pos[:,1] += sin*v0
 
    pos[pos>L] -= L
    pos[pos<0] += L
 
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin,orient)
    return qv,

st.title("Vicsek Model Animation in Python")
st.markdown('#### Author: Josh')
st.markdown('#### Date: 04/29/21')
st.markdown("##### Code Based On: [Minimal Vicsek Model in Python by Francesco Turci](https://francescoturci.net/2020/06/19/minimal-vicsek-model-in-python/#:~:text=Minimal%20Vicsek%20model%20in%20Python%20June%2019%2C%202020,matter.%20It%20displays%20interesting%20features%2C%20such%20as%20swarming)")
st.markdown('##### More information on parameters and Vicsek Model: [The Flocking Transition: A Review of The Vicsek Model](https://guava.physics.uiuc.edu/~nigel/courses/563/Essays_2017/PDF/Chatterjee.pdf)')
st.markdown("*Please note - you can change the number of agents in the lattice by either changing the density of the agents in the lattice by modifying the ρ parameter or by directly inputting the number of agents in the N parameter in the sidebar. The default N takes the default density multiplied by the area of the lattice.*")
with st.spinner("Running..."):
    ani = FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True)
components.html(ani.to_jshtml(), height=1000)
