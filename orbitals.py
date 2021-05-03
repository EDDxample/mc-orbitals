from numpy.linalg import norm as mag
from numpy import e, pi as π
from numpy import zeros, random, mgrid
from numpy import arctan2, arccos, arctan
from scipy import special
from scipy.special import sph_harm as spherical_harmonics, eval_genlaguerre as laguerrel

import matplotlib.pyplot as plt


def get_probability(n, l, m, x, y, z):
    '''
    An electron's likelihood of existing
    at a certain location is defined by 

        Pₙₗₘ(r,θ,φ) = 4πr² ψr² ψθ²
    
    where:
        ρ = 2r/n
        ψr = e⁻ᵖ ρˡ Lₙ₋₁₋ₗ²ˡ⁺¹(ρ)
        ψθ = Yₗᵐ(θ,φ)

    credits: http://bugman123.com/Physics/
    '''

    r = mag((x,y,z))
    θ = arctan2(mag((x,y)), z)
    φ = arctan2(y,x)

    ρ = 2*r/n
    ψr = e**(-ρ) * ρ**l * laguerrel(n-1-l, 2*l+1, 2*ρ)
    ψθ = abs(spherical_harmonics(m, l, θ, φ))
    return 4*π*r*r * ψr*ψr * ψθ*ψθ


def plot_orbital(n=3, l=0, m=0, SIZE=20):
    '''
    Plots a 2D image of the orbital defined by nlm
    '''
    values = zeros((SIZE*2+1, SIZE*2+1))
    space = mgrid[-SIZE:SIZE+1, -SIZE:SIZE+1].reshape(2,-1).T
    for xy in space:
        x, y = xy
        values[x + SIZE, y + SIZE] = get_probability(n, l, m, x, y, 0)
    
    test = plt.imshow(values, cmap='plasma', interpolation='nearest')
    plt.text(0, 60, f'n={n} l={l} m={m}', fontsize=13, color='white')
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_orbital(4,0,0)