#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:09:40 2022

@author: baptistelafoux
"""

import numpy as np 
import matplotlib.pyplot as plt 

import progressbar

from scipy.signal import convolve2d
from itertools import product 


global neighbours, lx, ly, n_iter
neighbours = np.array([[1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1]])

lx, ly = (200, 200)
n_iter = int(np.sqrt(3)/2 * np.mean([lx, ly]))

def compute_snowflake(alpha, beta, gamma): 
    print(f"\n{'alpha':>10} = {alpha:.3f} \n{'beta':>10} = {beta:.3f}\n{'gamma':>10} = {gamma:.3f}\n")

    cells, weights = init_cells(lx, ly, alpha, beta)

    for i in progressbar.progressbar(range(n_iter)):
        
        recep = receptive(cells)
        cells[recep] += gamma
    
        cells = recep * cells + convolve2d(~recep * cells, weights, 'same')
        
    ice = cells >= 1.
    
    return cells, ice 

def grid_to_hexa(lx, ly):
    
    grid_x, grid_y = np.meshgrid(np.arange(lx, dtype=float), np.arange(ly, dtype=float)) 

    rot_to_hexa = np.array([[0, -1],[np.sqrt(3)/2, 1/2]])
    grid_x, grid_y = (np.concatenate([grid_x[..., None], grid_y[..., None]], axis=-1) @ rot_to_hexa).T

    return grid_x, grid_y 

def init_cells(lx, ly, alpha, beta):
        
    cells_ini = np.ones((lx, ly)) * beta
    
    weights = 1. / 12. * alpha * neighbours
    weights[1, 1] = 1. - alpha / 2.
    
    cells_ini[lx // 2, ly // 2] = 1.0
    
    return cells_ini, weights

def sf_at_edge(cells):
    return any((cells[-1, :] >= 1) | (cells[0, :] >= 1) | (cells[:, -1] >= 1) | (cells[0, :] >= 1))
def receptive(cells):
    
    return (convolve2d(cells >= 1., neighbours, 'same') >= 1) | (cells >= 1.)

def video_growth_sf(alpha, beta, gamma): 
    cells, weights = init_cells(lx, ly, alpha, beta)
    grid_x, grid_y = grid_to_hexa(lx, ly)
    
    fig, ax = plt.subplots() 
    im = ax.pcolormesh(grid_x, grid_y, cells, shading='nearest', cmap='Greys_r', vmin=beta)
    
    ax.axis('scaled')
    ax.axis([0, lx * np.sqrt(3)/2, - 3*ly / 4 , ly / 4])
    ax.axis('off')
        
    ax.set_facecolor('k')
    fig.patch.set_facecolor('k')
    
    ax.set_title(rf'$\alpha$ = {alpha:.2f} | $\beta$={beta:.2f} | $\gamma$ = {gamma:.4f}', color='w')
    
    plt.tight_layout()
    for i in progressbar.progressbar(range(n_iter)):
        recep = receptive(cells)
        cells[recep] += gamma
    
        if sf_at_edge(cells) : break 
        cells = recep * cells + convolve2d(~recep * cells, weights, 'same')
        
        
        if i % 12 == 0:
            im.set_array(cells)
            plt.draw()
            plt.pause(0.001)      
            
def generate_a_gif(alpha, beta, gamma): 
    
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    cells, weights = init_cells(lx, ly, alpha, beta)
    grid_x, grid_y = grid_to_hexa(lx, ly)
    
    fig, ax = plt.subplots() 
    im = ax.pcolormesh(grid_x, grid_y, cells, shading='nearest', cmap='Greys_r', vmin=beta)
    
    ax.axis('scaled')
    ax.axis([0, lx * np.sqrt(3)/2, - 3*ly / 4 , ly / 4])
    ax.axis('off')
        
    ax.set_facecolor('k')
    fig.patch.set_facecolor('k')
    
    ax.set_title(rf'$\alpha$ = {alpha:.2f} | $\beta$={beta:.2f} | $\gamma$ = {gamma:.4f}', color='w')
    
    all_cells = np.zeros((n_iter, lx, ly))
    for i in progressbar.progressbar(range(n_iter)):
        recep = receptive(cells)
        cells[recep] += gamma
    
        if sf_at_edge(cells) : break 
        cells = recep * cells + convolve2d(~recep * cells, weights, 'same')
        
        all_cells[i] = cells
    
    def animation(i): 
        
        im.set_array(all_cells[i]) 
            
    
    anim_created = FuncAnimation(fig, animation, frames=n_iter, interval=25)
    anim_created.save(f'gifs/a={alpha:.2f}_b={beta:.2f}_g={gamma:.4f}.gif', dpi=100, writer=PillowWriter(fps=n_iter//5))
        
    
        
def main():
    gamma = 0.005
    
    alphas = np.linspace(1.3, 1.95, 4)
    betas  = np.linspace(0.45, 0.75, 5)
    
    fig, axs = plt.subplots(len(alphas), len(betas), figsize=(10, 10))
    
    grid_x, grid_y = grid_to_hexa(lx, ly)
    
    for (i, alpha), (j, beta) in product(enumerate(alphas), enumerate(betas)):
        
        cells, _ = compute_snowflake(alpha, beta, gamma)
        
        ax = axs[i, j]
        ax.pcolormesh(grid_x, grid_y, cells, shading='auto', cmap='Greys_r', vmin=beta)
        
        ax.axis('scaled')
        
        ax.axis([0, lx * np.sqrt(3)/2, - 3*ly / 4 , ly / 4])
        #ax.axis('off')

        fig.patch.set_facecolor('k') 
        ax.set_facecolor('k')
        
        if beta  ==  betas[0] : ax.set_ylabel(fr'$\alpha$ = {alpha:.2f}', color='w')
        if alpha == alphas[-1] : ax.set_xlabel(fr'$\beta$ = {beta:.2f}', color='w')
    
    plt.tight_layout()
    
if __name__ == '__main__':
    plt.close('all')
    #main()
    alpha = np.random.rand() * 1.5 + 0.6
    beta  = np.random.rand() * 0.3 + 0.55
    gamma = np.random.rand() * 0.005

    #video_growth_sf(alpha, beta, gamma)
    generate_a_gif(alpha, beta, gamma)

