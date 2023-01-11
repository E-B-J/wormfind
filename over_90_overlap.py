# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:29:17 2023

@author: ebjam
"""

def over_90_overlap(childmask, parentmask):
    '''
    

    Parameters
    ----------
    childmask : Smaller mask - in this case the embryo
    parentmask : Larger mask - in this case the worm

    '''
    embryolen = len(childmask) #Getting length of embryo mask - how many pixels
    intersection = [q for q in childmask if q in parentmask]
    percentage_overlap = 100*(len(intersection)/embryolen)
    if percentage_overlap > 90:
        intersect = True
    else:
        intersect = False
    return(intersect)
    