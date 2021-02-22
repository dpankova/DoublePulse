# Functions for visualizing nu_tau events in IceCube
# Aaron Fienberg
#
# must call init_dom_map before plotting
#
# April 2020

import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import math

max_str_colors = ['r', 'g', 'b']

deep_core_color = 'brown'

default_distsq_cut = 150**2


_global_dom_map_holder = []


def get_global_dom_map():
    try:
        return _global_dom_map_holder[0]
    except IndexError:
        raise RuntimeError('global dom map is uninitialized!')


def display_three_channel(evt, score):
    fig = plt.figure(figsize=(12, 5))

    img = evt['image']
    qs = evt['q_st_dist']['qs'].flatten()
    sts = evt['q_st_dist']['st_nums'].flatten()

    channels = img.T.reshape((3, 60, 500))

    for i in range(3):
        ax = fig.add_subplot(131+i)
        ax.set_title(f'ch. {i}, st. {sts[i]:.0f}, Q: {qs[i]:.1f}',
                     fontsize=16, color=max_str_colors[i])
        ax.imshow(channels[i], aspect=50/6)

    energy = evt['weight']['InIceNeutrinoEnergy'][0]/1e3

    fig.suptitle(f'{energy:.0f} TeV, score {score:.2f}',
                 fontsize=24)


x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
z_hat = np.array([0, 0, 1])


def init_global_dom_map(json_file):
    _global_dom_map_holder[:1] = [DOMPositionMap(json_file)]


class DOMPositionMap:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            pos_df = pd.DataFrame(json.load(f))
            pos_df['deep_core'] = pos_df.string >= 79
            self._pos_df = pos_df

            underground_df = pos_df[pos_df.z < 1500]

            # underground, deep core and non deep core
            self._ugdf = underground_df
            self._dc = underground_df[underground_df.deep_core]
            self._ndc = underground_df[~underground_df.deep_core]

            # df with average string positions
            st_df = pos_df.groupby('string').mean()
            self._string_df = st_df
            self._st_pos_index = self.create_st_pos_index(st_df)

    @property
    def pos_df(self):
        return self._pos_df

    def plot_doms(self, ax,
                  color_str_nums=[], colors=max_str_colors):
        ndc, dc = self._ndc, self._dc

        # remove colored strings from ndc, dc
        ndc = self.remove_strings(ndc, color_str_nums)
        dc = self.remove_strings(dc, color_str_nums)

        ax.scatter(ndc.x, ndc.y, ndc.z, color='black', marker='o')
        ax.scatter(dc.x, dc.y, dc.z, color=deep_core_color, marker='o')

        # now plot the colored strings with their appropriate colors
        for str_num, color in zip(color_str_nums, colors):
            ugdf = self._ugdf
            str_df = ugdf[ugdf.string == str_num]
            ax.scatter(str_df.x, str_df.y, str_df.z, color=color, marker='o')

        return ax

    def plot_doms_projected_2d(self, x_basis, y_basis, ax,
                               color_str_nums=[], colors=max_str_colors):
        ''' make a projected 2d plot of the dom positions'''
        ndc, dc = self._ndc, self._dc

        # remove colored strings from ndc, dc
        ndc = self.remove_strings(ndc, color_str_nums)
        dc = self.remove_strings(dc, color_str_nums)

        ndcpos = ndc[['x', 'y', 'z']].to_numpy()

        dcpos = dc[['x', 'y', 'z']].to_numpy()

        ndc_proj = projected(ndcpos, [x_basis, y_basis])
        dc_proj = projected(dcpos,   [x_basis, y_basis])

        ax.plot(*ndc_proj.T, 'ko')
        ax.plot(*dc_proj.T, 'o', color=deep_core_color)

        for str_num, color in zip(color_str_nums, colors):
            ugdf = self._ugdf
            str_df = ugdf[ugdf.string == str_num]
            str_df_pos = str_df[['x', 'y', 'z']].to_numpy()
            str_df_proj = projected(str_df_pos, [x_basis, y_basis])

            ax.plot(*str_df_proj.T, 'o', color=color)

    def get_string_sq_dists(self, st_nums):
        '''returns the squares of the distances 
           between each string in st_nums and 
           the first string in st_nums

           returns: array of same length as st_nums
           first entry should always be 0
        '''
        st_nums = st_nums.astype(np.int16)

        st_posns = self._st_pos_index[st_nums-1]

        return ((st_posns - st_posns[0])**2).sum(axis=1)

    @staticmethod
    def remove_strings(df, remove_nums):
        ''' returns copy of df with the chosen strings removed'''
        keep_inds = reduce(lambda x, y: x & (df.string != y),
                           remove_nums,
                           np.ones(len(df), dtype=np.bool))

        return df[keep_inds]

    @staticmethod
    def create_st_pos_index(string_df):
        ''' creates a simple numpy array containing all avg string positions
            which can be used to quickly calculate string distances
        '''
        max_st = string_df.index.max()

        pos_array = string_df.loc[np.arange(1, max_st+1),
                                  ['x', 'y', 'z']].to_numpy()

        return pos_array


def projected(vector, basis_vectors):
    output_cols = np.empty((vector.shape[0], len(basis_vectors)))

    for i, basis_vector in enumerate(basis_vectors):
        output_cols[:, i] = (vector*basis_vector).sum(axis=1)

    return output_cols


def get_tracks(event):
    ''' returns (nu_tau_track, tau_track)
    the tracks are just arrays containing the start 
    and stop positions

    they can be easily plotted with plt.plot(*track)
    '''
    nu_tau = event['nu_tau']
    tau = event['tau']
    tau_product = event['tau_product']

    nu_tau_track = np.vstack((nu_tau['position'], tau['position'])).T
    tau_track = np.vstack((tau['position'], tau_product['position'])).T

    return nu_tau_track, tau_track


def get_image_st_nums(event, dist_sq_cut=default_distsq_cut):
    ''' returns string numbers used to generate the
        CNN image for this particular event
    '''
    # ensure we only look at strings within 150 meters of
    # the max string, because those are the ones that were
    # chosen to make the images that go into the CNN
    st_nums = event['q_st_dist']['st_nums']
    # cut out empty string records in the event
    st_nums = st_nums[st_nums != -1]

    dom_map = get_global_dom_map()
    dist_sqs = dom_map.get_string_sq_dists(st_nums)

    # keep the first three neighboring strings
    st_nums = st_nums[dist_sqs < dist_sq_cut**2][:3]

    return st_nums


def visualize_tau_3d(event, zoom=False, ax=None, angle=(25, -80),
                     nu_tau_track_len=800, dist_sq_cut=default_distsq_cut):
    if ax is None:
        ax = plt.figure(figsize=(12, 12)).add_subplot(111, projection='3d')

    nu_tau = event['nu_tau']
    tau = event['tau']
    tau_product = event['tau_product']

    st_nums = get_image_st_nums(event, dist_sq_cut)

    nu_track, tau_track = get_tracks(event)

    # do not allow the nu_tau track to extend too far out of the image...
    # that seems to cause issues with the 3d plotting
    nu_dir = nu_track[:, 1] - nu_track[:, 0]
    nu_dir /= math.sqrt(nu_dir@nu_dir)

    nu_track[:, 0] = nu_track[:, 1] - nu_tau_track_len*nu_dir

    dom_map = get_global_dom_map()
    ax = dom_map.plot_doms(ax, st_nums)

    ax.plot(*nu_track, 'm-', linewidth=3)
    ax.plot(*tau_track, 'm', linewidth=3, linestyle='dotted')
    ax.plot(*tau['position'][:, None], 'g*', markersize=10)
    ax.plot(*tau_product['position'][:, None], 'g*', markersize=10)

    ax._axis3don = False
    if not zoom:
        ax.set_xlim3d(-400, 400)
        ax.set_ylim3d(-400, 400)
        ax.set_zlim(-400, 400)
    else:
        center = tau_product['position']

        ax.set_xlim3d(center[0]-75, center[0]+75)
        ax.set_ylim3d(center[1]-75, center[1]+75)
        ax.set_zlim3d(center[2]-75, center[2]+75)

    ax.view_init(*angle)


def visualize_tau_2d(event, basis_vecs, zoom, ax,
                     dist_sq_cut=default_distsq_cut):
    '''plot a 2d projection of the event'''

    nu_tau = event['nu_tau']
    tau = event['tau']
    tau_product = event['tau_product']

    st_nums = get_image_st_nums(event, dist_sq_cut)

    nu_track, tau_track = get_tracks(event)

    projected_nu_track = projected(nu_track.T, basis_vecs).T

    projected_tau_track = projected(tau_track.T, basis_vecs).T

    dom_map = get_global_dom_map()
    dom_map.plot_doms_projected_2d(*basis_vecs, ax, st_nums)

    ax.plot(*projected_nu_track, 'm-', linewidth=3)
    ax.plot(*projected_tau_track, 'm', linewidth=3, linestyle='dotted')

    proj_positions = []

    for particle in [tau, tau_product]:
        pos = particle['position'][None, :]
        proj_pos = projected(pos, basis_vecs)

        ax.plot(*proj_pos.T, 'g*', markersize=10)

        proj_positions.append(proj_pos.flatten())

    if zoom:
        #         center = np.mean(proj_positions, axis=0)
        center = proj_positions[-1]

        ax.set_xlim(center[0] - 125, center[0] + 125)
        ax.set_ylim(center[1] - 125, center[1] + 125)
    else:
        ax.set_xlim(-700, 700)
        ax.set_ylim(-700, 700)


def get_track_basis(track):
    '''
    returns basis vectors such that the provided track is moving
    in the x-z plane

    returned basis vector ordering is x'_hat, z_hat, y'_hat
    where primed variables are in the new coordinate system
    '''
    direction = track[:, 1] - track[:, 0]

    theta = math.atan2(direction[1], direction[0])

    basis_x = np.array([math.cos(theta), math.sin(theta), 0])
    basis_y = np.array([-math.sin(theta), math.cos(theta), 0])

    basis_vecs = [basis_x, z_hat, basis_y]

    return basis_vecs


def draw_rotated_axes(x_prime_hat, y_prime_hat, ax,
                      origin=np.array([550, -550]),
                      length=150):
    ''' draws rotated axes on a 2d projected plot'''
    x_axis = np.array([origin, origin+length*x_prime_hat[:2]])
    y_axis = np.array([origin, origin+length*y_prime_hat[:2]])

    ax.plot(*x_axis.T, 'k-', linewidth=3)
    ax.plot(*y_axis.T, 'k-', linewidth=3)

    x_axis_label_pos = x_axis.mean(axis=0) - 50*y_prime_hat[:2]
    plt.text(*x_axis_label_pos, 'x\'',
             horizontalalignment='center',
             verticalalignment='center', fontdict={'fontsize': 14})

    y_axis_label_pos = y_axis.mean(axis=0) - 50*x_prime_hat[:2]
    plt.text(*y_axis_label_pos, 'y\'',
             horizontalalignment='center',
             verticalalignment='center', fontdict={'fontsize': 14})


def visualize_tau_event(event):
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(221, projection='3d')
    visualize_tau_3d(event, False, ax)

#     ax = fig.add_subplot(222, projection='3d')
#     visualize_tau_3d(event, True, ax, angle=(6, -45))

    nu_track, tau_track = get_tracks(event)
    track_basis = get_track_basis(nu_track)

    ax = fig.add_subplot(222)
#     ax = fig.add_subplot(223)
    # top down
    visualize_tau_2d(event, [x_hat, y_hat], False, ax)
    ax.set_xlabel('x [m]', fontsize=16)
    ax.set_ylabel('y [m]', fontsize=16)
    draw_rotated_axes(track_basis[0], track_basis[2], ax)

    ax = fig.add_subplot(223)
#     ax = fig.add_subplot(224)
    # with track in x'-z plane
    visualize_tau_2d(event, track_basis[:-1], True, ax)
    ax.set_xlabel('x\' [m]', fontsize=16)
    ax.set_ylabel('z [m]', fontsize=16)

    ax = fig.add_subplot(224)
    # with track in y'-z plane
    visualize_tau_2d(event, [track_basis[2], track_basis[1]], True, ax)
    ax.set_xlabel('y\' [m]', fontsize=16)
    ax.set_ylabel('z [m]', fontsize=16)