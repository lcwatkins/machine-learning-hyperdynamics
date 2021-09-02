import numpy as np
import sklearn.cluster as skl
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from . import dihedrals as dih
from . import ramaplots as rm
#import dihedrals as dih
#import ramaplots as rm

'''
class to run and control Hyperdynamics (HD) method with dihedral angles
'''

class DihedralHD:
    def __init__(self, num_angles, sample_interval, num_simulations=1, scale=False):
        ''' initialize hyperdynamics method at beginning of run '''
        self.num_angles = num_angles
        self.data = []
        self.features = []
        self.traj_num_steps = 0
        self.sample_interval = sample_interval
        self.traj_len_time = self.traj_num_steps * self.sample_interval
        self.labels = []
        self.svm = None
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
        return

    def reset(self):
        self.data = []
        self.features = []
        self.traj_num_steps = 0
        self.traj_len_time = 0
        self.labels = []
        self.svm = None
        if self.scale:
            self.scaler = StandardScaler()
        return
        
    def update_data(self, new_data):
        '''
        Take in new trajectory data, add it to existing data
        new_data must be ndarray of shape [n_new_steps, num_angles]

        Get cos/sin features for dihedrals

        Note: if using multiple concurrent simulations, this loses that information
        '''

        # verify data has the right # of angles 
        if new_data.shape[1] != self.num_angles:
            print("New data does not have correct shape")
            exit()

        # Add new data to all data and features
        if len(self.data) == 0:
            self.data = np.array(new_data)
        else:
            self.data = np.concatenate([self.data, new_data])
        self.update_features(new_data)

        # Update total number of steps and total time length of simulation
        self.traj_num_steps = len(self.data)
        self.traj_len_time = self.traj_num_steps * self.sample_interval

        return

    def update_features(self, new_data):
        '''
        Turn dihedral angles into features used for clustering 
        '''
        new_features = []
        for dihedral_angles in new_data:
            angle_features = dih.angles_to_feature(dihedral_angles)
            new_features.append(angle_features)
        
        if len(self.features) == 0:
            self.features = np.array(new_features)
        else:
            self.features = np.concatenate([self.features, new_features])

        return
    
    def cluster(self, e=0.15, mn=15, k='euclidean'):
        '''
        Use DBSCAN to cluster all sampled points

        e, eps: maximum distance between two samples for one to be considered
            as in the neighborhood of the other

        mn, min_samples: number of samples in a neighborhood for a point to
            be considered as a core point
        '''
        db = skl.DBSCAN(metric=k, eps=e, min_samples=mn, n_jobs=4).fit(self.features)
        self.labels = db.labels_

        return

    def svm1(self, k='rbf', g=0.5, n=0.01):
        '''
        Use OneClassSvm unsupervised outlier detection method to determine the boundary
        around current region, based on points sampled in simulation so far
        
        kernel: radial basis function (rbf), exp(-gamma*||x-x'||^2)
            - need a non-linear kernel
            - similar to gaussian function, which is ideal

        g, gamma: rbf kernel coefficient
            - "defines how much influence a single training example has" --skl guide

        n, nu: upper bound on the fraction of training errors, lower bound of the fraction
            of support vectors. Range [0,1]
            - we want error to be low. We want the SVM to precisely define the region
                by the given points -- however, we don't want to make the region larger than
                it actually is and accidentally bias in transition regions. To be careful,
                we use a non-zero value for nu to slightly shrink the region


        Use last label to determine the current region of the simulation
        '''

        labels = self.labels
        reg = labels[-1]
        if reg == -1:
            print("in no-man's land")
            self.svm = None
            return

        # Relabel to make current state always 0
        if reg != 0:
            labels = [reg if i==0 else 0 if i==reg else i for i in labels]
            reg = 0
        self.labels = labels

        instate = [self.features[i] for i in range(len(labels)) if labels[i] == reg]
        instate = np.array(instate)

        # Scale features within region that will be used to fit SVM
        if self.scale:
            instate = self.scaler.fit_transform(instate)

        self.svm = svm.OneClassSVM(kernel=k, gamma=g, nu=n).fit(instate)

        return

    def update_boundary(self, cluster_e=0.15, cluster_mn=15, svm_g=0.8, svm_n=0.1):
        if len(self.data) < 5:
            return
        self.cluster(e=cluster_e, mn=cluster_mn)
        self.svm1(g=svm_g, n=svm_n)
        return

    def plot_trajectory(self):
        if self.num_angles != 2:
            print("Only implemented for 2 angles")
            return

        fig, ax = rm.setup_one(self.traj_len_time)
        if len(self.data) > 0:
            rm.scat(fig, ax, self.data)

        return


    def plot_boundary(self, space=3, time=False):
        '''
        Plot the clustering result and the SVM result in two subplots

        Currently only written to handle two dihedral angles, because it's harder to
        clearly visualize in more dimensions
        '''

        if self.num_angles != 2:
            print("Only implemented for 2 angles")
            return

        fig, axes = rm.setup_two()
        x = np.arange(-180,180,space)
        y = np.arange(-180,180,space)
        X,Y = np.meshgrid(x,y)

        t = None
        if time is True:
            t = self.traj_len_time
        if self.svm is not None:
            cosXY = dih.angles_to_feature(np.stack([X.flatten(),Y.flatten()], axis=-1))
            if self.scale:
                cosXY = self.scaler.transform(cosXY)
            Z = (self.svm.decision_function(cosXY))/len(self.svm.support_)*100
            Z = Z.reshape(X.shape)
            rm.scat2dists(fig, axes, self.data, self.labels, X, Y, Z, t=t)
        elif len(self.data) > 0:
            rm.scat2dists(fig, axes, self.data, self.labels, t=t)

        return




