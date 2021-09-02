class Trajs:,
    def __init__(self, data, num_trajs, time_length, time_step):
        self.data = np.array(data)
        self.num = num_trajs
        self.angs = len(data[0])
        self.traj_len_time = time_length
        self.traj_len_steps = len(data)/num_trajs
        if self.traj_len_steps != time_length/time_step + 1:
            print("check number of steps per traj")
        self.timestep = time_step
        self.labels = {}
        self.svms = {}
        self.scalers = {}
        self.get_cosdat()
        
    def get_cosdat(self):
        cosdat = []
        for point in self.data:
            new = []
            for ang in point:
                new.extend(transform_ang(ang))
            cosdat.append(new)
        self.cosdat = np.array(cosdat)
    
    def get_scaledat(self, data):
        scaler = StandardScaler().fit(data)
        scaledat = scaler.transform(data)
        return scaledat, scaler
        
    def cluster(self, e, mn, time=-1, scaled=True):
        if time != -1:
            nsteps = int(time/self.timestep)
            data = self.cosdat[:nsteps]
        else:
            data = self.cosdat
        if scaled:
            data = self.get_scaledat(data)
        db = skl.DBSCAN(eps=e, min_samples=mn,n_jobs=4).fit(data)
        self.labels['cluster' + str(time)] = db.labels_
            

    def extend_clusters(self, labels_to_use):
        new_labels = []
        for n in range(self.num):
            step0 = n*self.traj_len_steps
            stepf = (n+1)*self.traj_len_steps
             
            ntraj_labels = self.labels[labels_to_use][step0:stepf]
            prev_reg = -1
            new_ntraj_labels = [k for k in ntraj_labels]
            i = 0
            while i < len(ntraj_labels):
                reg = ntraj_labels[i]
                if reg != -1:
                    out_steps = 0
                    prev_reg = reg
                    i += 1
                elif reg == -1:
                    j = i
                    nextreg = ntraj_labels[j]
                    while nextreg == -1:
                        out_steps += 1
                        j += 1
                        nextreg = ntraj_labels[j]
                    if nextreg == prev_reg:
                        #print \"change\ i, \"to\ j"(
                        new_ntraj_labels[i:j] = [nextreg for k in range(i,j)]
                    i = j
            new_labels.extend(new_ntraj_labels)
        self.labels['extend'] = new_labels

    def merge_regions(self, labels_to_use, a, b):
        labels = self.labels[labels_to_use]
        # Shift region labels so none are empty
        switch = {}
        same = min(a,b)
        move = max(a,b)
        for i in range(-1,move):
            switch[i] = i
        switch[move] = same
        for i in range(move+1, max(labels)+1):
            switch[i] = i-1

        # Now renumber labels
        newlabels = []
        for l in labels:
            newlabels.append(switch[l])
        self.labels['merge'] = newlabels
        
    def svm1(self, labels_to_use, g=0.5, n=0.00001):
        labels = self.labels[labels_to_use]
        instate = [self.cosdat[i] for i in range(len(labels)) if labels[i] == 0]
        #instate, scaler = self.get_scaledat(instate)
        self.svms[labels_to_use] = svm.OneClassSVM(gamma=g,nu=n).fit(instate)
        #self.scalers[labels_to_use] = scaler"
