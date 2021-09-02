import numpy as np

def relimage(angle, relangle):
    if (angle - relangle) > 180.0:
        angle = angle - 360
    elif (angle - relangle) < -180.0:
        angle = angle + 360
    return angle

def angles_to_feature(angle_list):
    '''
    take in a list of N-dim angles and transform to N*2 cosine features
    (Ex: 2-D phi/psi turns into 4-D cosine angles)

    Note that as is, it separates features 1 and 2 for each angle, but as long 
    as this is consistent, that's fine
    '''

    if type(angle_list) != np.ndarray:
        angle_list = np.array(angle_list)

    feature_1 = -np.sin(angle_list*np.pi/180)
    feature_2 = np.cos(angle_list*np.pi/180)
    full_features = np.hstack([feature_1, feature_2])

    return full_features

def transform_ang(ang):
    a = -np.sin(ang*np.pi/180)
    b = np.cos(ang*np.pi/180)
    return np.array([a,b])

def angs_to_feat_der(ang):
    if hasattr(ang, "__len__"):
        ders = []
        for n in ang:
            a = -np.sin(n*np.pi/180)
            b = np.cos(n*np.pi/180)
            ders.extend([a,b])
        return ders
    else:
        a = -np.sin(ang*np.pi/180)
        b = np.cos(ang*np.pi/180)
        return [a,b]

def rbf_kern(x, svm):
    SV = svm.support_vectors_
    coeffs = svm.dual_coef_ 
    gamma = svm.get_params()['gamma']
    rho = svm.intercept_[0]
    diff = x - SV
    k = np.sum(coeffs * np.exp(-gamma * np.sum(diff**2, axis=1))) + rho
    return k
    
def rbf_kern_der(x, svm):
    SV = svm.support_vectors_
    coeffs = svm.dual_coef_ 
    gamma = svm.get_params()['gamma']
    diff = x - SV
    d = (-2*gamma*coeffs * np.exp(-gamma * np.sum(diff**2, axis=1))).T * diff
    dk = np.sum(d, axis=0)
    return dk

def compute_dihed(coords):
    a, b, c, d = np.array(coords)
    v1 = (b - a)/(np.linalg.norm(b - a))
    v2 = -(c - b)/(np.linalg.norm(c - b))
    v3 = (d - c)/(np.linalg.norm(d - c))
    n1 = np.cross(v1,v2)
    n2 = np.cross(v2,v3)
    m = np.cross(n1,v2)
    x = np.dot(n1,n2)
    y = np.dot(m,n2)
    ang = 180.0 / np.pi * np.arctan2(y,x)
    return ang
    
def compute_dihed_der(coords, check=False):
    u = np.array(coords)
    v1 = u[0] - u[1]
    v2 = u[1] - u[2]
    v3 = u[3] - u[2]
    
    a = np.cross(v1,v2)
    b = np.cross(v3,v2)
    rasq = np.sum(a**2)
    ra2inv = 1.0/rasq
    rbsq = np.sum(b**2)
    rb2inv = 1.0/rbsq
    
    rg = np.linalg.norm(v2)
    rginv = 1.0/rg
    
    fg = np.dot(v1,v2)
    hg = np.dot(v3,v2)
    fga = fg*ra2inv*rginv
    hgb = hg*rb2inv*rginv
    gaa = -ra2inv*rg
    gbb = rb2inv*rg
    
    dtf = gaa*a
    dtg = fga*a - hgb*b
    dth = gbb*b
    
    f1 = dtf
    f2 = dtg - dtf
    f3 = -dtg - dth
    f4 = dth
    
    return [f1,f2,f3,f4]
