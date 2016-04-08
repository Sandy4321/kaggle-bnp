import cPickle as pk

from sklearn.decomposition import PCA

with open('data/na_bin.pkl', 'r') as f:
    na_bin = pk.load(f)

'''
# estimate n_comp
pca = PCA()
pca.fit(na_bin)
print(pca.explained_variance_ratio_) 
'''

pca = PCA(n_components=5)

new_na_bin = pca.fit_transform(na_bin)


print new_na_bin

with open('data/na_bin_pca.pkl', 'w') as f:
    pk.dump(new_na_bin, f)


