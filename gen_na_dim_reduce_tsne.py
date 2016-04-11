import cPickle as pk
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


with open('data/na_bin.pkl', 'r') as f:
    na_bin = pk.load(f)

'''
# estimate n_comp
pca = PCA()
pca.fit(na_bin)
print(pca.explained_variance_ratio_) 
'''

print 'pca'
pca = PCA(n_components=5)
na_bin = pca.fit_transform(na_bin)

print 'tsne'
tsne = TSNE(n_components=2, random_state=0)
new_na_bin = tsne.fit_transform(na_bin)


print new_na_bin

with open('data/na_bin_tsne.pkl', 'w') as f:
    pk.dump(new_na_bin, f)


