import cPickle as pk
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


with open('data/na_bin.pkl', 'r') as f:
    na_bin = pk.load(f)

'''
# estimate n_comp
pca = PCA()
pca.fit(na_bin)
print(pca.explained_variance_ratio_) 
'''

'''
print 'pca'
pca = PCA(n_components=5)
new_na_bin = pca.fit_transform(na_bin)
print new_na_bin
'''

print 'trunc svd'
tsvd = TruncatedSVD(n_components=2, random_state=42)
new_na_bin = tsvd.fit_transform(na_bin)
print new_na_bin



with open('data/na_bin_tsvd_2.pkl', 'w') as f:
    pk.dump(new_na_bin, f)


