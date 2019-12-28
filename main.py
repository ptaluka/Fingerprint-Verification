#%% [markdown]
# # Loading and preprocessing of fingerprint images.

#%%
# Directory structure
# Fingerprint
#  |
#  |_____ Real
#  |
#  |_____ Altered
#           |
#           |_____ Altered-Easy
#           |
#           |_____ Altered-Medium
#           |
#           |_____ Altered-Hard
import os
try:
    path_main = os.getcwd()
    test_diff = 'Easy'
    path_test = '{}\\data\\Fingerprint\\Altered\\Altered-{}'.format(path_main, test_diff)
    path_train = '{}\\data\\Fingerprint\\Real'.format(path_main)
    os.chdir(path_train)
except:
    pass

#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
path_imgs = pd.Series(os.listdir())
id = path_imgs.map(lambda x: x.replace('.BMP', ''))

os.chdir(path_test)
path_test_imgs = pd.Series(os.listdir())
test_id = path_test_imgs.map(lambda x: x.replace('.BMP', ''))
os.chdir(path_train)

print(id.shape)
print(id.head())
print(test_id.shape)
print(test_id.head())

#%%
def get_image(idx, path):
    img = Image.open(path[idx]).convert(mode='L')
    img = img.crop((0, 0, 100, 100))
    
    return np.maximum(np.array(img), np.full((100, 100), np.uint8(100)))

#%%
def plot_kde(channel, color):
    data = channel.flatten()
    return pd.Series(data).plot.density(c=color)

plot_kde(get_image(0, path_imgs), 'w')

#%% [markdown]
# ## Generate the feature matrix from the images.

#%%
from skimage.feature import hog

def gen_features(img):
    vis_features = img.flatten()
    hog_features = hog(img, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    return np.concatenate((vis_features, hog_features), axis=0)

def gen_feature_matrix(label, imgs):
        features_list = []
        for idx in label.index:
            features_list.append(gen_features(get_image(idx, imgs)))
        return np.array(features_list)

feature_matrix = gen_feature_matrix(id, path_imgs)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)

#%% [markdown]
# ## Apply PCA to reduce dimensions

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=500)
pca.fit(scaled_features)
#%%
com_img = pca.components_[0, :].reshape(-1, 100)
com_img = com_img[:100, :]
plt.imshow(com_img, cmap='gray')
plt.show()

#%%
lim = -1
plt.bar(range(pca.n_components_)[:lim], pca.explained_variance_[:lim])
plt.show()

X = pd.DataFrame(pca.transform(scaled_features), index=id)
#%% [markdown]
# ## Verification process

#%%
curr_id = 9
curr_feature = X.iloc[curr_id, :]
similarities = X.dot(curr_feature)

print('Selected print:\n{}\n'.format(id[curr_id]))
print('Predicted prints:\n{}'.format(similarities.nlargest()))

def ver_img(idx):
    return X.dot(X.iloc[idx, :]).nlargest(1).index[0]

ver_img(9)

pred_labels = id.index.map(ver_img).values
y = id.values

#%%
from sklearn.metrics import accuracy_score
score = accuracy_score(y, pred_labels)
print(score)

#%% [markdown]
# # Testing

#%%
os.chdir(path_test)
test_idx = 954
test_img = get_image(test_idx, path_test_imgs)

print(test_id[test_idx])
plt.imshow(test_img, cmap='gray')
plt.show()

test_feature = gen_features(test_img)
scaled_test_feature = scaler.transform(test_feature.reshape(1, -1))
X_test = pd.DataFrame(pca.transform(scaled_test_feature))
pred_label = X.dot(X_test.iloc[0, :]).nlargest(1).index[0]

img_idx = id[id == pred_label].index[0]
os.chdir(path_train)
pred_img = get_image(img_idx, path_imgs)

print(pred_label)
plt.imshow(pred_img, cmap='gray')
plt.show()

os.chdir(path_main)