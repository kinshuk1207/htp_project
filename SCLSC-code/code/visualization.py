# visualization
import matplotlib.pyplot as plt
import umap
import pandas as  pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from models import project
import seaborn as sns

def util_plot(X, y, ax,cmaps,marker=None,s=0.5):
    for i in np.unique(y):
        ax.scatter(X[y == i, 0],
                   X[y == i, 1],
                   label=i,
                   marker=marker,
                   color=cmaps[i],
                   alpha=0.8,s=s)


# def umap_plot_batch(encoder_model, X, group_batch, save_path, model_info, encoder, device, data_profile):

#     f = plt.figure(figsize=(16,8), dpi=300)
#     ax_1 = f.add_subplot(121)
#     ax_2 = f.add_subplot(122)

#     ax_1.set_title('(batch) UMAP of raw data')
#     ax_2.set_title('(batch) UMAP of embedded data (CL+MLP)')

#     # The UMAP representation of training data embedding
#     umap_reducer = umap.UMAP()
#     X_train_umap = umap_reducer.fit_transform(X)
#     ax_1.scatter(X_train_umap[:, 0],X_train_umap[:, 1],c=group_batch,alpha=0.8,s=0.5)

#     # The UMAP representation of embedded training data embedding
#     emb_X_train = project(encoder, X, device, encoder_model)
#     umap_reducer = umap.UMAP()
#     emb_X_train_umap = umap_reducer.fit_transform(emb_X_train)
#     ax_2.scatter(emb_X_train_umap[:, 0],emb_X_train_umap[:, 1], c=group_batch, alpha=0.8,s=0.5)

#     f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")


# def umap_plot_celltype(encoder_model, X, y, save_path, model_info, encoder, device, data_profile):

#   f = plt.figure(figsize=(16,16), dpi=300)
#   ax_1 = f.add_subplot(121)
#   ax_2 = f.add_subplot(122)

#   ax_1.set_title('UMAP  of raw data')
#   ax_2.set_title('UMAP  of embedded data (CL+MLP)')

#   # The UMAP representation of training data embedding
#   umap_reducer = umap.UMAP()
#   X_train_umap = umap_reducer.fit_transform(X)
#   ax_1.scatter(X_train_umap[:, 0],X_train_umap[:, 1],c=y,alpha=0.8,s=0.5)

# 	# The UMAP representation of embedded training data embedding
#   emb_X_train = project(encoder, X, device, encoder_model)
#   umap_reducer = umap.UMAP()
#   emb_X_train_umap = umap_reducer.fit_transform(emb_X_train)
#   ax_2.scatter(emb_X_train_umap[:, 0], emb_X_train_umap[:, 1], c=y, alpha=0.8,s=0.5)

#   f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")



def umap_plot_batch(encoder_model, X_train, X_test, y_train, y_test,
                    save_path, model_info, encoder, device, data_profile):

    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16),sharey=True)

    ax_1=axes[0,0]
    ax_2=axes[0,1]
    ax_3=axes[1,0]
    ax_4=axes[1,1]

    ax_1.set_title('(batch) UMAP  of raw training data')
    ax_2.set_title('(batch) UMAP  of raw test data')

    ax_3.set_title('(batch) UMAP  of embedded training data (CL+MLP)')
    ax_4.set_title('(batch) UMAP  of embedded test data (CL+MLP)')



    # The UMAP representation of training data embedding
    umap_reducer = umap.UMAP().fit(np.vstack([X_train,X_test]))

    X_train_umap = umap_reducer.transform(X_train)
    unique_target = np.unique(np.concatenate((y_train,y_test)))
    cmaps=sns.color_palette("husl",len(unique_target))
    util_plot(X_train_umap,y_train,ax_1,cmaps)

    X_test_umap = umap_reducer.transform(X_test)
    util_plot(X_test_umap,y_test,ax_2,cmaps)

    # The UMAP representation of embedded training data embedding
    emb_X_train = project(encoder, X_train, device, encoder_model)
    emb_X_test = project(encoder, X_test, device, encoder_model)

    umap_reducer = umap.UMAP().fit(np.vstack([emb_X_train,emb_X_test]))
    emb_X_train_umap = umap_reducer.transform(emb_X_train)
    util_plot(emb_X_train_umap,y_train,ax_3,cmaps)

    emb_X_test_umap = umap_reducer.transform(emb_X_test)
    util_plot(emb_X_test_umap,y_test,ax_4,cmaps)

    plt.tight_layout()
    f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" +".png")
    return umap_reducer


def umap_plot_celltype(encoder_model, X_train, X_test, y_train, y_test,X_rep,y_rep,
                      save_path, model_info, encoder, device, data_profile):

  f, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7),sharey=True,sharex=True)

  ax_1=axes[0]
  ax_2=axes[1]
  ax_3=axes[2]


  ax_1.set_title('UMAP  of raw data')
  ax_2.set_title('UMAP  of embedded data')
  ax_3.set_title('UMAP  of representative samples')





  # The UMAP representation of training data embedding
  X = np.vstack([X_train,X_test])
  y = np.concatenate((y_train,y_test))

  unique_target = np.unique(y)
  if len(unique_target) <= 10:
    cmap = "tab10"
    cmaps = sns.color_palette(cmap)
  elif len(unique_target) <= 20:
    cmap = "tab20"
    cmaps = sns.color_palette(cmap)
  else:
    cmap = "husl"
    cmaps = sns.color_palette(cmap, len(unique_target))

  umap_reducer = umap.UMAP().fit(X)
  X_umap = umap_reducer.transform(X)
  util_plot(X_umap,y,ax_1,cmaps)

  # The UMAP representation of embedded training data embedding
  emb_X = project(encoder, X, device, encoder_model)
  emb_X_rep = project(encoder, torch.flatten(X_rep,0,1), device, encoder_model)
  last_emb_X_rep = project(encoder, X_rep[-1,:,:], device, encoder_model)
  umap_reducer = umap.UMAP().fit(np.vstack([emb_X,emb_X_rep]))
  emb_X_umap = umap_reducer.transform(emb_X)
  emb_X_rep_umap = umap_reducer.transform(emb_X_rep)
  last_emb_X_rep_umap= umap_reducer.transform(last_emb_X_rep)
  util_plot(emb_X_umap,y,ax_2,cmaps)
  util_plot(emb_X_rep_umap,y_rep.flatten(),ax_3,cmaps)
  util_plot(last_emb_X_rep_umap,y_rep[-1,:],ax_3,cmaps,s=5,marker="X")
  plt.tight_layout()
  f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell" + ".png")
