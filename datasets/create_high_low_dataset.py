import os
import random
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import shutil

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import torch
from facenet_pytorch import MTCNN
import torchvision
from torchvision import transforms

from common import load_model, load_data_by_class
from dataset_config import DatasetConfig
from utils import Rescale

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

mtcnn = MTCNN(image_size=160)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
composed_transforms = transforms.Compose([Rescale((160, 160)), normalize])


def get_embeddings(datapaths, model, model_type, model_embedding_size, use_mtcnn=False):
    embeddings = np.zeros((len(datapaths), model_embedding_size))
    for i, im_path in enumerate(datapaths):
        if use_mtcnn:
            try:
                img = Image.open(im_path)
                img = mtcnn(img)
            except:
                continue
        else:
            img = torchvision.io.read_image(im_path)

        image = composed_transforms(img)
        if model_type == 'pretrained_celeba':
            img_embedding = model(image.unsqueeze(0).float())[0]['classifier']
        else:
            img_embedding = model(image.unsqueeze(0).float())
        embeddings[i] = img_embedding.detach().numpy()
    return embeddings


def get_clusters_cores(X, unique_labels):
    cluster_means = []
    for label in unique_labels:
        cluster = X[X['cluster'] == label]
        cmean = cluster.mean(axis=0)
        cluster_means.append([cmean[0], cmean[1]])
    return cluster_means


def get_closest_cluster_mean(curr_core, cores):
    dists = []
    for i, core in enumerate(cores):
        # get Euclidean distance
        dist = np.linalg.norm(np.array(curr_core) - np.array(core))
        dists.append(dist)
    return np.argmin(dists)


def get_new_core_from_selected_clusters(selected_clusters, X):
    # from X df, get new df where cluster idx is in selected clusters
    combined_cluster_df = X[X['cluster'].isin(selected_clusters)]
    # get mean of new df
    new_core = combined_cluster_df.mean(axis=0)
    return [new_core[0], new_core[1]]


def get_pca(embeddings_list):
    pca = PCA(n_components=2)
    X = pca.fit_transform(embeddings_list)
    pca_df = pd.DataFrame(X)
    return pca_df


def get_dbscan_clustering(pca_df, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = dbscan.fit_predict(pca_df)
    pca_df['cluster'] = labels_db
    pca_df.columns = ['x1', 'x2', 'cluster']
    df_no_outliers = labels_db[labels_db != -1]
    return pca_df, df_no_outliers


def make_dirs(root_dir):
    dirs = []
    for set_var in ['low_var', 'high_var']:
        dirs.append(os.path.join(root_dir, set_var))
        for set_type in ['training', 'similar', 'dissimilar']:
            dirs.append(os.path.join(root_dir, set_var, set_type))
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def copy_images_to_folder(df, folder):
    for ind, row in df.iterrows():
        image_path = row['path']
        image_name = row['img_num']
        class_name = row['class']
        if not os.path.isdir(os.path.join(root_dir, folder, class_name)):
            os.mkdir(os.path.join(root_dir, folder, class_name))
        shutil.copy(image_path, os.path.join(root_dir, folder, class_name, f'{image_name}.jpg'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    parser.add_argument("-root_dir", "--root_dir", dest="root_dir", help="folder to save datasets")
    args = parser.parse_args()
    data_dir = args.data_dir
    root_dir = args.root_dir
    make_dirs(root_dir)
    config = DatasetConfig()
    my_model, model_embedding_size = load_model(model_type=config.model_type,
                                                model_weights=config.__getattribute__(f'{config.model_type}_weights'))

    # initalize dfs
    low_var_df_train = pd.DataFrame(columns=['class', 'img_num', 'path'])
    low_var_df_test_similar = pd.DataFrame(columns=['class', 'img_num', 'path'])
    low_var_df_test_dissimilar = pd.DataFrame(columns=['class', 'img_num', 'path'])
    high_var_df_train = pd.DataFrame(columns=['class', 'img_num', 'path'])
    high_var_df_test_similar = pd.DataFrame(columns=['class', 'img_num', 'path'])
    high_var_df_test_dissimilar = pd.DataFrame(columns=['class', 'img_num', 'path'])

    # create errors log
    errors_log = open(os.path.join(root_dir, 'errors_log.txt'), 'w')

    classes_list = os.listdir(data_dir)
    for class_name in tqdm(classes_list[:config.num_classes]):
        class_dir = os.path.join(data_dir, class_name)
        data_paths_list = load_data_by_class(class_dir)
        if len(data_paths_list) < config.min_instances_num:
            print(f'class does not contain enough images: {class_name}')
            errors_log.write(f'class does not contain enough images: {class_name}\n')
            continue

        embeddings_list = get_embeddings(data_paths_list, my_model, config.model_type, model_embedding_size)

        # low var training
        pca_df = get_pca(embeddings_list)
        eps_idx = config.__getattribute__(f'{config.model_type}_eps_idx')
        min_samples = config.__getattribute__(f'{config.model_type}_min_samples')
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(pca_df)
        distances, _ = nbrs.kneighbors(pca_df)
        distances = np.sort(distances[:, min_samples - 1], axis=0)
        # from distances, get epsilon where index is 100
        eps = distances[eps_idx]
        pca_df, labels_no_outliers = get_dbscan_clustering(pca_df, eps, min_samples)
        len_clusters_for_plot = len(labels_no_outliers)
        pca_df_for_plot = pca_df.copy()

        # Count the size of each cluster
        unique_labels = np.unique(labels_no_outliers)
        cluster_sizes = [(label, sum(pca_df['cluster'] == label)) for label in unique_labels]
        cluster_cores = get_clusters_cores(pca_df, unique_labels)

        # get the index of largest cluster
        largest_cluster_idx = max(range(len(cluster_sizes)), key=lambda i: cluster_sizes[i][1])

        largest_cluster = cluster_sizes[largest_cluster_idx]
        curr_size = largest_cluster[1]
        curr_core = cluster_cores[largest_cluster_idx]
        remaining_cores = cluster_cores.copy()
        remaining_cores.pop(largest_cluster_idx)
        remaining_clusters = cluster_sizes.copy()
        remaining_clusters.pop(largest_cluster_idx)
        selected_clusters = [largest_cluster_idx]

        while curr_size < config.total_set_size and len(remaining_cores) > 0:
            closest_cluster_idx = get_closest_cluster_mean(curr_core, remaining_cores)
            curr_size += remaining_clusters[closest_cluster_idx][1]
            selected_clusters.append(remaining_clusters[closest_cluster_idx][0])
            remaining_clusters.pop(closest_cluster_idx)
            remaining_cores.pop(closest_cluster_idx)
            curr_core = get_new_core_from_selected_clusters(selected_clusters, pca_df)

        if curr_size < config.total_set_size:
            print(f'not enough instances for class {class_name}, for low var training')
            errors_log.write(f'not enough instances for class {class_name}, for low var training\n')
            continue

        # append datapaths_list to pca_df as new column
        pca_df['path'] = data_paths_list
        pca_df['class'] = pca_df.apply(lambda x: x['path'].split('\\')[-2], axis=1)
        pca_df['img_num'] = pca_df.apply(lambda x: x['path'].split('\\')[-1].rstrip('.jpg'), axis=1)

        # get df of selected clusters
        curr_low_var_df = pca_df[pca_df['cluster'].isin(selected_clusters)]

        # split to train and test
        curr_low_var_df_train = curr_low_var_df.sample(n=config.train_set_size, random_state=200, replace=False)
        curr_low_var_df_test_similar = curr_low_var_df.drop(curr_low_var_df_train.index)
        curr_low_var_df_test_similar = curr_low_var_df_test_similar.sample(n=config.test_set_size, random_state=200,
                                                                           replace=False)

        # get dissimilar from outliers, -1
        curr_low_var_df_test_dissimilar = pca_df[pca_df['cluster'] == -1]
        other_clusters_df = pca_df[~pca_df['cluster'].isin(selected_clusters)]
        curr_low_var_df_test_dissimilar = pd.concat([curr_low_var_df_test_dissimilar, other_clusters_df])
        # get # LOW_VAR_TEST samples
        if len(curr_low_var_df_test_dissimilar) < config.test_set_size:
            print(f'not enough outliers for class {class_name}, for low var dissimilar test')
            errors_log.write(f'not enough outliers for class {class_name}, for low var dissimilar test\n')
        curr_low_var_df_test_dissimilar = curr_low_var_df_test_dissimilar.sample(n=config.test_set_size,
                                                                                 random_state=200, replace=False)

        # high var training
        # sample randomly 1 image from each cluster out of unique_labels, repeat until set size is reached
        curr_high_var_df = pd.DataFrame(columns=['class', 'img_num', 'path'])
        pca_df_no_outliers = pca_df.copy()
        pca_df_no_outliers = pca_df_no_outliers[pca_df_no_outliers['cluster'] != -1]
        while len(curr_high_var_df) < config.total_set_size and len(pca_df_no_outliers) > 0:
            for label in unique_labels:
                # check if there are samples left in current cluster
                if len(pca_df_no_outliers[pca_df_no_outliers['cluster'] == label]) == 0:
                    continue
                # sample one case randomly and remove it from df
                curr_high_var_df = pd.concat([curr_high_var_df,
                                              pca_df_no_outliers[pca_df_no_outliers['cluster'] == label].sample(n=1,
                                                                                                                random_state=200)])
                pca_df_no_outliers = pca_df_no_outliers.drop(
                    pca_df_no_outliers[pca_df_no_outliers['cluster'] == label].sample(n=1, random_state=200).index)

        if len(curr_high_var_df) < config.total_set_size:
            print(f'not enough instances for class {class_name}, for high var training')
            errors_log.write(f'not enough instances for class {class_name}, for high var training\n')
            continue

        curr_high_var_df_train = curr_high_var_df.sample(n=config.train_set_size, random_state=200, replace=False)
        curr_high_var_df_test_similar = curr_high_var_df.drop(curr_high_var_df_train.index)
        curr_high_var_df_test_similar = curr_high_var_df_test_similar.sample(n=config.test_set_size, random_state=200)

        # get dissimilar from outliers, -1
        curr_high_var_df_test_dissimilar = pca_df[pca_df['cluster'] == -1]
        # get # HIGH_VAR_TEST samples
        if len(curr_high_var_df_test_dissimilar) < config.test_set_size:
            print(f'not enough instances for class {class_name}, for high var test dissimilar')
            errors_log.write(f'not enough instances for class {class_name}, for high var test dissimilar\n')
            continue
        curr_high_var_df_test_dissimilar = curr_high_var_df_test_dissimilar.sample(n=config.test_set_size,
                                                                                   random_state=200, replace=False)

        if config.plot:
            plt.subplot(2, 2, 1)
            plt.scatter(pca_df['x1'], pca_df['x2'], c=pca_df['cluster'], cmap='Paired')
            plt.title(
                f'PCA with DBSCAN clustering, model type = {config.model_type} eps={np.round(eps, decimals=3)}, min_samples={min_samples}')
            # for x1, x2, cluster in zip(pca_df['x1'], pca_df['x2'], pca_df['cluster']):
            #     plt.annotate(cluster, (x1, x2))

            plt.subplot(2, 2, 3)
            plt.plot(distances)
            plt.hlines(y=eps, xmin=0, xmax=len(distances), colors='r', linestyles='dashed')
            plt.xlabel('Points')
            plt.ylabel('Distance')
            plt.title(
                f'k-dist graph, eps={np.round(eps, decimals=3)}, min_samples={min_samples}, num_train={len_clusters_for_plot}')
            plt.grid()
            plt.subplot(2, 2, 2)
            plt.scatter(pca_df['x1'], pca_df['x2'], c='black')
            plt.scatter(curr_low_var_df_train['x1'], curr_low_var_df_train['x2'], c='blue')
            plt.scatter(curr_low_var_df_test_similar['x1'], curr_low_var_df_test_similar['x2'], c='red')
            plt.scatter(curr_low_var_df_test_dissimilar['x1'], curr_low_var_df_test_dissimilar['x2'], c='green')
            plt.title(
                f'LOW VAR , model type = {config.model_type} eps={np.round(eps, decimals=3)}, min_samples={min_samples}')
            plt.legend(['all', 'train', 'similar', 'dissimilar'])
            plt.subplot(2, 2, 4)
            plt.scatter(pca_df['x1'], pca_df['x2'], c='black')
            plt.scatter(curr_high_var_df_train['x1'], curr_high_var_df_train['x2'], c='blue')
            plt.scatter(curr_high_var_df_test_similar['x1'], curr_high_var_df_test_similar['x2'], c='red')
            plt.scatter(curr_high_var_df_test_dissimilar['x1'], curr_high_var_df_test_dissimilar['x2'], c='green')
            plt.title(
                f'HIGH VAR, model type = {config.model_type} eps={np.round(eps, decimals=3)}, min_samples={min_samples}')
            plt.legend(['all', 'train', 'similar', 'dissimilar'])
            plt.show()

        # append all to final dfs
        low_var_df_train = pd.concat([low_var_df_train, curr_low_var_df_train])
        low_var_df_test_similar = pd.concat([low_var_df_test_similar, curr_low_var_df_test_similar])
        low_var_df_test_dissimilar = pd.concat([low_var_df_test_dissimilar, curr_low_var_df_test_dissimilar])
        high_var_df_train = pd.concat([high_var_df_train, curr_high_var_df_train])
        high_var_df_test_similar = pd.concat([high_var_df_test_similar, curr_high_var_df_test_similar])
        high_var_df_test_dissimilar = pd.concat([high_var_df_test_dissimilar, curr_high_var_df_test_dissimilar])

    # save to csv
    low_var_df_train.to_csv(os.path.join(root_dir, 'low_var', 'training', 'training.csv'), index=False)
    low_var_df_test_similar.to_csv(os.path.join(root_dir, 'low_var', 'similar', 'similar.csv'), index=False)
    low_var_df_test_dissimilar.to_csv(os.path.join(root_dir, 'low_var', 'dissimilar', 'dissimilar.csv'),
                                      index=False)

    high_var_df_train.to_csv(os.path.join(root_dir, 'high_var', 'training', 'training.csv'), index=False)
    high_var_df_test_similar.to_csv(os.path.join(root_dir, 'high_var', 'similar', 'similar.csv'), index=False)
    high_var_df_test_dissimilar.to_csv(os.path.join(root_dir, 'high_var', 'dissimilar', 'dissimilar.csv'),
                                       index=False)

    # for each df, copy the images to the relevant folder
    for set_var in ['low_var', 'high_var']:
        for df_type in ['training', 'similar', 'dissimilar']:
            df = pd.read_csv(os.path.join(root_dir, set_var, df_type, f'{df_type}.csv'))
            folder = os.path.join(root_dir, set_var, df_type)
            copy_images_to_folder(df, folder)
