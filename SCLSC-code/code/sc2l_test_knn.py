# testing prediction
import argparse
from util import set_seed
from data_loading import *
from models import *
from visualization import *
import pickle
import pandas as pd


def predict(dataset_name, data_dir, model_dir, gene_set, encoder_model, device, lr, margin, bs, epoch, knn_k=10,output_dim=16):

    train_path = data_dir + "/" + dataset_name + "_train.h5ad"
    test_path  = data_dir + "/" + dataset_name + "_test.h5ad"
    cell_train = sc.read_h5ad(train_path)
    cell_test = sc.read_h5ad(test_path)

    if gene_set == "hvg":
        cell_train = cell_train[:, cell_train.var["highly_variable"]]
        cell_test = cell_test[:, cell_test.var["highly_variable"]]



    gene_dim = cell_test.X.toarray().shape[1]

    X_train, y_train = cell_train.X.toarray(),cell_train.obs.target
    X_test = cell_test.X.toarray()


    ##### model part ####
    print("@ Loading models encoder...")
    model_info = dataset_name + "_" + encoder_model + "_lr-" + str(lr) + "_epoch-" + str(epoch) +  "_batchSize-" + str(bs) +   \
    "_margin-" + str(margin) +  "_glist-" + gene_set + "_outputDim-" + str(output_dim)
    model_info = model_info +"all_sample_mean"

    if encoder_model == "MLP":
        #model = MLP(gene_dim,128,64,16)
        model = MLP(gene_dim,512,128,output_dim)
    elif encoder_model == "CNN":
        model = CNN(gene_dim, 7)

    model.load_state_dict(torch.load(model_dir +"/"+ model_info+".pt"))
    model.to(device)
    model.eval()

    # knn_model_filename = 'KNN_'+str(knn_k)+"_" + model_info
    # knn = pickle.load(open(model_dir + "/model/" + knn_model_filename + ".pkl", 'rb'))

    print("Fitting the KNN model for the prediction ... ")
    knn = KNeighborsClassifier(n_neighbors=args.knn_k)

    emb_X_train = project(model, X_train, device, encoder_model)
    knn.fit(emb_X_train, y_train)

    print(f"@ Calculating ACC using KNN {knn_k} on dataset [{dataset_name}] ...")

    emb_X_test = project(model, X_test, device, encoder_model)
    y_predict_test = knn.predict(emb_X_test)
    cell_test.obs["SCLSC_prediction"] = y_predict_test
    cell_test.obs.to_csv(model_info+".csv")
    # acc = metrics.accuracy_score(y_test, y_predict_test)
    # f1_score = metrics.f1_score(y_test, y_predict_test, average='macro')
    # print(f"[Test] Accuracy: {acc}, F1 score: {f1_score}")



if __name__ == "__main__":

        set_seed(100)
        parser = argparse.ArgumentParser(description='<Contrastive learning for the single-cell representation>')

        parser.add_argument('--encoder',       default="MLP", type=str, required=True, help='contrastive learning encoding model')
        # parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
        parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
        parser.add_argument('--data_dir', action="store",   type=str, required=True,  help="directory of data.")
        parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory of the saved model")
        parser.add_argument('--dataset_name', action="store",   type=str, required=True,  help="name of dataset.")
        parser.add_argument('--gene_set', default="all",   type=str, required=True,  help="the gene list used.[all/hvg/lvg].")

        parser.add_argument('--margin',     default=1.0,       type=float, required=True, help='Margins used in the contrastive training')
        parser.add_argument('--knn_k',     default=10,       type=int, required=False, help='k of knn used model the cell type prediction')
        parser.add_argument('--lr',             default=1e-3,   type=float, required=False, help='Learning rate')
        parser.add_argument('--epoch',      default=100,       type=int, required=False, help='Training epcohs')
        parser.add_argument('--batch_size' ,default=256,      type=int,  required=False, help="batch_size of the training.")
        parser.add_argument('--dropout'    ,default=0.1,      type=float,  required=False, help="Dropout rate.")
        parser.add_argument('--workers',     default=64,       type=int, required=False, help='number of worker for data loading')

        args = parser.parse_args()
        predict(args.dataset_name, args.data_dir, args.model_dir, args.gene_set, args.encoder, args.device, args.lr, args.margin, args.batch_size, args.epoch,  args.knn_k)

        # predict(dataset_name, data_dir, model_dir, gene_set, encoder_model, device, lr, margin, bs, epoch, knn_k=10)
