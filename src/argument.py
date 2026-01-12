import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="pubmed", help="cora, citeseer, pubmed, computers, photo")
    
    # masking
    parser.add_argument("--label_rate", type=float, default=0.03)
    parser.add_argument("--folds", type=int, default=20)

    # Encoder
    parser.add_argument("--layers", nargs='+', default='[256, 256]', help="The number of units of each layer of the GNN. Default is [256]")
    
    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=200, help="The number of epochs")
    parser.add_argument("--lr", '-lr', type=float, default=0.1, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=5e-4, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=200)
    
    # hyper-parameter
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--thres", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--lam2", type=float, default=1)
    parser.add_argument("--lam3", type=float, default=0.5)
    parser.add_argument("--lam4", type=float, default=0.5)

    ## yalda added
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.3)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.1)


    # augmentation
    parser.add_argument("--df_1", type=float, default=0.5)
    parser.add_argument("--de_1", type=float, default=0.5)
    parser.add_argument("--df_2", type=float, default=0.2)
    parser.add_argument("--de_2", type=float, default=0.2)

    # specific for mix
    parser.add_argument('--gamma', type=float, default=0.5, help="threshold for pseudo labeling")
    parser.add_argument('--beta_s', type=float, default=0.5,
                        help="tuning strength of NLP similarity in NLD-aware sampling")
    parser.add_argument('--beta_d', type=float, default=0.5,
                        help="tuning strength of node degree in NLD-aware sampling")
    parser.add_argument('--temp', type=float, default=0.5, help="sharpness of distribution")
    parser.add_argument('--mixup_alpha', type=float, default=0.5, help="determing the Beta distribution")
    parser.add_argument('--lam_intra', type=float, default=0.1, help="balance hyperparameter of intra-class mixup loss")
    parser.add_argument('--lam_inter', type=float, default=0.1, help="balance hyperparameter of inter-class mixup loss")

    parser.add_argument("--device", '-d', type=int, default=0, help="GPU to use")
    
    return parser.parse_known_args()[0]
