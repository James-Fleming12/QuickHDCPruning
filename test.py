from specific.mnist_model import train_mnist_cnn, train_mnist_hdc, prune_mnist, eval_mnist

def main():
    # train_mnist_cnn()
    # train_mnist_hdc(5000, pruned=False)
    pruned_dim = prune_mnist()
    hdc = train_mnist_hdc(pruned_dim, pruned=True)
    eval_mnist(hdc)
    # test_model_consistency(pruned_dim)

if __name__=="__main__":
    main()