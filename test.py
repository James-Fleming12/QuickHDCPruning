from specific.mnist_model import train_mnist_cnn, train_mnist_hdc, prune_mnist, eval_mnist
from specific.isolet import train_isolet_extractor, train_isolet_hdc, prune_isolet, eval_isolet, micro_isolet
from specific.sensor_model import train_har_extractor, train_har_hdc, prune_har, eval_har
from specific.other_micro import train_fmnist

def mnist_pipeline():
    # train_mnist_cnn()
    # train_mnist_hdc(5000, pruned=False)
    pruned_dim = prune_mnist()
    hdc = train_mnist_hdc(pruned_dim, pruned=True)
    eval_mnist(hdc)
    # test_model_consistency(pruned_dim)

def isolet_pipeline():
    # train_isolet_extractor()
    # train_isolet_hdc(5000, pruned=False)
    pruned_dim = prune_isolet()
    # micro_dim = micro_isolet()
    hdc = train_isolet_hdc(pruned_dim, pruned=True)
    # eval_isolet(hdc)

def main():
    # mnist_pipeline()
    isolet_pipeline()

if __name__=="__main__":
    main()