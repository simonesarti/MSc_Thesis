def get_data_provider(**kwargs):
    
    dataset = kwargs["dataset"]
    
    if dataset == 'tiny_imagenet':
        from OFA_mbv3_extended.data_providers.tiny_imagenet import TinyImagenetDataProvider
        provider_class = TinyImagenetDataProvider
    elif dataset == 'imagenet':
        from .imagenet import ImagenetDataProvider
        provider_class = ImagenetDataProvider
    elif dataset == 'cifar10':
        from .cifar import CIFAR10DataProvider
        provider_class = CIFAR10DataProvider
    elif dataset == 'cifar100':
        from .cifar import CIFAR100DataProvider
        provider_class = CIFAR100DataProvider
    elif dataset == 'cinic10':
        from .cifar import CINIC10DataProvider
        provider_class = CINIC10DataProvider
    elif dataset == 'aircraft':
        from .aircraft import FGVCAircraftDataProvider
        provider_class = FGVCAircraftDataProvider
    elif dataset == 'cars':
        from .cars import StanfordCarsDataProvider
        provider_class = StanfordCarsDataProvider
    elif dataset == 'dtd':
        from .dtd import DTDDataProvider
        provider_class = DTDDataProvider
    elif dataset == 'flowers102':
        from .flowers102 import Flowers102DataProvider
        provider_class = Flowers102DataProvider
    elif dataset == 'food101':
        from .food101 import Food101DataProvider
        provider_class = Food101DataProvider
    elif dataset == 'pets':
        from .pets import OxfordIIITPetsDataProvider
        provider_class = OxfordIIITPetsDataProvider
    elif dataset == 'stl10':
        from .stl10 import STL10DataProvider
        provider_class = STL10DataProvider
    else:
        raise NotImplementedError

    provider = provider_class(
        save_path=kwargs['save_path'],
        train_batch_size=kwargs['train_batch_size'],
        test_batch_size=kwargs['test_batch_size'],
        valid_size=kwargs['valid_size'],
        n_worker=kwargs['n_worker'],
        image_size=kwargs['image_size'],
        resize_scale=kwargs["resize_scale"],
        distort_color=kwargs["distort_color"],
        num_replicas=kwargs["num_replicas"],
        rank=kwargs["rank"]
    )

    return provider
