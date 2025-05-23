import yaml


def load_yaml(file: str, encoding: str = 'utf-8'):
    with open(file, encoding=encoding) as fp:
        config = yaml.safe_load(fp)

    return config


if __name__ == '__main__':
    print(load_yaml('../configs/dataset/cifar10.yaml'))
