import yaml


def load_yaml(file, encoding='utf-8'):
    with open(file, encoding=encoding) as fp:
        config = yaml.safe_load(fp)

    return config
