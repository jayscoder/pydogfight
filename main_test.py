import yaml

if __name__ == '__main__':
    data = yaml.safe_load('x: 1\ny: 10')
    print(data)
