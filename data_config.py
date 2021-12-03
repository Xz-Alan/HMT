class DataConfig:
    data_name = ""
    root_dir = ""
    data_str = []
    num_class = 0
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'Potsdam':
            CLASSES = ['black', 'impervious_surfaces', 'building', 'low_vegetation', 'tree', 'car', 'background']
            self.num_class = len(CLASSES)
            self.data_str = ['dsm', 'irrg', 'gts']
            self.root_dir = '../data/Potsdam/'
        elif data_name == 'Vaihingen':
            CLASSES = ['black', 'impervious_surfaces', 'building', 'low_vegetation', 'tree', 'car', 'background']
            self.num_class = len(CLASSES)
            self.data_str = ['dsm', 'top', 'gts']
            self.root_dir = '../data/Vaihingen/'
        elif data_name == 'GID':
            self.root_dir = '../data/GID/'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)