'''data.py
Reads CSV files, stores data, access/filter data by variable name
Trey Tuscai
CS 251 Data Analysis and Visualization
Spring 2023
'''
import csv
import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        if filepath is not None:
            self.read(filepath)
        pass

    def read(self, filepath):
        self.header2col = {}
        self.data = []
        self.filepath = filepath
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            try:
                numeric_indices = [i for i, x in enumerate(rows[1]) if "numeric" in x]
            except IndexError:
                raise TypeError("The necessary type information is missing in the file.")
            if not numeric_indices:
                raise TypeError("No numeric columns found in the file.")
            self.headers = [rows[0][i] for i in numeric_indices]
            for i, header in enumerate(self.headers):
                self.header2col[header.strip()] = i
            for row in rows[2:]:
                    numeric_row = [float(row[i]) for i in numeric_indices]
                    self.data.append(numeric_row)
        self.data = np.array(self.data)
        pass

    def __str__(self):
        row_count, column_count = self.data.shape
        row_num = 5
        header_str = 'Headers:\n  ' + '    '.join(self.headers)
        data_str = '\n'.join(['    '.join(map(str, row)) for row in self.data[:5]])
        if row_count < 5:
            row_num = row_count
        return (f'-------------------------------\n'
                f'{self.filepath} ({row_count}x{column_count})\n'
                f'{header_str}\n'
                f'-------------------------------\n'
                f'Showing first {row_num}/{row_count} rows.\n'
                f'{data_str}\n'
                f'-------------------------------')

    def get_headers(self):
        return self.headers

    def get_mappings(self):
        return self.header2col

    def get_num_dims(self):
        return len(self.headers)

    def get_num_samples(self):
        return self.data.shape[0]

    def get_sample(self, rowInd):
        return self.data[rowInd, :]

    def get_header_indices(self, headers):
        header_indices = [self.header2col[header] for header in headers]
        return header_indices

    def get_all_data(self):
        return np.copy(self.data)

    def head(self):
        return self.data[:5,:]

    def tail(self):
        return self.data[-5:, :]

    def limit_samples(self, start_row, end_row):
        self.data = self.data[start_row:end_row]
        self.num_samples = end_row - start_row
        pass

    def select_data(self, headers, rows=[]):
        header_indices = self.get_header_indices(headers)
        if len(rows) == 0:
            return self.data[:, header_indices]
        else:
            return self.data[np.ix_(rows, header_indices)]