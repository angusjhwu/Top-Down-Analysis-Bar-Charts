# ================================
# Author: Angus Wu
# GitHub: angusjhwu
# Disclaimer:   This code was written for personal use, and open-sourced for your convenience.
#               The code may not be thoroughly tested, and may therefore produce erroneous charts.
#               Please use with caution. Otherwise, enjoy :)
# ================================

import sys
import os
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from math import ceil

class TopDownHierarchy:
    # Init
    def __init__(self, hierarchy: str = None) -> None:
        self._metric_id2name:   list[str]       = ['Top_Level']
        self._metric_name2id:   dict[str, int]  = {'Top_Level': 0}
        self._metric_parent:    list[int]       = [None]
        self._metric_value:     list[float]     = [None]
        self.runtime_ms: int = None
        
        self.perflist_path = f'{os.path.dirname(os.path.realpath(__file__))}/perflist'
        if not os.path.exists(self.perflist_path):
            self._init_perf()

        self._hierarchy = open(self.perflist_path).read()
        
        self._init_id()
        self.num_metrics: int = len(self._metric_id2name)
        self._init_parent()
        
    def _init_perf(self) -> None:
        with open(self.perflist_path + 'temp', 'w') as f:
            subprocess.run(["perf", "list"], stdout=f)

        dump: str = open(self.perflist_path + 'temp', 'r').read()
        
        def search_metric(metric: str, hierarchy: list[str], level=-1) -> None:
            if level >= 0:
                hierarchy.append('- ' * level + metric)
                
            pattern = metric + r"_group:(.*?)(\n{2}|\Z)"
            match = re.search(pattern, dump, re.DOTALL)
            if not match:
                return
            block = match.group(1)

            submodules: list[str] = []
            for b in block.split('\n'):
                if not b.startswith("  tma_"):
                    continue
                search_metric(b.strip(), hierarchy, level=level+1)
            
        search_term = 'tma_L1'  # this may change between processors/architectures
        search_hierarchy: str = []
        search_metric(search_term, search_hierarchy)
        
        with open(self.perflist_path, 'w') as f:
            f.write('\n'.join(search_hierarchy))
        
        os.remove(self.perflist_path + 'temp')
             
    def _init_id(self) -> None:
        cur_level = 1
        has_metric = True
        
        while has_metric:
            has_metric = False
            for metric in self._hierarchy.split('\n'):
                level = metric.count('-') + 1
                metric_name = metric[metric.rfind('-')+1:].strip()
                if level == cur_level:
                    self._metric_name2id[metric_name] = len(self._metric_id2name)
                    self._metric_id2name.append(metric_name)
                    has_metric = True
            cur_level += 1
    
    def _init_parent(self) -> None:
        self._metric_parent = [None] * (self.num_metrics + 1) # for Top_Level
        hierarchy = []
        for metric in self._hierarchy.split('\n'):
            if metric == '':
                continue
            
            level = metric.count('-') + 1
            metric = metric.replace('-', '').replace(' ', '')

            id = self.get_metric_id(metric)
            while len(hierarchy) >= level:
                hierarchy.pop()
                    
            if len(hierarchy) == 0:
                self._metric_parent[id] = 0
                hierarchy.append(id)
                continue
            self._metric_parent[id] = hierarchy[-1]
            hierarchy.append(id)
    
    def __repr__(self) -> str:
        repr_str = 'TopDownHierarchy:\n'
        for id, metric in enumerate(self._metric_id2name):
            repr_str += f'    [{id:2d}] {metric} - parent[{self.get_parent_id(id)}]\n'
        return repr_str
         
    # Checkers
    def check_id_valid(self, id: int) -> None:
        if id < 0 or id > self.num_metrics:
            raise RuntimeError(f'Metric ID [{id}] is out of range [0, {self.num_metrics}]')
        
    # Accessors      
    def get_metric_id(self, metric) -> int:
        """Return the metric ID given an ID or name. The none case will be treated as Top_Level (id=0)

        Args:
            metric (_type_): Metric name or ID

        Raises:
            RuntimeError: Metric name not found
            RuntimeError: Input is not int or str

        Returns:
            int: Metric ID
        """
        if metric is None:
            return 0
        elif type(metric) == int:
            self.check_id_valid(metric)
            return metric
        elif type(metric) == str:
            if metric not in self._metric_name2id:
                raise RuntimeError(f'Metric [{metric}] not found')
            return self._metric_name2id[metric]
        else:
            raise RuntimeError(f'Metric [{metric}] is of invalid type {type(metric)}')
    
    def get_metric_name(self, metric) -> str:
        """Return the metric name given an ID or name. The none case will be treated as 'Top_Level'

        Args:
            metric (_type_): _description_

        Raises:
            RuntimeError: Metric name not found
            RuntimeError: Input is not int or str

        Returns:
            str: Metric Name
        """
        if metric is None:
            return 'Top_Level'
        if type(metric) == int:
            self.check_id_valid(metric)
            return self._metric_id2name[metric]
        elif type(metric) == str:
            if metric not in self._metric_name2id:
                raise RuntimeError(f'Metric [{metric}] not found')
            return metric
        else:
            raise RuntimeError(f'Metric [{metric}] is of invalid type {type(metric)}')
    
    def get_parent_id(self, metric) -> int:
        id: int = self.get_metric_id(metric)
        self.check_id_valid(id)
        parent_id = self._metric_parent[id]
        if parent_id is None:   # top level case
            return None
        return parent_id

    def get_children_ids(self, metric) -> list[int]:
        id: int = self.get_metric_id(metric)
        children_list = []
        for child_id, parent_id in enumerate(self._metric_parent):
            if parent_id == id:
                children_list.append(child_id)
        return children_list
    
    def get_percentage_of_parent(self, metric) -> float:
        """
        Args:
            metric (str/int): Metric name or id

        Returns:
            float: percentage of parent as a float in [0, 100]
        """
        parent_id = self.get_parent_id(metric)
        cur_id = self.get_metric_id(metric)
        
        if parent_id == None: # top level case
            return 1.0
        
        children_ids: list[int] = self.get_children_ids(parent_id)
        assert cur_id in children_ids
        values: list[float] = [self.get_value(child_id) for child_id in children_ids]
        total: float = sum(values)
        cur_index = children_ids.index(cur_id)
        return values[cur_index] / total
    
    def get_metric_hierarchy(self, metric) -> list[int]:
        if metric == 'Top_Level' or metric == 0:
            return_list = [0]
            return return_list
        
        id: int = self.get_metric_id(metric)
        hierarchy: list[int] = [id]
        parent_id = self.get_parent_id(id)
        while parent_id is not None:
            hierarchy.append(parent_id)
            parent_id = self.get_parent_id(parent_id)
        hierarchy.reverse()
        if hierarchy[0] == 0:
            hierarchy.pop(0)
        return hierarchy
    
    def get_hierarchy_percentage(self, metric) -> float:
        ids: list[int] = self.get_metric_hierarchy(metric)
        percentages: list[float] = [self.get_percentage_of_parent(id) for id in ids]
        final_percentage: float = 1.0
        for p in percentages:
            final_percentage *= p
        return final_percentage
            
    # Metric Values
    def load_values(self, value_file: str) -> None:
        self._metric_value: list[float] = [None] * self.num_metrics
        with open(value_file, 'r') as file:
            for line in file:
                comment_start = line.find('#')
                if comment_start != -1:
                    comment = line[comment_start + 1:].strip()
                    if not ('%' in comment and '_' in comment):
                        continue
                    comment_args = comment.split()
                    percentage = float(comment_args[0])
                    metric = comment_args[2]
                    id: int = self.get_metric_id(metric)
                    self._metric_value[id] = percentage
                if line.startswith('runtime_ms'):
                    runtime_ms: int = int(line.split(' ')[1])
                    self.runtime_ms = runtime_ms

    def get_value(self, metric) -> float:
        id: int = self.get_metric_id(metric)
        return self._metric_value[id]
     
    # Breakdown
    def get_breakdown(self, metric = None) -> tuple[str, float, list[str], list[float]]:
        """
        Args:
            metric (_type_, optional): Metric name or id. Defaults to None.

        Raises:
            RuntimeError: If the sum of children values is non_positive

        Returns:
            str: metric_name
            float: metric_glob_pct
            float: metric_parent_pct
            list[str]: children_names
            list[float]: children_values
        """
        metric_id: int = self.get_metric_id(metric)
        children_ids:       list[int]   = self.get_children_ids(metric)
        hierarchy_names:    list[str]   = [self.get_metric_name(id) for id in self.get_metric_hierarchy(metric_id)]
        
        if metric is None or metric == 'Top_Level':  # Top Level (L1) Breakdown
            metric_name: str = 'Top_Level'
            metric_glob_pct: float = 1.0
        else:
            metric_name: str = '/'.join(hierarchy_names)
            metric_glob_pct: float = self.get_hierarchy_percentage(metric)
        if len(children_ids) == 0:
            return metric_name, metric_glob_pct, None, None
        
        values: list[float] = []
        for child_id in children_ids:
            if self.get_value(child_id) is None:
                print(f'None value in metric {self.get_metric_name(child_id)}')
            values.append(self.get_value(child_id))
        
        if sum(values) <= 0.0:
            raise RuntimeError(f'Metric "{metric}" has an invalid total of {sum(values)}, values: {values}')
        # percentages: list[float] = [val/total*100 for val in values]
        children_names: list[str] = [self.get_metric_name(id) for id in children_ids]
        children_values: list[float] = [self.get_value(id) for id in children_ids]
        
        metric_parent_pct = self.get_percentage_of_parent(metric_id)
        
        return metric_name, metric_glob_pct, metric_parent_pct, children_names, children_values
    
    def print_breakdown(self, metric = None) -> None:
        if metric is None:  # Top Level (L1) Breakdown
            metric: str = 'Top_Level'
        metric_path: str = '/'.join([self.get_metric_name(m) for m in self.get_metric_hierarchy(metric)])
        
        metric_name, metric_glob_pct, metric_parent_pct, children_names, children_values = self.get_breakdown(metric)
        print(f'Breakdown of "{metric_name}" ({metric_glob_pct:.2f}% of topdown total):')
        if children_names is None:
            print('    (No tracked children breakdown values)\n')
            return
        
        max_name_len: int = max([len(name) for name in children_names])
        total: float = sum(children_values)
        percentages: list[float] = [v / total * 100 for v in children_values]
        for name, val, percentage in zip(children_names, children_values, percentages):
            print(f'    {name:<{max_name_len}} {val:6.2f} | {percentage:6.2f}%')
        print(f'    total {"-"*(max_name_len-6)} {sum(children_values):6.2f} | {sum(percentages):6.2f}%\n')


class MetricEntry:
    
    def __init__(
        self,
        metric: str,
        breakdown_names: list[str] = None,
    ) -> None:
        
        assert type(metric) == str, type(metric)
        
        self.metric:            str         = metric
        self.metric_glob_pcts:  list[float] = []
        self.run_count:         int         = 0
        
        if breakdown_names is None or len(breakdown_names) == 0:
            
            self.breakdown_names = None
            self.breakdown_allvalues = None
            self.breakdown_allpercentages = None
            return 
        
        assert all([type(name) == str for name in breakdown_names]), [f'{name}({type(name)})' for name in breakdown_names]
        
        self.breakdown_names:   list[str]   = breakdown_names
        self.breakdown_allvalues:       list[list[float]]   = [[] for _ in range(len(self.breakdown_names))]
        self.breakdown_allpercentages:  list[list[float]]   = [[] for _ in range(len(self.breakdown_names))]
        
    def __str__(self) -> str:
        mean_percentage: float = sum(self.metric_glob_pcts) / self.run_count * 100
        
        if self.breakdown_names is None:
            s = f'Metric Entry of "{self.metric} ({mean_percentage:.2f}% of topdown total) [avg of {self.run_count} runs]:"\n'
            s += '    (No tracked children breakdown values)\n'
            return s
        
        max_name_len: int = max([len(name) for name in self.breakdown_names])
        means: list[float] = self.get_mean_percentages()
        maxes: list[float] = self.get_max_percentages()
        mins: list[float] = self.get_min_percentages()
        s = f'Metric Entry of "{self.metric} ({mean_percentage:.2f}% of topdown total) [avg of {self.run_count} runs]:"\n'
        s += f'    {" "*max_name_len} {"Mean":>6}  | {"Min":>6}  | {"Max":>6}\n'
        for name, percentage, min_, max_ in zip(self.breakdown_names, means, mins, maxes):
            percentage, min_, max_ = 100 * percentage, 100 * min_, 100 * max_
            s += f'    {name:<{max_name_len}} {percentage:6.2f}% | {min_:6.2f}% | {max_:6.2f}%\n'
        s += f'    total {"-"*(max_name_len-6)} {sum(means)*100:6.2f}%\n'
        return s
    
    # Accessors
    def get_runcount(self) -> int:
        return len(self.breakdown_allvalues[0])
    
    def _values2percentages(self, values: list[float]) -> list[float]:
        total: float = sum(values)
        run_percentages = [val / total for val in values]
        return run_percentages
                
    def get_max_percentages(self) -> list[float]:
        return [max(percentages) for percentages in self.breakdown_allpercentages]
    
    def get_min_percentages(self) -> list[float]:
        return [min(percentages) for percentages in self.breakdown_allpercentages]
    
    def get_mean_percentages(self) -> list[float]:
        return [sum(percentages) / self.run_count for percentages in self.breakdown_allpercentages]
    
    def get_glob_pct(self) -> float:
        return sum(self.metric_glob_pcts) / self.run_count

    # Adding Entries
    def add_entry(
        self,
        metric_glob_pct: float,
        breakdown_names: list[str],
        breakdown_values: list[float],
    ) -> None:

        assert type(metric_glob_pct) == float, type(metric_glob_pct)
        assert 0.0 <= metric_glob_pct <= 1.0, metric_glob_pct
        self.metric_glob_pcts.append(metric_glob_pct)
        self.run_count += 1
        
        if breakdown_names is None:
            assert self.breakdown_names is None
            return
        
        assert breakdown_names == self.breakdown_names, f'Entry should contain children names of {self.breakdown_names}, but given {breakdown_names}'
        assert len(breakdown_values) == len(breakdown_names), (len(breakdown_values), len(breakdown_names))
        assert all([type(val) == float for val in breakdown_values]), [f'{val}({type(val)})' for val in breakdown_values]
        
        breakdown_percentages = self._values2percentages(breakdown_values)
        breakdown_percentages = [[p] for p in breakdown_percentages]
        breakdown_values = [[v] for v in breakdown_values]
        self.breakdown_allvalues = [i+j for i,j in zip(self.breakdown_allvalues, breakdown_values)]
        self.breakdown_allpercentages = [i+j for i,j in zip(self.breakdown_allpercentages, breakdown_percentages)]


class MetricsDatabase:
    
    def __init__(self, categories: list[str], values: tuple, runtime_ms: int = None):
        
        assert len(categories) == len(values), (len(categories), len(values))
        
        # Categories and values
        self._categories: list[str] = categories
        self._values: tuple = values

        # Each entry can be the average of many runs
        self._metric_entries: dict[str, MetricEntry] = {}
        
        # Runtime is separate since the runtimes from the Perf runs are usually statistically slower than a normal run
        #   so use the runtime from non Perf runs
        if runtime_ms is not None:
            assert type(runtime_ms) == int, type(runtime_ms)
        self._runtime_ms = runtime_ms
        
    def __repr__(self):
        category_val_str = ', '.join([f'{cat}[{val}]' for cat, val in zip(self._categories, self._values)])
        metric_entry_str = ', '.join([metric for metric, entry in self._metric_entries.items()])
        s = f'MetricDatabase({category_val_str}, runtime_ms[{self._runtime_ms}]): {metric_entry_str}'
        return s
        
    # Breakdown
    def new_breakdown(self, metric: str, children_names: list[str]) -> None:
        if metric in self._metric_entries:
            return
        
        metric_entry = MetricEntry(metric, children_names)
        metric_nopath: str = metric.split('/')[-1] if ('/' in metric) else metric
        self._metric_entries[metric_nopath] = metric_entry
    
    def update_breakdown(
        self,
        metric: str,
        metric_glob_pct: float,
        names: list[str],
        values: list[float],
    ) -> None:
        
        metric_nopath: str = metric.split('/')[-1] if ('/' in metric) else metric
        
        if metric_nopath not in self._metric_entries:
            self.new_breakdown(metric_nopath, names)
        self._metric_entries[metric_nopath].add_entry(metric_glob_pct, names, values)
        
    def print_breakdown(self, metric = None) -> None:
        if metric is None:  # Top Level (L1) Breakdown
            metric: str = 'Top_Level'
        metric_path: str = '/'.join([self.get_metric_name(m) for m in self.get_metric_hierarchy(metric)])
        
        metric_entry = self.get_breakdown(metric)
        metric_entry_str = str(metric_entry)
        metric_entry_str = metric_entry_str[metric_entry_str.find('\n')+1:]
        metric_entry_str = f'Breakdown of "{metric_path}":\n' + metric_entry_str
        print(metric_entry_str)
        
    # Dataframe
    def gen_metric_df(self, metric: str) -> pd.DataFrame:
        assert metric in self._metric_entries, f'Metric "{metric}" not in {self._metric_entries.keys()}'
        columns: list[str] = self._categories + ['runtime_ms']
        data: list = list(self._values) + [self._runtime_ms]
        
        metric_entry: MetricEntry = self._metric_entries[metric]
        metric_children_names = metric_entry.breakdown_names
        if metric_children_names is None:
            return None
        metric_children_means = metric_entry.get_mean_percentages()
        assert len(metric_children_names) == len(metric_children_means), (len(metric_children_names), len(metric_children_means))
        
        columns.append('glob_pct')
        data.append(metric_entry.get_glob_pct())
        
        for name, mean in zip(metric_children_names, metric_children_means):
            columns.append(name)
            data.append(mean * 100)
            
        df = pd.DataFrame([data], columns=columns)
        return df    


# Runtime
def _get_runtime_from_summary(directory: str) -> list[int]:
    """A summary file is created the first time a directory is used in this script.
    Instead of searching through all the log files, get the runtime from summary instead
    
    Args:
        directory (str): path to a directory of log files

    Returns:
        list[int]: list of runtimes, as formated by user
    """
    files = os.listdir(directory)
    matching_files = [f for f in files if f.startswith("avg_") and f.endswith("ms")]
    if len(matching_files) != 1:
        return None
    
    file_path = os.path.join(directory, matching_files[0])
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        
    if line.startswith("[") and line.endswith("]"):
        line = line[1:-1]  # Remove the brackets
        try:
            integer_list = [int(num.strip()) for num in line.split(',')]
        except ValueError:
            return None  # In case there are non-integer values
    else:
        return None
    return integer_list
    
def _get_avg_from_list(runtimes: list[int], trim: float = None):
    if (trim is None) or (type(trim) != float) or (trim < 0.0):
        middle_portion: float = 1.0
    else:
        middle_portion: float = trim
        
    if len(runtimes) <= 2:
        trim_amount = 0
    else:
        trim_amount = ceil(len(runtimes) * (1-middle_portion)/2)
    runtimes_adjusted = runtimes[trim_amount : len(runtimes) - trim_amount]
        
    avg: float = sum(runtimes_adjusted) / len(runtimes_adjusted)
    return avg
    
def _get_runtime_avg(directory: str, func_get_runtime, makelogfile: bool = False, trim: float = None) -> tuple[list[int], str]:
    os.chdir(directory)
    
    write_file = False
    runtimes: list[int] = _get_runtime_from_summary(directory)
    if runtimes is None:
        runtimes=[]
        for file in os.listdir(directory):
            if file.endswith('.log'):
                runtimes.append(func_get_runtime(file))
        runtimes.sort()
        
        if makelogfile:
            write_file = True
        
    avg: float = _get_avg_from_list(runtimes, trim)
        
    if write_file:
        avg_file_name = f'avg_{int(avg)}ms'
        with open(f'avg_{int(avg)}ms', 'w') as file:
            file.write(f"{len(runtimes)} runs total: {runtimes}\n")
            if (trim is None) or (type(trim) != float) or (trim < 0.0):
                middle_portion: float = 1.0
            else:
                middle_portion: float = trim
            file.write(f"Runtime Mean: {avg} (middle {int(middle_portion*100)}% of runs)\n")
    else:
        avg_file_name = None
            
    return runtimes, avg_file_name

def get_runtime_from_dir(directory: str, func_get_runtime) -> int:
    """Gets average runtime from directory. Get avg runtime from summary file if it exists.
    Otherwise applies the function func_get_runtime to every file in directory then calculates trimmed average (middle 60%)

    Args:
        directory (str): Directory path
        func_get_runtime (func): Function to parse each file to return a runtime value

    Returns:
        int: Runtime value
    """
    files = os.listdir(directory)
    matching_files = [f for f in files if f.startswith("avg_") and f.endswith("ms")]
    if len(matching_files) == 0:
        _, runtime = _get_runtime_avg(directory, func_get_runtime, makelogfile=True, trim=0.6)
    else:
        runtime = matching_files[0]
        if len(matching_files) > 1:
            print(f"There are multiple avg_*ms files in directory, using {matching_files[0]}")
        
    runtime = int(runtime.replace('avg_', '').replace('ms', ''))
    return runtime

# Data Extraction from File
def extract_from_files(
    report_files: list[str],
    metrics: list[str],
    categories: list[str],
    runtime: dict[tuple, int],
    extract_func
    ) -> tuple[dict[tuple, MetricsDatabase], list[set]]:
    
    """For each file in report_files, extract data for each metric. Organize data by categories.
    
    Returns:
        dict[tuple, MetricsDatabase]: Each config of categories (the tuple) has a collection of metric numbers in MetricsDatabase object
        list[set]: All the values observed for each category
    """
        
    sets: list[set] = [set() for _ in range(len(categories))] 
    metricdb_dict: dict[tuple, MetricsDatabase] = {}
    
    # Parse and collect data
    for report_file in report_files:
        args_tup = extract_func(report_file)
        assert len(args_tup) == len(sets)
        for i, arg in enumerate(args_tup):
            sets[i].add(arg)
                
        tdh: TopDownHierarchy = TopDownHierarchy()
        tdh.load_values(report_file)
        
        if args_tup not in metricdb_dict:
            runtime = runtimes[args_tup] if (runtimes is not None) else None
            metricdb_dict[args_tup] = MetricsDatabase(categories, args_tup, runtime_ms=runtime)
        
        for m in metrics:
            m_name, m_glob_pct, m_parent_pct, c_names, c_values = tdh.get_breakdown(m)
            metricdb_dict[args_tup].update_breakdown(m_name, m_glob_pct, c_names, c_values)
    
    return metricdb_dict, sets

# plot function
def generate_batch_of_graphs(
    categories: list[str],
    diff_hyperparam: list[str],
    diff_hyperparam_order: list[list[str]], 
    report_files: list[str],
    extract_func,
    track_metrics: list[str],
    chart_dir: str,
    runtime: dict[tuple, int] = None,
    relative_scaling: bool = False
    ) -> None:
    
    max_hierarchy = 2
    assert 1 <= len(diff_hyperparam) <= max_hierarchy
    assert all([(d in categories) for d in diff_hyperparam]), ([(d in categories) for d in diff_hyperparam])
    assert len(diff_hyperparam) == len(diff_hyperparam_order)
    
    if (not runtime) and relative_scaling:
        print("No runtimes provided, relative_scaling is turned off")
        relative_scaling = False    # runtimes are required for this feature
    
    metricdb_dict: dict[tuple, MetricsDatabase] = {}
    print('extracting data from topdown log files...')
    metricdb_dict, sets = extract_from_files(report_files, track_metrics, categories, runtimes, extract_func)
    print('done')
    
    # created a folder for every comb of shared params
    shared_param_combinations = [[]]
    for idx, cat in enumerate(categories):
        new_comb_list = []
        if cat in diff_hyperparam:
            for comb in shared_param_combinations:
                new_comb_list.append(comb.copy() + [None])
        else:
            for comb in shared_param_combinations:
                for val in sets[idx]:
                    new_comb_list.append(comb.copy() + [val])
        shared_param_combinations = new_comb_list
        
    # Bar chart is grouped in order of higher to lower hierarchy
    for comb in shared_param_combinations:
        folder_name = 'chart' if (len(categories) == len(diff_hyperparam)) else '_'.join([c for c in comb if c is not None])
        os.makedirs(f'{chart_dir}/{folder_name}', exist_ok=True)
        
        # expand combinations, by replacing "None"s with diff_param options
        comb_list = [comb]
        for hyperparam, hyperparam_vals in zip(diff_hyperparam, diff_hyperparam_order):
            hyperparam_idx = categories.index(hyperparam)
            comb_list_wip = []
            for c in comb_list:
                for val in hyperparam_vals:
                    assert val in sets[hyperparam_idx], f'{val} not in {sets[hyperparam_idx]}'
                    c[hyperparam_idx] = val
                    comb_list_wip.append(c.copy())
            comb_list = comb_list_wip

        # Make the dataframes
        for metric in track_metrics:
            df_list = []
            for c in comb_list:
                assert all([v is not None for v in c]), c
                # print(f'{metric} {c}')
                df = metricdb_dict[tuple(c)].gen_metric_df(metric)
                df_list.append(df)
            df_combined = pd.concat(df_list)
            df_combined.reset_index(drop=True, inplace=True)
            
            plotname = f'{chart_dir}/{folder_name}/{folder_name}_{metric}'
    
            new_dataframe_to_stackedbarchart(df_combined, categories, diff_hyperparam, diff_hyperparam_order, plotname, rel_scaling=relative_scaling)

def new_dataframe_to_stackedbarchart(
    df: pd.DataFrame,
    categories: list[str],
    diff_hyperparam: list[str],
    diff_hyperparam_order: list[list[str]],
    plotname: str,
    rel_scaling: bool = False
    ) -> None:
    
    # Plot each category on top of the previous one
    bar_sections: list[str] = [s for s in df.columns if (s not in categories) and (s != 'runtime_ms') and (s != 'glob_pct')]
    
    # Plotting
    side_pad = 1
    gap_hier = [1, 0.8] # list is the opposite order as diff_hyperparams, in increasing hierarchy

    # Set the positions for bars on X-axis
    spacing = [1]
    for l in reversed([len(h) for h in diff_hyperparam_order]):
        spacing.append(l * spacing[-1])
    total_num_bars = spacing.pop()
    spacing = list(reversed(spacing))
    xpos = [0] * total_num_bars
    for hier_idx in reversed(range(len(diff_hyperparam_order))): # generate offsets per hierarchy
        offsets = [ [(i/spacing[hier_idx])*gap_hier[len(diff_hyperparam) -1 -hier_idx]] * spacing[hier_idx] for i in range(0, total_num_bars, spacing[hier_idx])]
        offsets = reduce(lambda a, b: a+b, offsets)
        xpos = [x+y for x,y in zip(xpos, offsets)]
                
    fig, ax = plt.subplots(figsize=(10, 6))

    # relative scaling (relative to the first bar of the lowest hierarchy)
    if rel_scaling:
        scaling: list[float] = []
        n_lowest_hier = len(diff_hyperparam_order[-1])
        baseline = 1
        for i, row in df.iterrows():
            rtms = row['runtime_ms'] if (row['runtime_ms'] is not None) else 1
            rel_val = rtms * row['glob_pct']
            if i % n_lowest_hier == 0:
                baseline = rel_val
            scaling.append(rel_val / baseline)
        assert len(scaling) == len(df)
    else:
        scaling = [1.0] * total_num_bars
            
    # x axis category
    bottom = [0] * len(df)
    for idx, section in enumerate(bar_sections):
        heights = list(df[section])
        heights = [h*s for h,s in zip(heights, scaling)]
        bars = ax.bar(xpos, heights, bottom=bottom, label=section)  #, color=colors[idx])
        # Add values on the bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + height / 2, 
                    f'{height:.2f}', 
                    ha='center', 
                    va='center',
                    fontsize=9
                )
        bottom = [i + j for i, j in zip(bottom, heights)]
        
    # Add scaling numbers on top of each stacked bar
    if rel_scaling:
        for i, (rect, scal) in enumerate(zip(ax.patches, scaling)):
            height = bottom[i]
            x_pos = rect.get_x() + rect.get_width() / 2
            ax.text(x_pos, height + 0.5,  # Add a small offset to avoid overlap
                    f'{scal:5.3f}x',
                    ha='center', va='bottom',
                    rotation=15, fontsize=8)
        
    # Set X-axis labels with hierarchical index
    ax.set_xticks(xpos, labels=diff_hyperparam_order[-1] * (total_num_bars // len(diff_hyperparam_order[-1])), rotation=20)
    ax.tick_params(axis='x', which='minor')
        
    # Group by Major category
    if len(diff_hyperparam) == 2:
        # center the major category labels among the minor labels
        maj = ax.secondary_xaxis(location=0)
        n_minor = len(diff_hyperparam_order[-1])
        maj_label_pos = [(xpos[i] + xpos[i+n_minor-1])/2 for i in range(0, total_num_bars, n_minor)]
        maj.set_xticks(maj_label_pos, labels=diff_hyperparam_order[-2])
        maj.tick_params('x', length=0, pad=30, which='major')

    ax.set_xlim(0-side_pad, xpos[-1] + side_pad)
    ax.set_ylim(0, max(bottom) + 8)  # Adjust the 15 to give enough space for the tallest bar
    
    # Add legend and labels
    xlab = ' & '.join(list(reversed(diff_hyperparam)))
    ax.set_xlabel(xlab, labelpad=20)
    ax.set_ylabel('Percentage Normalized to Runtime')
    plot_title = plotname.split('/')[-1].replace('_tma', '').replace('_normalized', '').replace('_', ' ')
    ax.set_title(f'{plot_title} Metrics by {xlab}{" (normalized runtime)"*rel_scaling}')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.04, 0.9), borderaxespad=0)
    
    plotname = f'{plotname}{"_normalized"*rel_scaling}.png'
    plt.savefig(f'{plotname}')
    plt.close(fig)
    print(f'plot saved to {plotname}')
    

if __name__ == '__main__':
    
    ##### CONFIGURE CHART OUTPUT #####
    # Edit these to match your use
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    chart_output_dir: str = f'{CUR_DIR}/demo/td_charts'
    
    # Each combination of these categories will have a separate folder of name '{cat0}_{cat1}_...'
    categories: list[str] = ['Algo', 'Impl']
    
    # Bar charts will be grouped by these categories in each figure in order of higher to lower hierarchy
    
    ### Example 1: compare between each implementation
    diff_hyperparam: list[str] = ['Impl']
    diff_hyperparam_order: list[list[str]] = [['implX', 'implY', 'implZ']]
    
    # ### Example 2: compare between each implementation, as well as algo
    # diff_hyperparam: list[str] = ['Algo', 'Impl']
    # diff_hyperparam_order: list[list[str]] = [['algoA', 'algoB'],
    #                                           ['implX', 'implY', 'implZ']]
        
    # The list of metrics to create top down charts of
    track_metrics = ['Top_Level', \
                    'tma_backend_bound',
                        'tma_core_bound', \
                        'tma_memory_bound',
                            'tma_dram_bound', \
                    'tma_bad_speculation', \
                    'tma_frontend_bound',
                        'tma_fetch_bandwidth', \
                        'tma_fetch_latency']     # "Top_Level" is a keyword that my script looks for, do not remove
    
    
    ##### TOP DOWN DATA INPUT #####
    # Collect a list of all the files you want to extract top down data from
    # Edit the logic below to fill "report_files"
    report_files: list[str] = []
    
    input_dir: str = f'{CUR_DIR}/demo/'
    root_folders = [f'{input_dir}/topdown_files/algoA_implX_run4',
                    f'{input_dir}/topdown_files/algoA_implY_run3',
                    f'{input_dir}/topdown_files/algoA_implZ_run1',
                    f'{input_dir}/topdown_files/algoB_implX_run3',
                    f'{input_dir}/topdown_files/algoB_implY_run7',
                    f'{input_dir}/topdown_files/algoB_implZ_run9']
    for folder in root_folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if not file.endswith('.log'):
                    continue
                path = os.path.join(root, file)
                report_files.append(path)
            
    # Each of the files above in "report_files" will be parsed y this function below
    def get_args_from_filename(file: str) -> tuple:
        """Edit this function to match your file naming convention
        Args:   file (str): Full path of input file
        Returns:     tuple: Tuple of all custom categories you wish to keep track for organization
        """
        algo, impl, _ = file.split('/')[-1].replace('.log','').split('_')
        return (algo, impl)


    ##### OPTIONAL - BAR CHART RELATIVE SCALING ##### 
    # Relative scaling is relative to the full runtime of the first bar of each option in the lowest hierarchy
    # Fill the dict "runtimes" using keys that match the same tuple format as get_args_from_filename() above
    
    runtimes: dict[tuple, int] = {}

    runtime_folders = [f'{input_dir}/runtime_files/algoA_implX_run4',
                       f'{input_dir}/runtime_files/algoA_implY_run3',
                       f'{input_dir}/runtime_files/algoA_implZ_run1',
                       f'{input_dir}/runtime_files/algoB_implX_run3',
                       f'{input_dir}/runtime_files/algoB_implY_run7',
                       f'{input_dir}/runtime_files/algoB_implZ_run9']
    
    def get_runtime_from_logfile(file: str) -> int:
        """Edit this function to extract runtime result from log dump
        Args: file (str): Full path of input file
        Returns:     int: Runtime data 
        """
        result = subprocess.run(['grep', 'runtime_ms', file], capture_output=True, text=True)
        result = result.stdout.splitlines()
        result: int = int(result[0].split(' ')[-1])
        assert isinstance(result, int)
        return result

    for folder in runtime_folders:
        # Edit this loop here to add logic on how to parse the folder name into your categories
        algo, impl, _ = folder.split('/')[-1].split('_')
        
        runtimes[(algo, impl)] = get_runtime_from_dir(folder, get_runtime_from_logfile)


    ##### Generate Charts #####
    generate_batch_of_graphs(categories,
                             diff_hyperparam,
                             diff_hyperparam_order,
                             report_files,
                             get_args_from_filename,
                             track_metrics,
                             chart_output_dir,
                             runtimes,
                             relative_scaling=False)
    generate_batch_of_graphs(categories,
                             diff_hyperparam,
                             diff_hyperparam_order,
                             report_files,
                             get_args_from_filename,
                             track_metrics,
                             chart_output_dir,
                             runtimes,
                             relative_scaling=True)
    