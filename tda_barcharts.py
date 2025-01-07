import os
import subprocess
import tda_lib

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
        
        runtimes[(algo, impl)] = tda_lib.get_runtime_from_dir(folder, get_runtime_from_logfile)


    ##### Generate Charts #####
    tda_lib.generate_batch_of_graphs(categories,
                                    diff_hyperparam,
                                    diff_hyperparam_order,
                                    report_files,
                                    get_args_from_filename,
                                    track_metrics,
                                    chart_output_dir,
                                    runtimes,
                                    relative_scaling=False)
    tda_lib.generate_batch_of_graphs(categories,
                                    diff_hyperparam,
                                    diff_hyperparam_order,
                                    report_files,
                                    get_args_from_filename,
                                    track_metrics,
                                    chart_output_dir,
                                    runtimes,
                                    relative_scaling=True)
