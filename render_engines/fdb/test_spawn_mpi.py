import os
import sys
import argparse

from mpi4py import MPI
from termcolor import colored

comm = MPI.COMM_WORLD
g_universe = MPI.UNIVERSE_SIZE
g_mpirank = comm.Get_rank()
g_mpisize = comm.Get_size()


def print0(*args):
    if g_mpisize > 1:
        if g_mpirank == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)
        

parser = argparse.ArgumentParser(description='Test Spawn MPI')
parser.add_argument('-m','--mpi_spawn', default=0, type = int, help='Num of GPUs in the node')

def main():

    args = parser.parse_args()
    # print0("\n\nAll arguments:\n",args)
    # print0("\n\n")
    SSD = os.getenv('SGE_LOCALDIR', 'ssd')
    # print0(SSD)
    # print0("Rank: {}".format(g_mpirank))
    
    ## Create the main directory with rank 0 to avoid threads race:
    save_root = SSD + '/fdb1000_GPU_spawn_egl'
    if g_mpirank % 4 == 0:
        if not os.path.exists(os.path.join(save_root)):
            os.mkdir(os.path.join(save_root))
       
    comm.barrier()    
    
    ## Set the host-file 
    
    mpi_info = MPI.Info.Create()
    host = MPI.Get_processor_name()
    # print(host)
    mpi_info.Set('host',host)
    
    ## For GPU
    new_comm = MPI.COMM_SELF.Spawn(sys.executable, 
                                   args=['spawn_mpi_gpu_ramtest.py', '--load_root', 'csv/searched_params/csv_rate0.2_category1000_points200000','-g','4','--save_root',SSD + '/fdb1000_GPU_spawn','-p',str(g_mpirank),'-w',str(g_mpisize)], 
                                   maxprocs=int(args.mpi_spawn),
                                   info=mpi_info)
    
    ## For CPU
    # new_comm = MPI.COMM_SELF.Spawn(sys.executable, 
    #                                args=['spawn_mpi_cpu_ramtest.py', '--load_root', 'csv/searched_params/csv_rate0.2_category1000_points200000','--save_root',SSD + '/fdb1000_CPU_spawn','-p',str(g_mpirank),'-w',str(g_mpisize)], 
    #                                maxprocs=int(args.mpi_spawn),
    #                                info=mpi_info)
    
    # new_comm.barrier()
    
    # print0("Size of the MPI: {}".format(g_mpisize))
    # print0("Universe MPI: {}".format(g_universe))
    new_comm.Disconnect()
    comm.barrier()
    
    print0(colored('Finished the spawn...','blue', 'on_black',['bold', 'blink']))

    # MPI.Finalize()

if __name__ == "__main__":
    main()