
import time
from datetime import timedelta
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import argparse

from PIL import Image
from cupy.cuda import memory_hooks

from mpi4py import MPI
#comm = MPI.COMM_WORLD
#mpirank = comm.Get_rank()
mpirank = 0
#mpisize = comm.Get_size()
mpisize = 1

compute_rev_file = os.path.join(os.path.dirname(__file__), "cu_src","compute_rev.cu")
compute_rev_file2 = os.path.join(os.path.dirname(__file__), "cu_src","compute_rev2.cu")
compute_rev_file3 = os.path.join(os.path.dirname(__file__), "cu_src","compute_rev3.cu")
compute_rev_file4 = os.path.join(os.path.dirname(__file__), "cu_src","compute_rev4.cu")
compute_rev_file5 = os.path.join(os.path.dirname(__file__), "cu_src","compute_rev5.cu")

borders_rev_file = os.path.join(os.path.dirname(__file__), "cu_src","borders_rev.cu")

def read_code(code_filename):
    with open(code_filename, 'r') as f:
        code = f.read()
    #for k, v in params.items():
    #    code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


def print0(*args):
    if mpisize > 1:
        if mpirank == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)

def make_directory(save_root, name):
    if not os.path.exists(os.path.join(save_root, name)):
        os.mkdir(os.path.join(save_root, name))

parser = argparse.ArgumentParser(description='Fast Renderer for FractalDB using GPUs')
parser.add_argument('--load_root', default='./csv/data1k_fromPython/csv_rate0.2_category1000', type = str, help='load csv root')
parser.add_argument('--save_root', default='./bake_db/test', type = str, help='save png root')
parser.add_argument('--image_size_x', default=362, type = int, help='image size x')
parser.add_argument('--image_size_y', default=362, type = int, help='image size y')
parser.add_argument('--image_res', default=362, type = int, help='image size y')
parser.add_argument('--pad_size_x', default=6, type = int, help='padding size x')
parser.add_argument('--pad_size_y', default=6, type = int, help='padding size y')
parser.add_argument('--iteration', default=600, type = int, help='iteration')
parser.add_argument('--draw_type', default='patch_gray', type = str, help='{point, patch}_{gray, color}')
parser.add_argument('--weight_csv', default='./weights/weights_0.4.csv', type = str, help='weight parameter')
parser.add_argument('--instance', default=10, type = int, help='#instance, 10 => 1000 instance, 100 => 10,000 instance per category')
parser.add_argument('--rotation', default=1, type = int, help='Flip per category')
parser.add_argument('--nweights', default=25, type = int, help='Transformation of each weights. Original DB is 25 from csv files')
parser.add_argument('-g','--ngpus-pernode', default=1, type = int, help='Num of GPUs in the node')
parser.add_argument('--backend', default='egl', type = str, help='{glfw, egl}')
parser.add_argument('-d', '--debug', action='store_true',default=False,help='Check sanity for all the images... pixel count')
parser.add_argument('-t', '--tomemory', action='store_true',default=False,help='Do not save the image but only retain to memory')

def main():
    args = parser.parse_args()
    print0("\n\nAll arguments:\n",args)
    print0("\n\n")
    args = parser.parse_args()

    # Added global relosolution
    g_res = args.image_res
    # Main variables
    csv_names = os.listdir(args.load_root)
    csv_names.sort()
    weights = np.genfromtxt(args.weight_csv,dtype=float,delimiter=',')
    
    if args.nweights <= 25 and args.nweights > 0:
        weights = weights[:args.nweights]
    else:
        print('error on weights [1-25]')
        exit(0)

    # MPI related configurations
    DEV = mpirank % args.ngpus_pernode
    
    nlist = len(csv_names)
    print0(f"Number of Classes found in csv files {nlist}")
    nlist_per_rank = (nlist+mpisize-1)//mpisize
    start_list = mpirank*nlist_per_rank
    end_list = min((mpirank+1)*nlist_per_rank, nlist)

    csv_names = csv_names[start_list:end_list]
    print0(f"rank: {mpirank}, csv_names:{csv_names}]\n\n")
    # comm.Barrier()
    if mpirank == 0:
        if not os.path.exists(os.path.join(args.save_root)):
            # print("Error: No directory to save DB")
            # exit(0)
            os.mkdir(os.path.join(args.save_root))
    #comm.Barrier()

    npts = args.iteration
    #cp_dposs = cp.zeros((npts,2),dtype=cp.float32)
    #cp_npts = cp.zeros((1),dtype=cp.int32)
    #cp_npts[0] = npts

    initial_experiment_time = time.perf_counter()
    count = 0
    class_num = 0

    code = read_code(compute_rev_file)
    module = cp.RawModule(code=code)
    compute_kernel = module.get_function('compute_rev')
    compute_kernel_2 = module.get_function('compute_rev2')
    compute_kernel_3 = module.get_function('compute_rev3')
    compute_kernel_4 = module.get_function('compute_rev4')
    compute_kernel_5 = module.get_function('compute_rev5')

    code2 = read_code(borders_rev_file)
    module2 = cp.RawModule(code=code2)
    borders_kernel = module2.get_function('borders_rev')

    #config = {'PARAM_SIZE': 2}
    #code_2 = read_code(compute_rev_file2)
    #compute_kernel_2 = cp.RawKernel(code_2,'compute_rev2')
    #config = {'PARAM_SIZE': 3}
    #code_3 = read_code(compute_rev_file3)
    #compute_kernel_3 = cp.RawKernel(code_3,'compute_rev3')
    #config = {'PARAM_SIZE': 4}
    #code_4 = read_code(compute_rev_file4)
    #compute_kernel_4 = cp.RawKernel(code_4,'compute_rev4')
    #config = {'PARAM_SIZE': 5}
    #code_5 = read_code(compute_rev_file5)
    #compute_kernel_5 = cp.RawKernel(code_5,'compute_rev5')

    #cp.cuda.set_allocator(None)
    #end_gpu = cp.cuda.Event()
    #start_gpu = cp.cuda.Event()
    for csv_name in tqdm(csv_names):
        #if csv_name == "00004.csv" or csv_name == "00003.csv":
        #    continue
        #print(cp.get_default_memory_pool())
        mempool = cp.get_default_memory_pool()
        #mempool.set_limit(size=24*1024**3)
        #pinned_mempool = cp.get_default_pinned_memory_pool()
        initial_time = time.perf_counter()
        name, ext = os.path.splitext(csv_name)
        # class_str =  '%05d' % class_num
        print0(' ->'+ csv_name)        
        
        if ext != '.csv': # Skip except for csv file
            continue
        #print(name)
        if not args.tomemory:
            make_directory(args.save_root, name) # Make directory
        fractal_name=name
        fractal_weight = 0

        ################### This for is the main rendering whic changes parameters
        
        for weight in weights:
            #with memory_hooks.DebugPrintHook():
            print(fractal_weight,flush=True)
            #start_gpu.record()
            padded_fractal_weight = '%02d' % fractal_weight
            fractal_weight_count = padded_fractal_weight
            
            params = np.genfromtxt(os.path.join(args.load_root, csv_name), dtype=float, delimiter=',')
                            

            #print("DEBUG",flush=True)
            rnd = np.random.uniform(size=2).astype(cp.float32)
            cp_rand = cp.random.rand(npts)
            print(mempool.used_bytes())
            param_size = params.shape[0]
            #cp_param_size = cp.zeros((1),dtype=cp.int32)
            #cp_param_size[0] = params.shape[0]
            # param_size = len(params[class_num])
            rotation = [1.0,1.0]      
            #print("DEBUG",flush=True)
            
            #uniforms_buffer = struct.pack('ffi2f',rnd[0],rnd[1],param_size,rotation[0],rotation[1])
            #uniforms.write(uniforms_buffer)
            #uniforms.bind_to_uniform_block(3)

            cp_arr = cp.zeros((20,7),dtype=cp.float32)
            print(mempool.used_bytes())
            #cp_arr = cp.random.uniform(size=(param_size, 7)).astype(cp.float32)
            for i in range(param_size):
                cp_arr[i][0] = params[i][0] * weight[0]
                cp_arr[i][1] = params[i][1] * weight[1]
                cp_arr[i][2] = params[i][2] * weight[2]
                cp_arr[i][3] = params[i][3] * weight[3]
                cp_arr[i][4] = params[i][4] * weight[4]
                cp_arr[i][5] = params[i][5] * weight[5]
                cp_arr[i][6] = params[i][6] 
            #print("DEBUG",flush=True)

            #4 bytes for a float32
            compute_shared_mem = 5 * (7) * 4 
            #compute_kernel.max_dynamic_shared_size_bytes(shared_mem)

            #m_arr = array.array('f')
            #for i in range(param_size):
            #    m_arr.append(params[i][0] * weight[0])
            #    m_arr.append(params[i][1] * weight[1])
            #    m_arr.append(params[i][2] * weight[2])
            #    m_arr.append(params[i][3] * weight[3])
            #    m_arr.append(params[i][4] * weight[4])
            #    m_arr.append(params[i][5] * weight[5])
            #    m_arr.append(params[i][6])
            #unif.write(m_arr)
            #unif.bind_to_uniform_block(0)
            #del m_arr
            
            #poss.bind_to_storage_buffer(0)
            #color.bind_to_storage_buffer(1)
            #prj.bind_to_storage_buffer(2)
            
            #print("DEBUG",flush=True)
            cp_dposs = cp.zeros((npts,2),dtype=cp.float32)
            print(mempool.used_bytes())
            #compute.run(group_x=1)
            #print("DEBUG",flush=True)

            print("1",cp_dposs)
            compute_args = (param_size,npts,cp_arr,cp_rand,cp_dposs)
            grid = (1,)
            block = (256,)
            compute_kernel(grid, block, args=compute_args, shared_mem=compute_shared_mem)
            print("2",cp_dposs)
            #if param_size == 2:
            #    compute_kernel_2(grid, block, args=compute_args, shared_mem=shared_mem)
            #elif param_size == 3:
            #    compute_kernel_3(grid, block, args=compute_args, shared_mem=shared_mem)
            #elif param_size == 4:
            #    compute_kernel_4(grid, block, args=compute_args, shared_mem=shared_mem)
            #elif param_size == 5:
            #    compute_kernel_5(grid, block, args=compute_args, shared_mem=shared_mem)
            #end_gpu.record()
            #end_gpu.synchronize()
            #del cp_arr,cp_rand,cp_dposs
            #print("DEBUG",flush=True)
            #compute_kernel((1,),(256,),(cp_npts,cp_arr, cp_rand, cp_dposs))
            #ctx.memory_barrier(barriers=ModernGL.SHADER_STORAGE_BARRIER_BIT)
            cp_dbor = cp.zeros((4),dtype=cp.float32)
            borders_args = (cp_dposs, npts, cp_dbor, args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y)
            borders_shared_mem = 256 * (4) * 4 
            borders_kernel(grid,block,args=borders_args,shared_mem=borders_shared_mem)

            print("Before",cp_dposs)
            cp_dposs = cp_dposs.astype(cp.uint16)
            print("After",cp_dposs)
            #borders.run(group_x=1)
            #ctx.memory_barrier(barriers=ModernGL.SHADER_STORAGE_BARRIER_BIT)

            for count in range(args.instance):
                                
                for trans_type in range(args.rotation):
                                            
                    if trans_type == 0:
                        rotation = [1.0,1.0]
                    elif trans_type == 1:
                        rotation = [-1.0,1.0]
                    elif trans_type == 2:
                        rotation = [1.0,-1.0]
                    elif trans_type == 3:
                        rotation = [-1.0,-1.0]

                    rnd = np.random.uniform(size=2)
                    cp_rand = cp.random.rand(2)

                    image = cp.array(Image.new("RGB", (args.image_size_x, args.image_size_y)))
                    for i in range(npts):
                        #print(cp_dposs[i][1],cp_dposs[i][0],flush=True)
                        image[cp_dposs[i][1],cp_dposs[i][0],:] = 127,127,127
                    image = cp.asnumpy(image)
                    image = Image.fromarray(image)
                    image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + ".png"))
            
            #prj.bind_to_uniform_block(0)
            fractal_weight += 1
            
            #mempool.free_all_blocks()
            #pinned_mempool.free_all_blocks()
        class_num += 1
        total_time = time.perf_counter() - initial_time

if __name__ == "__main__":
    main()
"""
        
                    ctx.clear(0.0, 0.0, 0.0, 1.0) 
                    
                    #uniforms_buffer = struct.pack('ffi2f',rnd[0],rnd[1],param_size,rotation[0],rotation[1])
                    #uniforms.write(uniforms_buffer)
                    #uniforms.bind_to_uniform_block(3)
                    
                    ######### Get the image
                    #if args.backend == 'glfw':
                    #    vao.render(mode=ModernGL.POINTS)
                    #    glfw.swap_buffers(window)
                    #    data = glReadPixels(0.0, 0.0, g_res, g_res, GL_RGB, GL_UNSIGNED_BYTE, None)
                    #elif args.backend == 'egl':
                    #    fbo.use()
                    #    vao.render(mode=ModernGL.POINTS) 
                    #    fbo.read_into(buf1,attachment=0, components=g_components, alignment=g_alignment)
                    #    data =  buf1.read()
                        
                    ####### Choose the image sanity
                    
                    if args.debug:
                        data_np = np.frombuffer(data, dtype=np.byte)
                        print("Item size in np array and itemsize: {:,} , {}".format(data_np.size,data_np.itemsize))
                        print("Raw data from framebuffer to numpy: {:,} bytes".format(int(data_np.size * data_np.itemsize)))
                        
                        pixel_count = np.sum(data_np/127/3)
                        print(colored("\nPixel count: {:,}".format(int(pixel_count)),'light_magenta'))
                        
                        if pixel_count <= 9.0:
                            print("\n\nFrameBuffer bytes is not full !!")
                            exit(0)
                        else:
                            image = Image.frombytes('RGB', (g_res, g_res), data)
                            
                            if not image.getbbox():
                                print("\n\nSave image is empty!!")
                                
                                image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                                print(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                                exit(0)                   
                            image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                            
                    else:
                        
                        if args.tomemory:
                            # TODO -- Send everything to RAM in efficient PNG binary format
                            pass  
                        else:
                            image = Image.frombytes('RGB', (g_res, g_res), data)
                            image.save(os.path.join(args.save_root, fractal_name, fractal_name + "_" + fractal_weight_count + "_count_" + str(count) + "_flip"+ str(trans_type) + ".png"))
                    
                    if args.backend == 'glfw':
                        glfw.poll_events()        
"""
