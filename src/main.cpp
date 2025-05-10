#include "starpu_setup.hpp"
#include "args_parser.hpp"
#include <iostream>

class VectorTest 
{
 public:
  std::unique_ptr<float[]> data_;
  size_t size;
  size_t data_size;
  starpu_data_handle_t in_handle;

  VectorTest(size_t n) : size(n), data_size(sizeof(float)) 
  {
    data_ = std::make_unique<float[]>(size);
    for (size_t i = 0; i < size; ++i) {
      data_[i] = static_cast<float>(i);
    }
  }

  void register_vector() 
  {
    starpu_vector_data_register(&in_handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t> (data_.get()), size, data_size);
  }

  void cleanup() {
    starpu_data_unregister(in_handle);
  }
};

int main(int argc, char* argv[])
{
  ProgramOptions opts = parse_arguments(argc, argv);

  if (opts.show_help)
  {
    display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid)
    return 1;

  std::cout << "Scheduler : " << opts.scheduler << "\n";
  std::cout << "Iteration   : " << opts.iterations << "\n";

  try
  {
    StarPUSetup starpu(opts.scheduler.c_str());

    for (int i = 0; i < opts.iterations; ++i)
    {
      int N = 1000;
      float* output_buffer = static_cast<float*>(malloc(N * sizeof(float)));
      starpu_data_handle_t output_handle;
      starpu_variable_data_register(&output_handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(output_buffer), N * sizeof(float));      

      int num_buffers = 1;

      struct starpu_task* task = starpu_task_create();
      task->handles[0] = output_handle;
      task->cl = starpu.codelet();
      task->synchronous = 1;
      task->cl_arg_free = 1;
      task->cl_arg = (void*)opts.model_path.c_str();
      task->cl_arg_size = opts.model_path.size() + 1;
      task->dyn_handles = (starpu_data_handle_t*)malloc(num_buffers * sizeof(*(task->dyn_handles)));
      task->dyn_modes = (starpu_data_access_mode*)malloc(num_buffers * sizeof(*(task->dyn_modes)));
      task->nbuffers = num_buffers;

      //Input handle
      VectorTest vec(10);
      vec.register_vector();

      task->dyn_handles[0] = vec.in_handle;    
      task->dyn_modes[0] = STARPU_R;

      int ret = starpu_task_submit(task);
      if (ret != 0)
			{
        std::cerr << "Task submission error : " << ret << std::endl;
			}
      starpu_task_wait_for_all();
      std::cout << "end" << std::endl;
      vec.cleanup();      
	  }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}