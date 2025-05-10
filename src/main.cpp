#include "starpu_setup.hpp"
#include "args_parser.hpp"
#include <iostream>

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
      float* output_buffer = (float*)malloc(N * sizeof(float));
      starpu_data_handle_t output_handle;
      starpu_variable_data_register(&output_handle, STARPU_MAIN_RAM, (uintptr_t)output_buffer, N * sizeof(float));      

      struct starpu_task* task = starpu_task_create();
      task->handles[0] = output_handle;
      task->cl = starpu.codelet();
      task->cl_arg = (void*)opts.model_path.c_str();
      task->cl_arg_size = opts.model_path.size() + 1;

      int ret = starpu_task_submit(task);
      if (ret != 0)
			{
        std::cerr << "Task submission error : " << ret << std::endl;
			}
      starpu_task_wait_for_all();
      /*for (int i = 0; i < N; ++i) 
      {
        std::cout << output_buffer[i] << " ";
      }*/
      std::cout << std::endl;
	  }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}