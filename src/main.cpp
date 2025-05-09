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
      struct starpu_task* task = starpu_task_create();
      task->cl = starpu.codelet();

      int ret = starpu_task_submit(task);
      if (ret != 0)
			{
        std::cerr << "Task submission error : " << ret << std::endl;
			}
      starpu_task_wait_for_all();
	  }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
