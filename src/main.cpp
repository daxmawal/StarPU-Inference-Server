#include "starpu_setup.hpp"
#include "args_parser.hpp"
#include <torch/torch.h>
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

  std::cout << "Scheduler  : " << opts.scheduler << "\n";
  std::cout << "Iterations : " << opts.iterations << "\n";

  try
  {
    StarPUSetup starpu(opts.scheduler.c_str());

    for (int i = 0; i < opts.iterations; ++i)
    {
      // Créer un tenseur d’entrée arbitraire
      if (opts.input_shape.empty()) {
        std::cerr << "Error: you must provide --shape for the input tensor.\n";
        return 1;
      }
      torch::Tensor input_tensor = torch::rand(opts.input_shape);
      float* input_ptr = input_tensor.data_ptr<float>();
      int64_t input_size = input_tensor.numel();
      
      // Enregistrer le buffer dans StarPU
      starpu_data_handle_t input_handle;
      starpu_variable_data_register(&input_handle, STARPU_MAIN_RAM,
        reinterpret_cast<uintptr_t>(input_ptr), input_size * sizeof(float));
      
      // Préparer les arguments pour la codelet
      InferenceParams* args = new InferenceParams();
      std::strncpy(args->model_path, opts.model_path.c_str(), sizeof(args->model_path));
      args->input_size = input_size;

      auto sizes = input_tensor.sizes();
      args->ndims = sizes.size();
      if (args->ndims > 8) {
        std::cerr << "Error: the tensor has more than 8 dimensions, which is not supported" << std::endl;
        return 1;
      }
      for (int i = 0; i < args->ndims; ++i) {
        args->dims[i] = sizes[i];
      }

      // Créer et configurer la tâche
      struct starpu_task* task = starpu_task_create();
      task->handles[0] = input_handle;
      task->nbuffers = 1;
      task->cl = starpu.codelet();
      task->synchronous = 1;
      task->cl_arg = args;
      task->cl_arg_size = sizeof(InferenceParams);
      task->cl_arg_free = 1;

      // Soumettre la tâche
      int ret = starpu_task_submit(task);
      if (ret != 0)
      {
        std::cerr << "Task submission error: " << ret << std::endl;
      }

      starpu_task_wait_for_all();

      // Nettoyage
      starpu_data_unregister(input_handle);
      std::cout << "End of iteration " << i << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
