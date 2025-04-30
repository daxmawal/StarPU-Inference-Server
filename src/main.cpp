#include <torch/script.h>
#include <starpu.h>

#include <iostream>
#include <memory>

void inference_cpu(void* buffers[], void* cl_arg) {
    auto* module = static_cast<torch::jit::script::Module*>(cl_arg);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}));

    at::Tensor output = module->forward(inputs).toTensor();

    std::cout << "Sortie (5 premières valeurs):\n"
              << output.slice(1, 0, 5) << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <chemin_model.pt>" << std::endl;
        return 1;
    }

    starpu_init(nullptr);

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Erreur : impossible de charger le modèle TorchScript.\n";
        return 1;
    }

    struct starpu_codelet cl;
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = inference_cpu;
    cl.nbuffers = 0; 
    cl.modes[0] = STARPU_R;
    cl.name = "torch_inference";

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl;
    task->cl_arg = &module;
    task->cl_arg_size = sizeof(torch::jit::script::Module);
    task->cl_arg_free = 0;

    int ret = starpu_task_submit(task);
    if (ret != 0) {
        std::cerr << "Erreur : échec de soumission StarPU\n";
    }

    starpu_task_wait_for_all();

    starpu_shutdown();
    return 0;
}
